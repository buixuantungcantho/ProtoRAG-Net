from __future__ import annotations
import time, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from .losses import total_loss
from .metrics import sigmoid, select_thresholds, compute_metrics, bootstrap_macro_ci
from .utils import save_json, sha256_file


def move_batch(batch,device):
    for k in ['images','view_mask','targets','target_mask','evidence','evidence_mask','text_at_inference_allowed']:
        batch[k]=batch[k].to(device,non_blocking=True)
    return batch

def autocast_ctx(cfg, device):
    use=bool(cfg['training'].get('amp',True)) and device.type=='cuda'
    dtype=torch.bfloat16 if str(cfg['training'].get('amp_dtype','bfloat16')).lower()=='bfloat16' else torch.float16
    return torch.amp.autocast(device_type='cuda',dtype=dtype,enabled=use)

def _apply_view_fraction(mask, fraction):
    if fraction>=0.999: return mask
    out=torch.zeros_like(mask)
    for i in range(mask.size(0)):
        idx=torch.where(mask[i])[0]
        k=max(1,int(math.ceil(len(idx)*fraction)))
        out[i,idx[:k]]=True
    return out

def predict(model, loader, device, model_name, cfg, view_keep_fraction=1.0, ablation=None):
    model.eval(); rows=[]; ys=[]; ms=[]; logits_all=[]; es=[]; ems=[]; ep=[]
    start=time.time()
    with torch.no_grad():
        for batch in tqdm(loader,desc='predict',leave=False):
            ids=batch['bag_id']; makcb=batch['makcb']; batch=move_batch(batch,device)
            vm=_apply_view_fraction(batch['view_mask'],float(view_keep_fraction))
            kw={'teacher_findings_text':None,'training_with_text':False}
            if ablation=='disable_graph': kw['disable_graph']=True
            if ablation=='disable_retrieval': kw['disable_retrieval']=True
            with autocast_ctx(cfg,device): out=model(batch['images'],vm,**kw)
            lg=out['logits'].float().cpu().numpy()
            if not np.isfinite(lg).all(): raise FloatingPointError(f'Non-finite logits during prediction: {model_name}')
            logits_all.append(lg); ys.append(batch['targets'].cpu().numpy()); ms.append(batch['target_mask'].cpu().numpy())
            if 'evidence_logits' in out:
                eprob=torch.sigmoid(out['evidence_logits']).float().cpu().numpy(); ep.append(eprob)
                es.append(batch['evidence'].cpu().numpy()); ems.append(batch['evidence_mask'].cpu().numpy())
            for i,bid in enumerate(ids): rows.append({'bag_id':bid,'makcb':makcb[i]})
    logits=np.concatenate(logits_all); y=np.concatenate(ys); m=np.concatenate(ms); p=sigmoid(logits)
    for i,r in enumerate(rows):
        for j in range(p.shape[1]):
            r[f'logit_{j}']=float(logits[i,j]); r[f'prob_{j}']=float(p[i,j]); r[f'target_{j}']=float(y[i,j]); r[f'mask_{j}']=float(m[i,j])
    extra={}
    if ep:
        eprob=np.concatenate(ep); ey=np.concatenate(es); em=np.concatenate(ems)
        for i,r in enumerate(rows):
            for j in range(eprob.shape[1]):
                r[f'evidence_prob_{j}']=float(eprob[i,j]); r[f'evidence_target_{j}']=float(ey[i,j]); r[f'evidence_mask_{j}']=float(em[i,j])
        extra={'evidence_y':ey,'evidence_mask':em,'evidence_p':eprob}
    return pd.DataFrame(rows), y, m, p, time.time()-start, extra

def _live(out_dir, payload):
    p=Path(out_dir)/'LIVE_PROGRESS.json'; p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); root=Path(out_dir).parents[1]; (root/'LIVE_TRAINING_PROGRESS.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')

def train_model(model, train_loader, val_loader, device, model_name, cfg, out_dir, progress_tracker=None):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['lr']),weight_decay=float(cfg['training']['weight_decay']))
    best=np.inf; wait=0; history=[]; total_epochs=int(cfg['training']['epochs'])
    for epoch in range(1,total_epochs+1):
        model.train(); losses=[]; t0=time.time(); parts_sum={}; n_batches=len(train_loader)
        for bi,batch in enumerate(tqdm(train_loader,desc=f'{model_name} e{epoch}',leave=False),1):
            batch=move_batch(batch,device); opt.zero_grad(set_to_none=True)
            use_text=(model_name=='mm_mil_ca' and cfg['scientific_policy']['text_mode']=='teacher_only')
            with autocast_ctx(cfg,device):
                out=model(batch['images'],batch['view_mask'],teacher_findings_text=batch['teacher_findings_text'],training_with_text=use_text,
                          view_dropout=float(cfg.get('view_dropout',0.0)),retrieval_targets=batch['targets'],retrieval_masks=batch['target_mask'],
                          retrieval_update=(model_name=='protorag_net'))
                loss,parts=total_loss(model_name,out,batch,cfg)
            if not torch.isfinite(loss): raise FloatingPointError(f'Non-finite loss at {model_name} epoch={epoch} batch={bi}')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip'])); opt.step()
            losses.append(float(loss.detach()))
            for k,v in parts.items(): parts_sum[k]=parts_sum.get(k,0.0)+float(v)
            if bi==1 or bi%int(cfg.get('orchestration',{}).get('progress_every_batches',25))==0:
                _live(out_dir,{'model':model_name,'epoch':epoch,'epochs_planned':total_epochs,'batch':bi,'batches':n_batches,'train_loss_running':float(np.mean(losses)),'updated':time.time()})
        val_df,y,m,p,_,extra=predict(model,val_loader,device,model_name,cfg)
        thresholds=select_thresholds(y,p,m,float(cfg['training']['default_threshold'])); met=compute_metrics(y,p,m,thresholds)
        evidence_info=None
        if extra:
            eth=select_thresholds(extra['evidence_y'],extra['evidence_p'],extra['evidence_mask'],0.5)
            evidence_info={'thresholds':eth.tolist(),'metrics':compute_metrics(extra['evidence_y'],extra['evidence_p'],extra['evidence_mask'],eth)}
        val_score=-met['macro']['auprc']; rec={'epoch':epoch,'train_loss':float(np.mean(losses)),'loss_parts':{k:v/max(1,n_batches) for k,v in parts_sum.items()},'val':met['macro'],'seconds':time.time()-t0}
        history.append(rec); save_json(history,out_dir/'history.json')
        if progress_tracker is not None: progress_tracker.epoch(epoch,total_epochs,rec['seconds'],message=f"{model_name}: epoch {epoch}, val macro-AUPRC={met['macro']['auprc']:.4f}")
        if val_score<best:
            best=val_score; wait=0
            ck={'model':model.state_dict(),'epoch':epoch,'thresholds':thresholds.tolist(),'val_metrics':met,'evidence_validation':evidence_info}
            torch.save(ck,out_dir/'best.pt'); val_df.to_csv(out_dir/'val_predictions_best.csv',index=False); save_json({'thresholds':thresholds.tolist()},out_dir/'thresholds.json')
            if evidence_info: save_json(evidence_info,out_dir/'evidence_validation_best.json')
        else:
            wait+=1
            if wait>=int(cfg['training']['patience']): break
    (out_dir/'best.sha256').write_text(sha256_file(out_dir/'best.pt')+'\n',encoding='utf-8')
    return out_dir/'best.pt'

def evaluate_checkpoint(model,ckpt,loader,device,model_name,cfg,out_dir,prefix,view_keep_fraction=1.0,ablation=None):
    d=torch.load(ckpt,map_location=device); model.load_state_dict(d['model']); thresholds=np.array(d['thresholds'],dtype=float)
    pred,y,m,p,secs,extra=predict(model,loader,device,model_name,cfg,view_keep_fraction,ablation)
    met=compute_metrics(y,p,m,thresholds); obj={'metrics':met,'thresholds':thresholds.tolist(),'seconds':secs,'view_keep_fraction':view_keep_fraction,'ablation':ablation}
    if cfg.get('reporting',{}).get('bootstrap_ci') and prefix.startswith('test'):
        obj['bootstrap_95ci']=bootstrap_macro_ci(y,p,m,thresholds,int(cfg.get('bootstrap',{}).get('n_resamples',1000)),int(cfg.get('bootstrap',{}).get('seed',2026)))
    if extra:
        ev=d.get('evidence_validation') or {}; eth=np.array(ev.get('thresholds',[.5]*extra['evidence_p'].shape[1]),dtype=float)
        obj['evidence_metrics']=compute_metrics(extra['evidence_y'],extra['evidence_p'],extra['evidence_mask'],eth)
    pred.to_csv(Path(out_dir)/f'{prefix}_predictions.csv',index=False); save_json(obj,Path(out_dir)/f'{prefix}_metrics.json')
    return met

def evaluate_scientific_ablations(model,ckpt,test_loader,device,model_name,cfg,out_dir):
    if model_name=='mv_consmil':
        for f in cfg.get('robustness',{}).get('view_keep_fractions',[.75,.5,.25]):
            if float(f)>=.999: continue
            evaluate_checkpoint(model,ckpt,test_loader,device,model_name,cfg,out_dir,f'test_preliminary_viewkeep_{str(f).replace(".","p")}',float(f))
    elif model_name=='rg_hgnn':
        evaluate_checkpoint(model,ckpt,test_loader,device,model_name,cfg,out_dir,'test_preliminary_ablation_no_graph',1.0,'disable_graph')
    elif model_name=='protorag_net':
        evaluate_checkpoint(model,ckpt,test_loader,device,model_name,cfg,out_dir,'test_preliminary_ablation_no_retrieval',1.0,'disable_retrieval')
