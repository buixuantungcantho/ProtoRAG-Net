import torch
import torch.nn.functional as F

def masked_bce(logits, targets, mask):
    loss=F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    return (loss*mask).sum()/mask.sum().clamp_min(1)

def bernoulli_js_from_logits(a,b):
    # Compute probability-space JS in FP32 even when model forward uses BF16.
    # BF16 cannot reliably represent 1-1e-6, which can otherwise round to 1.0
    # and produce log(0/0) after sigmoid saturation.
    a=a.float()
    b=b.float()
    eps=1e-6
    pa=torch.sigmoid(a).clamp(eps,1.0-eps)
    pb=torch.sigmoid(b).clamp(eps,1.0-eps)
    m=(0.5*(pa+pb)).clamp(eps,1.0-eps)

    def kl(p,q):
        return (
            p*torch.log(p/q) +
            (1.0-p)*torch.log((1.0-p)/(1.0-q))
        )

    js=0.5*(kl(pa,m)+kl(pb,m)).mean()

    if not torch.isfinite(js):
        raise FloatingPointError("Non-finite Bernoulli JS consistency loss")

    return js

def total_loss(model_name, out, batch, cfg):
    loss=masked_bce(out['logits'],batch['targets'],batch['target_mask'])
    parts={'diagnosis':float(loss.detach())}
    if 'evidence_logits' in out:
        w=float(cfg.get('evidence_loss_weight',0.35)); le=masked_bce(out['evidence_logits'],batch['evidence'],batch['evidence_mask'])
        loss=loss+w*le; parts['evidence']=float(le.detach())
    if 'teacher_logits' in out:
        wt=float(cfg.get('teacher_supervision_weight',1.0))
        lt=masked_bce(out['teacher_logits'],batch['targets'],batch['target_mask'])
        loss=loss+wt*lt; parts['teacher_supervision']=float(lt.detach())
        w=float(cfg.get('distill_weight',cfg.get('text',{}).get('teacher_distill_weight',0.15)))
        with torch.no_grad(): t=torch.sigmoid(out['teacher_logits'])
        ld=F.binary_cross_entropy_with_logits(out['logits'],t)
        loss=loss+w*ld; parts['privileged_text_distill']=float(ld.detach())
    if model_name=='rg_hgnn' and 'graph_logits' in out and 'direct_logits' in out:
        w=float(cfg.get('graph_consistency_weight',0.05))
        lc=bernoulli_js_from_logits(out['graph_logits'],out['direct_logits'].detach())
        loss=loss+w*lc; parts['graph_consistency']=float(lc.detach())
    if model_name=='mv_consmil' and 'consistency_logits' in out:
        w=float(cfg.get('consistency_weight',0.15)); lc=bernoulli_js_from_logits(out['logits'],out['consistency_logits'])
        loss=loss+w*lc; parts['view_consistency_js']=float(lc.detach())
    return loss,parts
