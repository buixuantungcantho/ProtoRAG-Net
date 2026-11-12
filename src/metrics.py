from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss

def sigmoid(x):
    x=np.asarray(x,dtype=float); x=np.clip(x,-50,50); return 1/(1+np.exp(-x))

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7); return np.log(p/(1-p))

def _finite(y,p,m,j):
    return (m[:,j]>0)&np.isfinite(y[:,j])&np.isfinite(p[:,j])

def select_thresholds(y,p,mask,default=0.5):
    th=[]
    for j in range(y.shape[1]):
        ok=_finite(y,p,mask,j)
        if ok.sum()<5 or len(np.unique(y[ok,j]))<2: th.append(default); continue
        best=(default,-1)
        for t in np.linspace(.05,.95,91):
            f=f1_score(y[ok,j],(p[ok,j]>=t).astype(int),zero_division=0)
            if f>best[1]: best=(float(t),float(f))
        th.append(best[0])
    return np.array(th)

def expected_calibration_error(y,p,mask,n_bins=15):
    vals=[]
    for j in range(y.shape[1]):
        ok=_finite(y,p,mask,j)
        if ok.sum()==0: continue
        yy=y[ok,j].astype(int); pp=p[ok,j]
        e=0.0
        for lo,hi in zip(np.linspace(0,1,n_bins+1)[:-1],np.linspace(0,1,n_bins+1)[1:]):
            sel=(pp>=lo)&(pp<(hi if hi<1 else 1.000001))
            if sel.any(): e += sel.mean()*abs(pp[sel].mean()-yy[sel].mean())
        vals.append(e)
    return float(np.mean(vals)) if vals else float('nan')

def compute_metrics(y,p,mask,thresholds):
    per=[]
    for j in range(y.shape[1]):
        ok=_finite(y,p,mask,j); yy=y[ok,j].astype(int); pp=p[ok,j]
        if len(yy)==0:
            per.append(dict(label=j,n=0,auroc=np.nan,auprc=np.nan,f1=np.nan,sensitivity=np.nan,specificity=np.nan,brier=np.nan)); continue
        pred=(pp>=thresholds[j]).astype(int)
        if len(np.unique(yy))>=2:
            auroc=roc_auc_score(yy,pp); auprc=average_precision_score(yy,pp)
        else: auroc=np.nan; auprc=np.nan
        tp=((pred==1)&(yy==1)).sum(); tn=((pred==0)&(yy==0)).sum(); fp=((pred==1)&(yy==0)).sum(); fn=((pred==0)&(yy==1)).sum()
        per.append(dict(label=j,n=int(ok.sum()),auroc=float(auroc),auprc=float(auprc),f1=float(f1_score(yy,pred,zero_division=0)),
                        sensitivity=float(tp/max(1,tp+fn)),specificity=float(tn/max(1,tn+fp)),brier=float(brier_score_loss(yy,pp))))
    keys=['auroc','auprc','f1','sensitivity','specificity','brier']
    macro={k:float(np.nanmean([r[k] for r in per])) for k in keys}
    macro['ece']=expected_calibration_error(y,p,mask)
    return {'macro':macro,'per_label':per}

def bootstrap_macro_ci(y,p,mask,thresholds,n_resamples=1000,seed=2026):
    rng=np.random.default_rng(seed); n=len(y); vals={k:[] for k in ['auroc','auprc','f1','brier','ece']}
    for _ in range(int(n_resamples)):
        ix=rng.integers(0,n,n); m=compute_metrics(y[ix],p[ix],mask[ix],thresholds)['macro']
        for k in vals:
            if np.isfinite(m[k]): vals[k].append(m[k])
    return {k:{'mean':float(np.mean(v)),'low95':float(np.percentile(v,2.5)),'high95':float(np.percentile(v,97.5))} if v else None for k,v in vals.items()}
