from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .utils import load_yaml, deep_update, seed_everything, save_json, capture_environment, code_manifest
from .data import ExamBagDataset, collate_exam_bags
from .models import build_model
from .engine import train_model, evaluate_checkpoint, evaluate_scientific_ablations
from .policy import test_label_status, result_label

def paths(cfg):
    root=Path(cfg["data_root"]); rel=root/cfg["release_dir"]
    return root,rel

def run(model_name, model_cfg_path, seed, smoke=False):
    code_root=Path(__file__).resolve().parents[1]
    cfg=load_yaml(code_root/"configs"/"base.yaml")
    if model_cfg_path:
        cfg=deep_update(cfg,load_yaml(model_cfg_path))
    cfg["model_name"]=model_name
    seed_everything(seed)

    root,rel=paths(cfg)
    if cfg["scientific_policy"]["require_data_ready_lock"] and not Path(cfg["data_ready_lock"]).exists():
        raise RuntimeError("DATA_READY.lock missing; refusing to train.")
    sp=cfg["splits"]
    images_root=root/cfg["images_dir"]
    lim=16 if smoke else None
    train_ds=ExamBagDataset(rel/sp["train_bags"],rel/sp["train_images"],images_root,cfg["image"]["size"],cfg["image"]["max_images_per_exam"],True,seed,lim)
    val_ds=ExamBagDataset(rel/sp["val_bags"],rel/sp["val_images"],images_root,cfg["image"]["size"],cfg["image"]["max_images_per_exam"],False,seed,lim)
    test_ds=ExamBagDataset(rel/sp["test_bags"],rel/sp["test_images"],images_root,cfg["image"]["size"],cfg["image"]["max_images_per_exam"],False,seed,lim)

    bs=2 if smoke else int(cfg["training"]["batch_size"])
    nw=0 if smoke else int(cfg["training"]["num_workers"])
    loaders=[
        DataLoader(train_ds,bs,shuffle=True,num_workers=nw,collate_fn=collate_exam_bags,pin_memory=True),
        DataLoader(val_ds,bs,shuffle=False,num_workers=nw,collate_fn=collate_exam_bags,pin_memory=True),
        DataLoader(test_ds,bs,shuffle=False,num_workers=nw,collate_fn=collate_exam_bags,pin_memory=True),
    ]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(model_name,cfg).to(device)

    out=Path(cfg["results_root"])/model_name/f"seed_{seed}"
    if smoke: out=Path("/tmp/JOURNAL_SMOKE")/model_name
    out.mkdir(parents=True,exist_ok=True)
    save_json(cfg,out/"resolved_config.json"); capture_environment(out)
    code_manifest(code_root,out/"code_sha256.txt")

    status=test_label_status(rel/sp["adjudication"],rel/sp["test_bags"])
    save_json(status,out/"scientific_policy_status.json")
    (out/"RESULT_STATUS.txt").write_text(result_label(status)+"\n",encoding="utf-8")

    if smoke:
        cfg["training"]["epochs"]=1; cfg["training"]["patience"]=1
        # Disable pretrained download for smoke if no internet weights cache.
        # Model itself remains architecture-identical.
    ck=train_model(model,loaders[0],loaders[1],device,model_name,cfg,out)
    evaluate_checkpoint(model,ck,loaders[1],device,model_name,cfg,out,"validation")
    evaluate_checkpoint(model,ck,loaders[2],device,model_name,cfg,out,"test_preliminary")
    if not smoke:
        evaluate_scientific_ablations(model,ck,loaders[2],device,model_name,cfg,out)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--config",default=None)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    run(a.model,a.config,a.seed,a.smoke)

if __name__=="__main__": main()
