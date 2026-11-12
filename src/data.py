from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from .schema import (
    assert_schema, diagnosis_target_cols, diagnosis_mask_cols,
    evidence_target_cols, evidence_mask_cols,
)

def make_transform(size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

class ExamBagDataset(Dataset):
    def __init__(
        self, bag_csv, image_csv, images_root, image_size=224,
        max_images=8, train=False, seed=2026, limit=None,
    ):
        self.bags = pd.read_csv(bag_csv)
        assert_schema(self.bags)
        self.images = pd.read_csv(image_csv)
        self.images_root = Path(images_root)
        self.max_images = int(max_images)
        self.train = bool(train)
        self.seed = int(seed)
        self.transform = make_transform(image_size, train)
        self.groups = {str(k): g.copy() for k,g in self.images.groupby("bag_id")}
        if limit:
            self.bags = self.bags.iloc[:int(limit)].copy()

    def __len__(self):
        return len(self.bags)

    def _resolve_image(self, row):
        # Archive path is <makcb>/<relative_path>.
        rel = str(row["relative_path"]).replace("\\", "/")
        return self.images_root / str(row["makcb"]) / rel

    def __getitem__(self, idx):
        row = self.bags.iloc[idx]
        bag_id = str(row["bag_id"])
        g = self.groups.get(bag_id)
        if g is None or len(g) == 0:
            raise RuntimeError(f"No image rows for bag_id={bag_id}")

        recs = list(g.to_dict("records"))
        if len(recs) > self.max_images:
            if self.train:
                rng = random.Random(self.seed + idx)
                recs = rng.sample(recs, self.max_images)
            else:
                recs = recs[:self.max_images]

        ims = []
        paths = []
        for r in recs:
            p = self._resolve_image(r)
            if not p.exists():
                raise FileNotFoundError(f"Missing image: {p}")
            with Image.open(p) as im:
                im = im.convert("RGB")
                ims.append(self.transform(im))
            paths.append(str(p))

        images = torch.stack(ims)
        y = pd.to_numeric(row[diagnosis_target_cols()], errors="coerce").fillna(0).to_numpy(np.float32)
        ymask = pd.to_numeric(row[diagnosis_mask_cols()], errors="coerce").fillna(0).to_numpy(np.float32)
        e = pd.to_numeric(row[evidence_target_cols()], errors="coerce").fillna(0).to_numpy(np.float32)
        emask = pd.to_numeric(row[evidence_mask_cols()], errors="coerce").fillna(0).to_numpy(np.float32)

        return {
            "bag_id": bag_id,
            "makcb": str(row["makcb"]),
            "images": images,
            "n_images": images.shape[0],
            "image_paths": paths,
            "targets": torch.from_numpy(y),
            "target_mask": torch.from_numpy(ymask),
            "evidence": torch.from_numpy(e),
            "evidence_mask": torch.from_numpy(emask),
            "teacher_findings_text": "" if pd.isna(row["teacher_findings_text"]) else str(row["teacher_findings_text"]),
            "label_conclusion_text": "" if pd.isna(row["label_conclusion_text"]) else str(row["label_conclusion_text"]),
            "text_at_inference_allowed": int(row["text_at_inference_allowed"]) if "text_at_inference_allowed" in row and not pd.isna(row["text_at_inference_allowed"]) else 0,
        }

def collate_exam_bags(batch):
    b = len(batch)
    max_n = max(x["n_images"] for x in batch)
    c,h,w = batch[0]["images"].shape[1:]
    images = torch.zeros((b,max_n,c,h,w), dtype=batch[0]["images"].dtype)
    view_mask = torch.zeros((b,max_n), dtype=torch.bool)
    for i,x in enumerate(batch):
        n=x["n_images"]
        images[i,:n]=x["images"]
        view_mask[i,:n]=True
    return {
        "bag_id":[x["bag_id"] for x in batch],
        "makcb":[x["makcb"] for x in batch],
        "images":images,
        "view_mask":view_mask,
        "targets":torch.stack([x["targets"] for x in batch]),
        "target_mask":torch.stack([x["target_mask"] for x in batch]),
        "evidence":torch.stack([x["evidence"] for x in batch]),
        "evidence_mask":torch.stack([x["evidence_mask"] for x in batch]),
        "teacher_findings_text":[x["teacher_findings_text"] for x in batch],
        "label_conclusion_text":[x["label_conclusion_text"] for x in batch],
        "text_at_inference_allowed":torch.tensor([x["text_at_inference_allowed"] for x in batch]),
    }
