from __future__ import annotations
from pathlib import Path
import pandas as pd

def adjudication_complete(path: Path) -> bool:
    if not path.exists():
        return False
    df = pd.read_csv(path)
    adjudicated = [c for c in df.columns if c.startswith("adjudicated_")]
    if not adjudicated:
        return False
    return bool(df[adjudicated].notna().all().all())

def patient_independence_confirmed(bag_csv: Path) -> bool:
    df = pd.read_csv(bag_csv, usecols=lambda c: c == "patient_level_independence_confirmed")
    if "patient_level_independence_confirmed" not in df.columns:
        return False
    vals = set(df["patient_level_independence_confirmed"].dropna().astype(str).str.upper())
    return vals == {"YES"}

def infer_text_policy(bag_csv: Path) -> str:
    df = pd.read_csv(bag_csv, usecols=lambda c: c == "text_at_inference_allowed")
    if "text_at_inference_allowed" not in df.columns:
        return "teacher_only"
    vals = set(pd.to_numeric(df["text_at_inference_allowed"], errors="coerce").dropna().astype(int))
    return "inference_allowed" if vals == {1} else "teacher_only"

def test_label_status(adjudication_csv: Path, test_bags_csv: Path) -> dict:
    return {
        "adjudication_complete": adjudication_complete(adjudication_csv),
        "patient_independence_confirmed": patient_independence_confirmed(test_bags_csv),
        "text_policy": infer_text_policy(test_bags_csv),
    }

def final_claim_allowed(status: dict) -> bool:
    return bool(status["adjudication_complete"] and status["patient_independence_confirmed"])

def result_label(status: dict) -> str:
    return "FINAL_CLINICAL_TEST" if final_claim_allowed(status) else "PRELIMINARY_EXAM_LEVEL_TEST"
