from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

DIAGNOSES = [
    "hepatic_steatosis",
    "urinary_tract_stone",
    "renal_cyst",
    "gallbladder_polyp",
    "hydronephrosis",
    "cholelithiasis",
    "hepatic_cyst",
]

EVIDENCE = [
    "liver_increased_echogenicity",
    "liver_posterior_attenuation",
    "liver_vessel_blurring",
    "hepatorenal_contrast_increased",
    "urinary_stone_present",
    "urinary_stone_shadow",
    "urinary_twinkle_artifact",
    "collecting_system_dilatation",
    "ureteral_dilatation",
    "renal_cystic_lesion",
    "renal_cyst_anechoic",
    "renal_cyst_thin_wall",
    "renal_cyst_posterior_enhancement",
    "gallstone_present",
    "gallstone_shadow",
    "gallstone_mobility",
    "gallbladder_polyp_present",
    "gallbladder_polyp_nonshadowing",
    "gallbladder_polyp_nonmobile",
    "hepatic_cystic_lesion",
    "hepatic_cyst_anechoic",
    "hepatic_cyst_thin_wall",
    "hepatic_cyst_posterior_enhancement",
    "gallbladder_wall_thickening",
    "bile_duct_dilatation",
    "ascites",
]

def diagnosis_target_cols():
    return [f"diag_target_{x}" for x in DIAGNOSES]

def diagnosis_mask_cols():
    return [f"diag_loss_mask_{x}" for x in DIAGNOSES]

def evidence_target_cols():
    return [f"evidence_target_{x}" for x in EVIDENCE]

def evidence_mask_cols():
    return [f"evidence_loss_mask_{x}" for x in EVIDENCE]

def assert_schema(df: pd.DataFrame) -> None:
    required = ["bag_id", "makcb", "teacher_findings_text", "label_conclusion_text"]
    required += diagnosis_target_cols() + diagnosis_mask_cols()
    required += evidence_target_cols() + evidence_mask_cols()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns ({len(missing)}): {missing[:20]}")
