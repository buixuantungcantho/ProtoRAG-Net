# Paper 4: protorag_net — Research code release candidate

**Status: code-only package; not validated as an end-to-end reproducible public release.**

This repository candidate is separated from the five-paper abdominal-ultrasound research code.
No clinical images, clinical reports, real patient/examination identifiers, prediction CSV files,
original checkpoints, or private datasets have been included.

## Provenance

- Paper 1–4 Python code comes from the `PAPER2_FULL.zip` bundle's shared canonical `src/` directory.
- Paper 5 Python code comes from `PAPER5_FULL.zip`.
- Paper 1/3/4 model YAML parameters were **reconstructed** from the matching `resolved_config.json` in
  `PAPER5_FULL.zip` (seed 2026), so they should be reconciled with the original canonical model YAML.
- Paper 2 and 5 model-specific YAML are directly copied from their source bundles.
- No claim is made that these code-only archives reproduce the published numbers without the original data,
  checkpoints, prediction exports, environment, and a successful end-to-end rerun.

## Setup

1. Create a Python environment and install `pip install -r requirements.txt`.
2. Set authorized paths in `configs/base.yaml`; check image/model settings and release schema.
3. Obtain the dataset only through the authorized research data custodian.
4. Run from this repository root:

```bash
python -m src.train_one --model protorag_net --config configs/paper4_protorag_net.yaml --seed 2026
```

## Paper 5 compatibility

Paper 5 needs the **same exam-level split and aligned per-seed validation/test predictions** from
MM-MIL-CA, RG-HGNN, MV-ConsMIL and ProtoRAG-Net. Do not use a randomly generated or modified
prediction CSV to assert scientific results.

## Public release checklist

- [ ] Confirm institutional/IRB and data-controller permissions for sharing any clinical records
- [ ] Independently inspect all metadata and output files for identifiers and quasi-identifiers
- [ ] Review patient/exam key pseudonymization and image burned-in annotations
- [ ] Check original trained code/checkpoint hash and exact versions/configuration
- [ ] Run from scratch on authorized data and compare metrics to the paper
- [ ] Add author-approved software license and citation metadata
- [ ] Verify correctness with all coauthors before public push

This bundle is provided as a research-code candidate without a software license. Do not infer
open-source licensing merely from a public GitHub upload.
