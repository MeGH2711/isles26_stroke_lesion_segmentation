# ISLES 2026 Docker Submission v3 — Walkthrough

## What Was Created in `isles_docker_v3/`

This submission package incorporates all empirical leaderboard optimizations validated on the 5-fold cross-validation cohort:
- **Weighted Softmax Ensemble**: 65% ResEnc-M + 35% Baseline (10 folds total)
- **Optimal Probability Threshold**: $T = 0.45$ (captures subtle ischemic stroke lesion boundaries)
- **Connected Component Filtering**: 26-connectivity filter removing isolated noise clusters $< 8$ voxels
- **Stripped Inference Weights**: Clean checkpoints (50% size reduction, ~2.36 GB model tarball)
- **Dual Outputs**:
  - `stroke-lesion-segmentation/<uuid>.mha` (uint8 binary mask)
  - `lesion-probability-map/<uuid>.mha` (float32 continuous probability)

---

### File Overview

| File | Purpose |
|---|---|
| [Dockerfile](Dockerfile) | Docker build config — PyTorch 2.13 + CUDA 12.6 runtime base |
| [app.py](app.py) | FastAPI HTTP server (`/health` + `/invoke` on port 4743) |
| [inference.py](inference.py) | **Core logic** — weighted ensembling ($0.65/0.35$), $T=0.45$, 26-conn component filter |
| [requirements.txt](requirements.txt) | Dependencies (`nnunetv2`, `SimpleITK`, `scikit-image`, `fastapi`, `uvicorn`, `numpy`) |
| [custom_trainers/logging_trainer.py](custom_trainers/logging_trainer.py) | Custom trainer class definition for ResEnc compatibility |
| [export_clean_weights.py](export_clean_weights.py) | Strips optimizer states from training checkpoints into `model/` |
| [package_model_tar.py](package_model_tar.py) | Creates `model.tar.gz` with version metadata for Grand Challenge upload |
| [do_build.sh](do_build.sh) | Builds Docker image `isles26-ensemble-algorithm-v3` |
| [do_test_run.sh](do_test_run.sh) | End-to-end local test run with staging, health check, invoke, and output verification |
| [do_save.sh](do_save.sh) | Exports Docker image as `isles26-ensemble-algorithm-v3.tar.gz` |
| `model.tar.gz` | Ready-to-upload model tarball (2.36 GB) |

---

## Step-by-Step Submission Guide

### Step 1: Build Docker Image Locally

Make sure **Docker Desktop** is running, then run from bash (or WSL / Git Bash):

```bash
cd isles_docker_v3
bash do_build.sh
```

Or in PowerShell:
```powershell
docker build --platform=linux/amd64 --tag "isles26-ensemble-algorithm-v3" "c:\Users\Admin\Desktop\meghpatel\isles26\isles_docker_v3"
```

---

### Step 2: Run End-to-End Local Test

```bash
cd isles_docker_v3
bash do_test_run.sh
```

This verifies:
1. Container boots up and `/health` returns 200 within seconds.
2. Models load on CUDA GPU.
3. `/invoke` executes weighted inference on sample input `r001s001_0000.nii.gz`.
4. Outputs are saved to `test/output/interf0/` in `.mha` format.

---

### Step 3: Save Algorithm Image for Grand Challenge

```bash
cd isles_docker_v3
bash do_save.sh
```

This generates `isles26-ensemble-algorithm-v3.tar.gz`.

---

### Step 4: Upload to Grand Challenge

1. **Upload Algorithm Container**:
   - Go to your algorithm page on [Grand Challenge](https://grand-challenge.org/).
   - Upload `isles26-ensemble-algorithm-v3.tar.gz` as a new Container image.
2. **Upload Model Weights**:
   - Go to **Your algorithm → Models**.
   - Upload `model.tar.gz` (already generated in `isles_docker_v3/model.tar.gz`).
3. **Submit to Preliminary Evaluation Phase**:
   - Run the sanity check on test cases `sub-r001s001` and `sub-soop0468`.
   - Ensure a **GPU** is requested for the 10-fold ensemble.
4. **Submit to Final Test Phase**:
   - Once Preliminary Evaluation succeeds, submit to the Final Test Phase before August 20th.
