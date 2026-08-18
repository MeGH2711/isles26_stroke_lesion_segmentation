# ISLES 2026: Stroke Lesion Segmentation

Deep learning pipeline for the **ISLES 2026 (Ischemic Stroke Lesion Segmentation) Challenge**, powered by optimized nnU-Net v2 architectures, weighted cross-validation ensembling, connected-component post-processing, and production-ready Grand Challenge Docker containers.

---

## 🌟 Overview & Key Features

- **Multi-Model Weighted Softmax Ensemble**: 
  - 65% Residual Encoder UNet (ResEnc-M Plans: `nnUNetTrainerWithLogging__nnUNetResEncUNetMPlans__3d_fullres`)
  - 35% Baseline 3D full-resolution nnU-Net (`nnUNetTrainer__nnUNetPlans__3d_fullres`)
  - 10 folds combined for inference stability.
- **Empirical Lesion Boundary Calibration**: Optimized probability threshold ($T = 0.45$) for capturing subtle ischemic stroke lesion boundaries.
- **Noise Suppression**: 26-connectivity 3D connected component filter removing spurious clusters $< 8$ voxels.
- **Grand Challenge Docker Architecture**: Standardized web API container (`/health` and `/invoke`) compliant with Grand Challenge evaluation protocol, outputting:
  - Binary stroke lesion segmentation mask (`.mha`)
  - Continuous lesion probability map (`.mha`)
- **Automated Checkpoint Stripping & Packaging**: Automated weight optimization stripping training optimizer states to reduce model bundle size by ~50%.

---

## 📂 Repository Structure

```
├── isles_docker_v3/             # Latest production ensemble Docker container (v3)
│   ├── Dockerfile               # Linux AMD64 PyTorch 2.13 / CUDA 12.6 container definition
│   ├── app.py                   # FastAPI server (/health, /invoke)
│   ├── inference.py             # 10-fold weighted ensemble inference & post-processing
│   ├── export_clean_weights.py  # Checkpoint optimizer state stripper
│   ├── package_model_tar.py     # Model packaging script for Grand Challenge
│   ├── save_docker_image.py     # Docker export helper
│   ├── requirements.txt         # Container runtime Python requirements
│   ├── custom_trainers/         # Custom nnU-Net trainer definitions
│   └── walkthrough.md           # Docker submission walkthrough
├── isles26_docker_v2/           # Docker submission iteration v2
├── isles26_docker/              # Docker submission iteration v1
├── src/
│   ├── manual_cross_ensemble.py # Validation fold evaluation & cross-model grid search
│   ├── patch_component_diagnostic.py # Component size diagnostic utility
│   ├── csvtojson/               # Conversion utilities
│   └── summary/                 # Cross-validation metrics aggregation
├── utils/
│   ├── csvtojson.py             # CSV to JSON conversion script
│   └── txtlogstocsv.py          # Training log parser to structured CSV
├── docs/                        # Setup guides and workflow documentation
├── sync_weights.py              # Cloud / Rclone automated checkpoint synchronization
├── requirements.txt             # Full development and training dependencies
└── .gitignore                   # Clean ignore rules excluding heavy data & weights
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

- Python 3.11+
- CUDA-compatible GPU (CUDA 11.8 / 12.x)
- Docker Desktop (for container deployment)

```bash
# Clone the repository
git clone https://github.com/MeGH2711/isles26_stroke_lesion_segmentation.git
cd isles26_stroke_lesion_segmentation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables for nnU-Net

Set the paths for nnU-Net v2 data folders:

```powershell
# PowerShell
$env:nnUNet_raw          = "path/to/data/nnunet_raw"
$env:nnUNet_preprocessed = "path/to/data/nnunet_preprocessed"
$env:nnUNet_results      = "path/to/data/nnunet_results"
```

```bash
# Bash / Linux
export nnUNet_raw="/path/to/data/nnunet_raw"
export nnUNet_preprocessed="/path/to/data/nnunet_preprocessed"
export nnUNet_results="/path/to/data/nnunet_results"
```

---

## 🐳 Docker Container Build & Submission (Grand Challenge)

Navigate to `isles_docker_v3/`:

```bash
cd isles_docker_v3
```

### Build the Docker Image
```bash
bash do_build.sh
```
Or with PowerShell:
```powershell
docker build --platform=linux/amd64 --tag "isles26-ensemble-algorithm-v3" .
```

### Run Local End-to-End Test
```bash
bash do_test_run.sh
```

### Package Container & Model Weights for Upload
```bash
# Package Docker image
bash do_save.sh

# Package model weights
python package_model_tar.py
```

Upload `isles26-ensemble-algorithm-v3.tar.gz` and `model.tar.gz` to the [Grand Challenge Algorithm Portal](https://grand-challenge.org/).

---

## 📊 Cross-Validation & Ensembling Analysis

Run the cross-model ensemble evaluation across all 5 folds:

```bash
python src/manual_cross_ensemble.py
```

Analyze lesion size stratification and component metrics:
```bash
python src/patch_component_diagnostic.py
```

---

## 📄 Citation & Acknowledgements

- **ISLES 2026 Challenge**: Ischemic Stroke Lesion Segmentation Challenge.
- **nnU-Net**: Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods.
- **ATLAS v2 / v3**: Anatomical Tracings of Lesions After Stroke.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
