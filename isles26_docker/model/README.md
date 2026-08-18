# Model Weights Directory

This directory should contain the nnU-Net model checkpoints for both the
**Baseline** and **ResEnc** models, structured exactly as nnU-Net expects.

## Required Structure

```
model/
└── Dataset001_ISLES26/
    ├── nnUNetTrainer__nnUNetPlans__3d_fullres/
    │   ├── fold_0/checkpoint_final.pth
    │   ├── fold_1/checkpoint_final.pth
    │   ├── fold_2/checkpoint_final.pth
    │   ├── fold_3/checkpoint_final.pth
    │   ├── fold_4/checkpoint_final.pth
    │   ├── dataset.json
    │   ├── plans.json
    │   ├── postprocessing.pkl
    │   └── postprocessing.json
    └── nnUNetTrainerWithLogging__nnUNetResEncUNetMPlans__3d_fullres/
        ├── fold_0/checkpoint_final.pth
        ├── fold_1/checkpoint_final.pth
        ├── fold_2/checkpoint_final.pth
        ├── fold_3/checkpoint_final.pth
        ├── fold_4/checkpoint_final.pth
        ├── dataset.json
        ├── plans.json
        ├── postprocessing.pkl
        └── postprocessing.json
```

## How to populate

Run the `copy_weights.ps1` script from the `isles26_docker/` directory, or
copy manually from your training results:

```powershell
.\copy_weights.ps1
```

## For Grand Challenge Upload

When ready, create a tarball of this directory's *contents*:

```bash
cd model
tar -czf ../model.tar.gz .
```

Upload `model.tar.gz` via **Your algorithm → Models** on Grand Challenge.
Grand Challenge will extract it to `/opt/ml/model/` inside the container.
