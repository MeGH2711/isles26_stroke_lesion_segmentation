# Cross-Validation & Ensemble Guide

## Validation Strategy
The dataset uses a 5-fold cross-validation scheme.

## Cross-Model Ensembling
Standard `nnUNetv2_find_best_configuration` evaluates Cartesian products of plans and trainers. For custom plans (such as `nnUNetResEncUNetMPlans` combined with standard `nnUNetPlans`), use `src/manual_cross_ensemble.py`.

### Running Evaluation:
```bash
python src/manual_cross_ensemble.py
```

### Key Parameters:
- **ResEnc Weight**: 0.65
- **Baseline Weight**: 0.35
- **Decision Threshold**: $T = 0.45$
- **Min Cluster Size (26-connectivity)**: 8 voxels
