# copy_weights.ps1
# Copies nnU-Net model checkpoints from the training results directory
# into the Docker model/ directory with the correct structure.
#
# Run from the isles26_docker/ directory:
#   .\copy_weights.ps1

$ErrorActionPreference = "Stop"

# Use python export_clean_weights.py to strip optimizer states and save clean inference weights
$PYTHON = "$PSScriptRoot\..\venv\Scripts\python.exe"
if (Test-Path $PYTHON) {
    Write-Host "Running export_clean_weights.py with virtualenv Python..." -ForegroundColor Cyan
    & $PYTHON "$PSScriptRoot\export_clean_weights.py"
    exit $LASTEXITCODE
} else {
    Write-Host "Running export_clean_weights.py with system Python..." -ForegroundColor Cyan
    python "$PSScriptRoot\export_clean_weights.py"
    exit $LASTEXITCODE
}

# Copy dataset.json, plans.json, dataset_fingerprint.json
New-Item -ItemType Directory -Force -Path $DST_BASELINE | Out-Null
Copy-Item "$SRC_BASELINE\dataset.json" -Destination "$DST_BASELINE\dataset.json" -Force
Copy-Item "$SRC_BASELINE\plans.json" -Destination "$DST_BASELINE\plans.json" -Force
if (Test-Path "$SRC_BASELINE\dataset_fingerprint.json") {
    Copy-Item "$SRC_BASELINE\dataset_fingerprint.json" -Destination "$DST_BASELINE\dataset_fingerprint.json" -Force
}

# Copy postprocessing files if present
$cv_baseline = "$SRC_BASELINE\crossval_results_folds_0_1_2_3_4"
if (Test-Path "$cv_baseline\postprocessing.pkl") {
    $dst_cv = "$DST_BASELINE\crossval_results_folds_0_1_2_3_4"
    New-Item -ItemType Directory -Force -Path $dst_cv | Out-Null
    Copy-Item "$cv_baseline\postprocessing.pkl" -Destination "$dst_cv\postprocessing.pkl" -Force
    Copy-Item "$cv_baseline\postprocessing.pkl" -Destination "$DST_BASELINE\postprocessing.pkl" -Force
    if (Test-Path "$cv_baseline\postprocessing.json") {
        Copy-Item "$cv_baseline\postprocessing.json" -Destination "$dst_cv\postprocessing.json" -Force
        Copy-Item "$cv_baseline\postprocessing.json" -Destination "$DST_BASELINE\postprocessing.json" -Force
    }
    Write-Host "  Copied postprocessing files"
}
Write-Host "  Copied dataset.json, plans.json, and metadata"

# Copy fold checkpoints
for ($fold = 0; $fold -le 4; $fold++) {
    $src_fold = "$SRC_BASELINE\fold_$fold"
    $dst_fold = "$DST_BASELINE\fold_$fold"
    New-Item -ItemType Directory -Force -Path $dst_fold | Out-Null
    
    $checkpoint = "checkpoint_final.pth"
    if (Test-Path "$src_fold\$checkpoint") {
        Copy-Item "$src_fold\$checkpoint" -Destination "$dst_fold\$checkpoint" -Force
        $size = [math]::Round((Get-Item "$dst_fold\$checkpoint").Length / 1MB, 1)
        Write-Host "  fold_$fold/$checkpoint (${size} MB)" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: $src_fold\$checkpoint not found!" -ForegroundColor Yellow
    }
}

# ───────────────────────────────────────────────────────────────────
# Model 2: ResEnc (nnUNetTrainerWithLogging / nnUNetResEncUNetMPlans / 3d_fullres)
# ───────────────────────────────────────────────────────────────────
$RESENC = "nnUNetTrainerWithLogging__nnUNetResEncUNetMPlans__3d_fullres"
$SRC_RESENC = "$SRC_ROOT\$RESENC"
$DST_RESENC = "$DST_ROOT\$RESENC"

Write-Host "`n=== Copying ResEnc model ===" -ForegroundColor Cyan

# Copy dataset.json, plans.json, dataset_fingerprint.json
New-Item -ItemType Directory -Force -Path $DST_RESENC | Out-Null
Copy-Item "$SRC_RESENC\dataset.json" -Destination "$DST_RESENC\dataset.json" -Force
Copy-Item "$SRC_RESENC\plans.json" -Destination "$DST_RESENC\plans.json" -Force
if (Test-Path "$SRC_RESENC\dataset_fingerprint.json") {
    Copy-Item "$SRC_RESENC\dataset_fingerprint.json" -Destination "$DST_RESENC\dataset_fingerprint.json" -Force
}

# Copy postprocessing files if present
$cv_resenc = "$SRC_RESENC\crossval_results_folds_0_1_2_3_4"
if (Test-Path "$cv_resenc\postprocessing.pkl") {
    $dst_cv = "$DST_RESENC\crossval_results_folds_0_1_2_3_4"
    New-Item -ItemType Directory -Force -Path $dst_cv | Out-Null
    Copy-Item "$cv_resenc\postprocessing.pkl" -Destination "$dst_cv\postprocessing.pkl" -Force
    Copy-Item "$cv_resenc\postprocessing.pkl" -Destination "$DST_RESENC\postprocessing.pkl" -Force
    if (Test-Path "$cv_resenc\postprocessing.json") {
        Copy-Item "$cv_resenc\postprocessing.json" -Destination "$dst_cv\postprocessing.json" -Force
        Copy-Item "$cv_resenc\postprocessing.json" -Destination "$DST_RESENC\postprocessing.json" -Force
    }
    Write-Host "  Copied postprocessing files"
}
Write-Host "  Copied dataset.json, plans.json, and metadata"

# Copy fold checkpoints
for ($fold = 0; $fold -le 4; $fold++) {
    $src_fold = "$SRC_RESENC\fold_$fold"
    $dst_fold = "$DST_RESENC\fold_$fold"
    New-Item -ItemType Directory -Force -Path $dst_fold | Out-Null
    
    $checkpoint = "checkpoint_final.pth"
    if (Test-Path "$src_fold\$checkpoint") {
        Copy-Item "$src_fold\$checkpoint" -Destination "$dst_fold\$checkpoint" -Force
        $size = [math]::Round((Get-Item "$dst_fold\$checkpoint").Length / 1MB, 1)
        Write-Host "  fold_$fold/$checkpoint (${size} MB)" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: $src_fold\$checkpoint not found!" -ForegroundColor Yellow
    }
}

# ───────────────────────────────────────────────────────────────────
# Summary
# ───────────────────────────────────────────────────────────────────
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
$total_files = (Get-ChildItem -Path $DST_ROOT -Recurse -File).Count
$total_size = [math]::Round((Get-ChildItem -Path $DST_ROOT -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
Write-Host "  Total files: $total_files"
Write-Host "  Total size:  ${total_size} GB"
Write-Host ""
Write-Host "Model directory is ready at: $DST_ROOT" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Place a test .nii.gz scan in: test\input\interf0\images\t1-brain-mri\"
Write-Host "  2. Build the Docker image:       bash do_build.sh"
Write-Host "  3. Test locally:                 bash do_test_run.sh"
Write-Host "  4. Save for upload:              bash do_save.sh"
