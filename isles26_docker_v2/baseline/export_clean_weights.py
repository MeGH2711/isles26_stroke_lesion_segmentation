"""
export_clean_weights.py

Reads training checkpoints from data/nnunet_results/Dataset001_ISLES26/nnUNetTrainer__nnUNetPlans__3d_fullres,
strips out optimizer states, and writes clean, lightweight inference checkpoints
into isles26_docker/baseline/model/Dataset001_ISLES26/nnUNetTrainer__nnUNetPlans__3d_fullres/.
"""

import os
import shutil
import time
from pathlib import Path
import torch

SRC_ROOT = Path(r"C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results\Dataset001_ISLES26")
DST_ROOT = Path(__file__).resolve().parent / "model" / "Dataset001_ISLES26"

MODELS = [
    {
        "folder": "nnUNetTrainer__nnUNetPlans__3d_fullres",
        "label": "Baseline",
    },
]

FOLDS = [0, 1, 2, 3, 4]


def process_model(model_info):
    model_folder = model_info["folder"]
    label = model_info["label"]
    src_dir = SRC_ROOT / model_folder
    dst_dir = DST_ROOT / model_folder
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {label}: {model_folder}")
    print(f"{'='*60}")

    # 1. Copy JSON / PKL configuration files
    for fname in ["dataset.json", "plans.json", "dataset_fingerprint.json", "postprocessing.json", "postprocessing.pkl"]:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / fname)
            print(f"  [+] Copied {fname}")

    # Also check crossval postprocessing
    cv_dir = src_dir / "crossval_results_folds_0_1_2_3_4"
    if cv_dir.exists():
        dst_cv = dst_dir / "crossval_results_folds_0_1_2_3_4"
        dst_cv.mkdir(parents=True, exist_ok=True)
        for fname in ["postprocessing.json", "postprocessing.pkl"]:
            src_file = cv_dir / fname
            if src_file.exists():
                shutil.copy2(src_file, dst_cv / fname)
                shutil.copy2(src_file, dst_dir / fname)
                print(f"  [+] Copied crossval {fname}")

    # 2. Process fold checkpoints
    for fold in FOLDS:
        src_ckpt = src_dir / f"fold_{fold}" / "checkpoint_final.pth"
        dst_fold = dst_dir / f"fold_{fold}"
        dst_fold.mkdir(parents=True, exist_ok=True)
        dst_ckpt = dst_fold / "checkpoint_final.pth"

        if not src_ckpt.exists():
            print(f"  [!] Warning: {src_ckpt} not found!")
            continue

        orig_size = os.path.getsize(src_ckpt) / (1024 * 1024)
        t0 = time.time()

        ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)

        clean_ckpt = {
            "network_weights": ckpt["network_weights"],
            "inference_allowed_mirroring_axes": ckpt.get("inference_allowed_mirroring_axes", (0, 1, 2)),
            "trainer_name": ckpt.get("trainer_name"),
            "init_args": ckpt.get("init_args"),
        }

        torch.save(clean_ckpt, dst_ckpt)
        clean_size = os.path.getsize(dst_ckpt) / (1024 * 1024)
        elapsed = time.time() - t0

        print(
            f"  [OK] fold_{fold}: {orig_size:.1f} MB -> {clean_size:.1f} MB "
            f"({(1 - clean_size/orig_size)*100:.0f}% reduced) in {elapsed:.1f}s"
        )


def main():
    print(f"Source:      {SRC_ROOT}")
    print(f"Destination: {DST_ROOT}")

    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source directory does not exist: {SRC_ROOT}")

    for m in MODELS:
        process_model(m)

    total_bytes = sum(f.stat().st_size for f in DST_ROOT.rglob("*") if f.is_file())
    total_gb = total_bytes / (1024 ** 3)
    print(f"\n{'='*60}")
    print(f"Done! Clean baseline model directory total size: {total_gb:.2f} GB")
    print(f"Location: {DST_ROOT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
