"""
Generate a stratified Dice / HD95 summary table for ensembled ISLES26 predictions.

Produces the same layout as your baseline nnU-Net cross-validation report:
    OVERALL / TRAIN / VAL / LARGE / MEDIUM / SMALL / EMPTY

Requirements:
    pip install medpy SimpleITK pandas numpy tqdm

Adjust the CONFIG block to match your folder layout before running.
"""

import os
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from medpy.metric.binary import dc, hd95
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# =========================== CONFIG ===========================
# Folder containing your ENSEMBLED prediction .nii.gz files

DATA_ROOT = r"C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results\Dataset001_ISLES26\nnUNetTrainerWithLogging__nnUNetResEncUNetMPlans__3d_fullres\fold_3"
PRED_DIR =  DATA_ROOT + r"\validation"

# Folder containing the ground-truth label .nii.gz files (same case IDs as PRED_DIR)
GT_DIR = r"C:\Users\Admin\Desktop\meghpatel\isles26\data\nnUNet_raw\Dataset001_ISLES26\labelsTr"

# nnU-Net splits_final.json used to tag cases as TRAIN / VAL for a given fold.
# Set to None or leave a bad path if you don't want the train/val breakdown
# (e.g. if PRED_DIR only contains a held-out validation set already).
SPLITS_JSON = r"C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_preprocessed\Dataset001_ISLES26\splits_final.json"
FOLD_FOR_SPLIT = 0  # which fold's train/val split definition to use for tagging

# Lesion-size bin thresholds, in ground-truth foreground voxel count.
# EMPTY: 0 voxels | SMALL: 1..SMALL_MAX | MEDIUM: SMALL_MAX+1..MEDIUM_MAX | LARGE: >MEDIUM_MAX
# Tune these if you want to reproduce the exact n=747/513/187/6 split from your baseline.
SMALL_MAX = 500
MEDIUM_MAX = 3000

N_WORKERS = 8
OUT_CSV = DATA_ROOT + r"\summary_report\per_case_metrics.csv"
OUT_TXT = DATA_ROOT + r"\summary_report\summary_table.txt"

OUT_DIR = os.path.join(DATA_ROOT, "summary_report")

OUT_CSV = os.path.join(OUT_DIR, "per_case_metrics.csv")
OUT_TXT = os.path.join(OUT_DIR, "summary_table.txt")
# ================================================================


def load_case_ids(pred_dir):
    files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".nii.gz"))
    return [f[:-len(".nii.gz")] for f in files]


def load_split_tags(splits_json, fold):
    if not splits_json or not os.path.exists(splits_json):
        return {}
    with open(splits_json, "r") as f:
        splits = json.load(f)
    split = splits[fold]
    tags = {}
    for cid in split["train"]:
        tags[cid] = "TRAIN"
    for cid in split["val"]:
        tags[cid] = "VAL"
    return tags


def compute_case_metrics(case_id, pred_dir, gt_dir):
    pred_path = os.path.join(pred_dir, case_id + ".nii.gz")
    gt_path = os.path.join(gt_dir, case_id + ".nii.gz")
    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        return None

    pred_img = sitk.ReadImage(pred_path)
    gt_img = sitk.ReadImage(gt_path)

    pred = sitk.GetArrayFromImage(pred_img).astype(bool)
    gt = sitk.GetArrayFromImage(gt_img).astype(bool)

    # sitk spacing is (x, y, z); GetArrayFromImage returns array in (z, y, x) order
    spacing = pred_img.GetSpacing()[::-1]

    gt_voxels = int(gt.sum())
    pred_voxels = int(pred.sum())

    if gt_voxels == 0 and pred_voxels == 0:
        dice = 1.0
        hd_val = 0.0
    elif gt_voxels == 0 or pred_voxels == 0:
        # one side empty, other not: HD95 undefined, Dice is 0
        dice = 0.0
        hd_val = np.nan
    else:
        dice = dc(pred, gt)
        try:
            hd_val = hd95(pred, gt, voxelspacing=spacing)
        except Exception:
            hd_val = np.nan

    return {
        "case_id": case_id,
        "dice": dice,
        "hd95": hd_val,
        "gt_voxels": gt_voxels,
    }


def size_bin(gt_voxels):
    if gt_voxels == 0:
        return "EMPTY"
    elif gt_voxels <= SMALL_MAX:
        return "SMALL"
    elif gt_voxels <= MEDIUM_MAX:
        return "MEDIUM"
    else:
        return "LARGE"


def summarize(sub, label):
    n = len(sub)
    if n == 0:
        return None
    dice_mean, dice_std = sub["dice"].mean(), sub["dice"].std()
    hd_mean = sub["hd95"].dropna().mean()
    return f"{label:<7}(n={n:<5}): Dice = {dice_mean:.4f} \u00b1 {dice_std:.4f} | HD95 = {hd_mean:.2f} mm"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    case_ids = load_case_ids(PRED_DIR)
    if not case_ids:
        raise RuntimeError(f"No .nii.gz files found in {PRED_DIR}")

    split_tags = load_split_tags(SPLITS_JSON, FOLD_FOR_SPLIT)

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(compute_case_metrics, cid, PRED_DIR, GT_DIR): cid for cid in case_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring cases", unit="case"):
            r = fut.result()
            if r is not None:
                results.append(r)
            else:
                cid = futures[fut]
                tqdm.write(f"[skip] missing pred or GT for case: {cid}")

    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("No cases were successfully processed. Check PRED_DIR / GT_DIR paths.")

    df["split"] = df["case_id"].map(split_tags).fillna("UNK")
    df["size_group"] = df["gt_voxels"].apply(size_bin)

    lines = []
    lines.append("=" * 65)
    lines.append("SUMMARY".center(65))
    lines.append("=" * 65)
    lines.append(summarize(df, "OVERALL"))

    if not (df["split"] == "UNK").all():
        for split_label in ["TRAIN", "VAL"]:
            sub = df[df["split"] == split_label]
            if len(sub):
                lines.append(summarize(sub, split_label))
    else:
        print("[info] No split tags matched (splits_final.json missing/empty) — skipping TRAIN/VAL rows.")

    for grp in ["LARGE", "MEDIUM", "SMALL", "EMPTY"]:
        sub = df[df["size_group"] == grp]
        if len(sub):
            lines.append(summarize(sub, grp))

    lines.append("=" * 65)

    report = "\n".join(l for l in lines if l)
    print(report)

    df.to_csv(OUT_CSV, index=False)
    with open(OUT_TXT, "w") as f:
        f.write(report)

    print(f"\nSaved per-case metrics to {OUT_CSV}")
    print(f"Saved summary table to {OUT_TXT}")


if __name__ == "__main__":
    main()