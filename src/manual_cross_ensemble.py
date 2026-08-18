"""
manual_ensemble_crossval.py

nnUNetv2_find_best_configuration assumes every listed trainer works with
every listed plans file (it tests the full cross product). That breaks when,
like here, each trainer only has trained models under ONE specific plans file
(baseline + instance-loss under nnUNetPlans, ResEnc under
nnUNetResEncUNetMPlans) - it tries invalid combinations and errors out.

This script does the same thing find_best_configuration does under the hood
(pool each fold's own held-out validation predictions, ensemble-average
probabilities across chosen models, compute size-stratified Dice) but lets
you specify EXACTLY which trainer/plans pairs to include, and computes every
individual model, every pair, and the full combination in one pass.

Requires: every fold (0-4) for every model listed below must already have
run `--val --npz` (so validation/<case_id>.npz files exist).
"""

import json
import csv
from pathlib import Path
from itertools import combinations

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


# ============================================================================
# CONFIG
# ============================================================================

DATA_ROOT = Path(r"C:\Users\Admin\Desktop\meghpatel\isles26\data")
RESULTS_ROOT = DATA_ROOT / "nnunet_results" / "Dataset001_ISLES26"
SPLITS_JSON = DATA_ROOT / "nnunet_preprocessed" / "Dataset001_ISLES26" / "splits_final.json"
GT_SEGMENTATIONS_DIR = DATA_ROOT / "nnunet_preprocessed" / "Dataset001_ISLES26" / "gt_segmentations"

# Everything this script writes goes here - kept separate from nnU-Net's own
# results folders so nothing here can ever collide with or overwrite them.
OUTPUT_DIR = RESULTS_ROOT / "manual_ensemble_results"
PER_CASE_DIR = OUTPUT_DIR / "per_case"

# Every model you want considered, individually and in every combination.
MEMBER_MODELS = [
    {"label": "baseline",      "trainer": "nnUNetTrainer",             "plans": "nnUNetPlans"},
    {"label": "instance_loss", "trainer": "nnUNetTrainerInstanceLoss",  "plans": "nnUNetPlans"},
    {"label": "resenc",        "trainer": "nnUNetTrainerWithLogging",  "plans": "nnUNetResEncUNetMPlans"},
]

FOLDS = [0, 1, 2, 3, 4]


# ============================================================================
# Helpers
# ============================================================================

def load_npz_probabilities(npz_path: Path) -> np.ndarray:
    """Loads softmax probabilities from a validation .npz file, defensively
    checking a few common key names since nnU-Net versions have varied."""
    data = np.load(npz_path)
    for key in ("probabilities", "softmax", "data", "arr_0"):
        if key in data:
            return data[key]
    raise KeyError(f"No recognized probability key in {npz_path}. Found keys: {list(data.keys())}")


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Binary Dice. True empty-empty match (both masks all-zero) scores 1.0,
    matching the convention used throughout this project rather than NaN."""
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return (2.0 * tp) / (2.0 * tp + fp + fn)


def size_bucket(n_ref: int) -> str:
    if n_ref == 0:
        return "EMPTY"
    elif n_ref < 500:
        return "SMALL"
    elif n_ref < 3000:
        return "MEDIUM"
    else:
        return "LARGE"


def build_case_to_fold_map(splits: list) -> dict:
    case_to_fold = {}
    for fold_idx, split in enumerate(splits):
        for case_id in split["val"]:
            case_to_fold[case_id] = fold_idx
    return case_to_fold


def model_results_dir(model: dict) -> Path:
    return RESULTS_ROOT / f"{model['trainer']}__{model['plans']}__3d_fullres"


def save_per_case_csv(combo_name: str, rows: list):
    PER_CASE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = combo_name.replace(" + ", "_plus_").replace(" ", "_")
    out_path = PER_CASE_DIR / f"{safe_name}_per_case.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "n_ref", "dice", "size_group"])
        writer.writerows(rows)
    return out_path


# ============================================================================
# Core: evaluate one combination of models
# ============================================================================

def evaluate_combination(models: list, case_to_fold: dict) -> dict:
    """Averages probabilities across `models` for every case (using each
    case's own held-out fold), computes Dice, returns size-stratified stats."""
    rows = []

    case_items = list(case_to_fold.items())
    combo_label = " + ".join(m["label"] for m in models)
    for case_id, fold_idx in tqdm(case_items, desc=f"Scoring [{combo_label}]", unit="case"):
        prob_sum = None
        missing = False

        for model in models:
            npz_path = model_results_dir(model) / f"fold_{fold_idx}" / "validation" / f"{case_id}.npz"
            if not npz_path.exists():
                missing = True
                break
            probs = load_npz_probabilities(npz_path)
            prob_sum = probs if prob_sum is None else prob_sum + probs

        if missing:
            tqdm.write(f"  WARNING: skipping {case_id} (fold {fold_idx}) - missing .npz for one or more models")
            continue

        avg_probs = prob_sum / len(models)
        pred_seg = np.argmax(avg_probs, axis=0)
        pred_mask = pred_seg == 1  # binary foreground, matches project convention

        gt_path = GT_SEGMENTATIONS_DIR / f"{case_id}.nii.gz"
        gt_img = sitk.ReadImage(str(gt_path))
        gt_arr = sitk.GetArrayFromImage(gt_img)
        gt_mask = gt_arr > 0

        dice = compute_dice(pred_mask, gt_mask)
        n_ref = int(gt_mask.sum())
        rows.append((case_id, n_ref, dice, size_bucket(n_ref)))

    # aggregate
    buckets = {"LARGE": [], "MEDIUM": [], "SMALL": [], "EMPTY": []}
    for _, _, dice, b in rows:
        buckets[b].append(dice)

    result = {"n_total": len(rows)}
    for b in ["LARGE", "MEDIUM", "SMALL", "EMPTY"]:
        vals = buckets[b]
        result[b] = {"n": len(vals), "dice": float(np.mean(vals)) if vals else float("nan")}
    all_dice = [d for _, _, d, _ in rows]
    result["OVERALL"] = {"n": len(all_dice), "dice": float(np.mean(all_dice))}
    result["rows"] = rows
    return result


# ============================================================================
# MAIN - test every individual model, every pair, and the full combination
# ============================================================================

def main():
    splits = json.load(open(SPLITS_JSON))
    case_to_fold = build_case_to_fold_map(splits)
    print(f"Total cases across all folds' validation sets: {len(case_to_fold)}")

    combos_to_test = []
    for r in range(1, len(MEMBER_MODELS) + 1):
        combos_to_test.extend(combinations(MEMBER_MODELS, r))

    all_results = {}

    for combo in combos_to_test:
        labels = [m["label"] for m in combo]
        combo_name = " + ".join(labels)
        print(f"\n{'=' * 70}\nEvaluating: {combo_name}\n{'=' * 70}")

        result = evaluate_combination(list(combo), case_to_fold)
        all_results[combo_name] = result

        csv_path = save_per_case_csv(combo_name, result["rows"])
        print(f"  Per-case results saved to: {csv_path}")

        print(f"  n={result['n_total']}")
        for b in ["LARGE", "MEDIUM", "SMALL", "EMPTY", "OVERALL"]:
            print(f"  {b:10} n={result[b]['n']:4d}  Dice={result[b]['dice']:.4f}")

    print(f"\n{'=' * 70}\nSUMMARY - OVERALL Dice per combination, sorted best to worst\n{'=' * 70}")
    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["OVERALL"]["dice"], reverse=True)
    for name, result in ranked:
        print(f"  {result['OVERALL']['dice']:.4f}   {name}")

    print(f"\n{'=' * 70}\nSUMMARY - SMALL bucket Dice per combination, sorted best to worst\n{'=' * 70}")
    ranked_small = sorted(all_results.items(), key=lambda kv: kv[1]["SMALL"]["dice"], reverse=True)
    for name, result in ranked_small:
        print(f"  {result['SMALL']['dice']:.4f}   {name}")


if __name__ == "__main__":
    main()