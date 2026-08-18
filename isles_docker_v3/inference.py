"""
ISLES 2026 — nnU-Net Inference (Optimized Weighted Ensemble: ResEnc-M + Baseline)

This module implements the Grand Challenge inference interface:
    init_model()  — Load both Baseline and ResEnc nnU-Net models at startup
    run(model)    — Run 10-fold weighted ensembled inference on a single T1 MRI scan

Optimized Strategy:
    1. Run nnUNetPredictor for Baseline (nnUNetTrainer / nnUNetPlans / 3d_fullres, folds 0–4)
    2. Run nnUNetPredictor for ResEnc (nnUNetTrainerWithLogging / nnUNetResEncUNetMPlans / 3d_fullres, folds 0–4)
    3. Weighted Softmax Ensembling: 0.65 * ResEnc + 0.35 * Baseline
    4. Thresholding at T = 0.45 to capture subtle ischemic lesion boundaries
    5. Connected Component Filtering (min_size = 8 voxels, 26-connectivity) to remove isolated noise
    6. Continuous float32 lesion probability map output

Outputs:
    /output/images/stroke-lesion-segmentation/<uuid>.mha   (uint8 binary mask)
    /output/images/lesion-probability-map/<uuid>.mha        (float32 probability)
"""

import glob
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from skimage.measure import label as sk_label

# ──────────────────────────────────────────────────────────────────────────────
# Logging helper — prints with elapsed-time prefix for Grand Challenge logs
# ──────────────────────────────────────────────────────────────────────────────
_T0 = time.time()


def _log(msg, level="INFO"):
    """Print a timestamped log line. Flushes immediately for real-time GC logs."""
    elapsed = time.time() - _T0
    mins, secs = divmod(elapsed, 60)
    print(f"[{level}] [{int(mins):02d}:{secs:05.2f}] {msg}", flush=True)
    sys.stdout.flush()


def _progress_bar(step, total, label="", width=30):
    """Print a text progress bar for real-time monitoring."""
    pct = step / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    _log(f"  {bar} {step}/{total} ({pct:.0%}) {label}")


# ──────────────────────────────────────────────────────────────────────────────
# Paths (Grand Challenge conventions)
# ──────────────────────────────────────────────────────────────────────────────
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

# Grand Challenge extracts uploaded model tarballs here
MODEL_DIR = Path("/opt/ml/model")

# ──────────────────────────────────────────────────────────────────────────────
# Model configuration — 10-fold Optimized Weighted Ensemble
# ──────────────────────────────────────────────────────────────────────────────
DATASET_NAME = "Dataset001_ISLES26"
FOLDS = (0, 1, 2, 3, 4)

MODELS_CONFIG = [
    {
        "label": "baseline",
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetPlans",
        "configuration": "3d_fullres",
        "weight": 0.35,
    },
    {
        "label": "resenc",
        "trainer": "nnUNetTrainerWithLogging",
        "plans": "nnUNetResEncUNetMPlans",
        "configuration": "3d_fullres",
        "weight": 0.65,
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Threshold & Postprocessing Configurations
# ──────────────────────────────────────────────────────────────────────────────
# T=0.45 captures subtle ischemic lesion boundaries and penumbra while avoiding FP inflation
THRESHOLD = 0.45

# 26-connectivity filter removing isolated false-positive noise (< 8 voxels)
MIN_COMPONENT_VOXELS = 8


def _remove_small_components(binary_seg, min_voxel_size):
    """
    Remove connected components smaller than min_voxel_size voxels from a
    binary segmentation. Uses full 3D 26-connectivity (connectivity=3 in skimage).

    Returns (cleaned_seg, num_components_removed).
    """
    if min_voxel_size <= 0:
        return binary_seg, 0

    labeled = sk_label(binary_seg, connectivity=3)
    if labeled.max() == 0:
        return binary_seg, 0

    sizes = np.bincount(labeled.ravel())
    label_ids = np.arange(len(sizes))
    small_labels = label_ids[(sizes < min_voxel_size) & (label_ids != 0)]

    if len(small_labels) == 0:
        return binary_seg, 0

    mask = np.isin(labeled, small_labels)
    cleaned = binary_seg.copy()
    cleaned[mask] = 0
    return cleaned, int(len(small_labels))


def _show_torch_cuda_info():
    """Print CUDA availability info for debugging."""
    _log("═" * 50)
    _log("Collecting Torch CUDA information")
    available = torch.cuda.is_available()
    _log(f"Torch CUDA is available: {available}")
    if available:
        _log(f"  Number of devices: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        _log(f"  Current device: {current_device}")
        props = torch.cuda.get_device_properties(current_device)
        _log(f"  GPU: {props.name} ({props.total_memory // (1024**2)} MB VRAM)")
    _log("═" * 50)


def init_model():
    """
    Load Baseline and ResEnc nnU-Net models and return predictors with weights.

    This is called once by app.py during server startup (before /health
    returns 200). Predictors are reused across all /invoke calls.
    """
    _show_torch_cuda_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Using device: {device}")

    # nnU-Net needs results directory set to find model weights
    os.environ["nnUNet_results"] = str(MODEL_DIR)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictors = []
    for i, cfg in enumerate(MODELS_CONFIG):
        model_name = f"{cfg['trainer']}__{cfg['plans']}__{cfg['configuration']}"
        _log("")
        _log(f"Loading model {i+1}/{len(MODELS_CONFIG)}: {cfg['label']} ({model_name}, weight={cfg['weight']})")
        t_load = time.time()

        _log("  Creating nnUNetPredictor (tile_step_size=0.5, mirroring=on) ...")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,         # Flip-based 3D TTA across orthogonal axes
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )

        model_folder = MODEL_DIR / DATASET_NAME / model_name
        _log(f"  Model folder: {model_folder}")
        _log(f"  Folder exists: {model_folder.exists()}")

        _log(f"  Loading {len(FOLDS)} fold checkpoints (checkpoint_final.pth) ...")
        predictor.initialize_from_trained_model_folder(
            str(model_folder),
            use_folds=FOLDS,
            checkpoint_name="checkpoint_final.pth",
        )

        predictors.append({
            "label": cfg["label"],
            "predictor": predictor,
            "weight": cfg["weight"],
        })
        load_elapsed = time.time() - t_load
        _log(f"  ✓ {cfg['label']} loaded successfully ({load_elapsed:.1f}s)")

    # Load postprocessing if present
    pp_fns = []
    pp_kwargs = []
    try:
        from batchgenerators.utilities.file_and_folder_operations import load_pickle
        for cfg in MODELS_CONFIG:
            model_folder = (
                MODEL_DIR / DATASET_NAME
                / f"{cfg['trainer']}__{cfg['plans']}__{cfg['configuration']}"
            )
            for pp_path in [
                model_folder / "postprocessing.pkl",
                model_folder / "crossval_results_folds_0_1_2_3_4" / "postprocessing.pkl",
            ]:
                if pp_path.exists():
                    fns, kwargs = load_pickle(str(pp_path))
                    if fns:
                        pp_fns = fns
                        pp_kwargs = kwargs
                        _log(f"  ✓ Loaded postprocessing functions: {pp_fns}")
                        break
            if pp_fns:
                break
    except Exception as e:
        _log(f"Note on postprocessing loading: {e}", level="WARN")

    _log("")
    _log("══════════════════════════════════════════════════")
    _log(f"  All {len(predictors)} models loaded. Ready for inference.")
    _log("══════════════════════════════════════════════════")
    return {
        "predictors": predictors,
        "pp_fns": pp_fns,
        "pp_kwargs": pp_kwargs,
    }


def run(model):
    """
    Run 10-fold weighted ensembled inference on a single T1 MRI scan.
    """
    t_start = time.time()
    _log("")
    _log("══════════════════════════════════════════════════")
    _log("  INFERENCE STARTED (Optimized Ensemble v3)")
    _log("══════════════════════════════════════════════════")

    if isinstance(model, dict):
        predictors = model["predictors"]
        pp_fns = model.get("pp_fns", [])
        pp_kwargs = model.get("pp_kwargs", [])
    else:
        predictors = model
        pp_fns = []
        pp_kwargs = []

    # ──────────────────────────────────────────────────────────────────────
    # 1. Locate input T1 MRI scan
    # ──────────────────────────────────────────────────────────────────────
    t1_dir = INPUT_PATH / "images" / "t1-brain-mri"
    valid_exts = (".mha", ".mhd", ".nii.gz", ".nii", ".nrrd", ".tiff", ".tif")

    t1_files = []
    if t1_dir.exists():
        t1_files = [
            f for f in sorted(t1_dir.iterdir())
            if f.is_file() and any(f.name.lower().endswith(ext) for ext in valid_exts)
        ]

    if not t1_files:
        t1_files = [
            f for f in sorted(INPUT_PATH.rglob("*"))
            if f.is_file() and any(f.name.lower().endswith(ext) for ext in valid_exts)
        ]

    if not t1_files:
        raise FileNotFoundError(
            f"No medical image files found in {t1_dir}. "
            f"Contents of /input: {list(INPUT_PATH.rglob('*'))}"
        )

    input_image_path = str(t1_files[0])
    _log(f"[Step 1/6] Input image found: {input_image_path}")

    # Read image to preserve spatial metadata (spacing, origin, direction)
    _log("[Step 2/6] Reading and preprocessing image ...")
    t_read = time.time()
    original_sitk_image = sitk.ReadImage(input_image_path)

    # ──────────────────────────────────────────────────────────────────────
    # 2. Convert to numpy array for in-memory sliding window prediction
    # ──────────────────────────────────────────────────────────────────────
    image_npy = sitk.GetArrayFromImage(original_sitk_image).astype(np.float32)
    image_npy = image_npy[np.newaxis]  # Shape: (1, z, y, x)

    spacing = original_sitk_image.GetSpacing()  # (x, y, z)
    image_properties = {"spacing": list(spacing)}

    voxels = image_npy.shape[1] * image_npy.shape[2] * image_npy.shape[3]
    read_elapsed = time.time() - t_read
    _log(f"  Image shape: {image_npy.shape} ({voxels:,} voxels)")
    _log(f"  Spacing: {spacing}")
    _log(f"  Image loaded in {read_elapsed:.2f}s")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Run predictions and collect softmax probabilities
    # ──────────────────────────────────────────────────────────────────────
    weighted_softmax_sum = None
    total_weight = 0.0

    for pred_idx, pred_info in enumerate(predictors):
        label = pred_info["label"]
        predictor = pred_info["predictor"]
        weight = pred_info.get("weight", 1.0)

        _log("")
        _log(f"[Step 3/6] Running inference: {label} (model {pred_idx+1}/{len(predictors)}, weight={weight}) ...")

        t_pred = time.time()
        seg, softmax = predictor.predict_single_npy_array(
            input_image=image_npy,
            image_properties=image_properties,
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=True,
        )

        if weighted_softmax_sum is None:
            weighted_softmax_sum = weight * softmax
        else:
            weighted_softmax_sum += weight * softmax

        total_weight += weight
        elapsed = time.time() - t_pred
        _log(f"  ✓ {label} inference complete: softmax {softmax.shape} ({elapsed:.1f}s)")
        _progress_bar(pred_idx + 1, len(predictors), label="models done")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Weighted Ensemble Softmax Blending
    # ──────────────────────────────────────────────────────────────────────
    _log("")
    _log("[Step 4/6] Ensembling predictions with weighted blending ...")
    t_ensemble = time.time()
    ensemble_softmax = weighted_softmax_sum / total_weight
    _log(f"  Ensemble softmax shape: {ensemble_softmax.shape} (Total weight: {total_weight:.2f})")
    _log(f"  ✓ Ensembling done ({time.time() - t_ensemble:.2f}s)")

    # ──────────────────────────────────────────────────────────────────────
    # 5. Generate Segmentation & Probability Map with Optimized Parameters
    # ──────────────────────────────────────────────────────────────────────
    _log("")
    _log("[Step 5/6] Generating segmentation (T=0.45, min_size=8) and probability map ...")
    t_post = time.time()

    # Probability map = continuous foreground probability (class index 1)
    prob_map = ensemble_softmax[1].astype(np.float32)

    # Thresholding at T = 0.45
    binary_seg = (prob_map >= THRESHOLD).astype(np.uint8)

    # Optional nnU-Net postprocessing
    if pp_fns:
        try:
            from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing
            binary_seg = apply_postprocessing(binary_seg, pp_fns, pp_kwargs)
            _log("  ✓ Applied configuration postprocessing")
        except Exception as e:
            _log(f"  Warning: Failed to apply postprocessing: {e}", level="WARN")

    # 26-connectivity noise filter removing isolated false positives < 8 voxels
    _log(f"  Removing isolated components smaller than {MIN_COMPONENT_VOXELS} voxels (26-connectivity) ...")
    binary_seg, n_removed = _remove_small_components(binary_seg, MIN_COMPONENT_VOXELS)
    if n_removed:
        _log(f"  ✓ Filtered {n_removed} spurious component(s) below {MIN_COMPONENT_VOXELS}-voxel cutoff")
    else:
        _log("  No small components below cutoff")

    fg_voxels = int(np.sum(binary_seg == 1))
    _log(f"  Binary seg: unique={np.unique(binary_seg)}, foreground={fg_voxels:,} voxels")
    _log(f"  Prob map: min={prob_map.min():.4f}, max={prob_map.max():.4f}, mean={prob_map.mean():.4f}")
    _log(f"  ✓ Outputs processed in ({time.time() - t_post:.2f}s)")

    # ──────────────────────────────────────────────────────────────────────
    # 6. Write Output MetaImage (.mha) Files
    # ──────────────────────────────────────────────────────────────────────
    _log("")
    _log("[Step 6/6] Writing output MetaImage (.mha) files ...")
    t_write = time.time()
    output_uuid = str(uuid.uuid4())

    # 1. Binary segmentation output (.mha format for Grand Challenge socket)
    seg_output_dir = OUTPUT_PATH / "images" / "stroke-lesion-segmentation"
    seg_output_dir.mkdir(parents=True, exist_ok=True)
    seg_output_path = seg_output_dir / f"{output_uuid}.mha"

    seg_sitk = sitk.GetImageFromArray(binary_seg)
    seg_sitk.CopyInformation(original_sitk_image)
    sitk.WriteImage(seg_sitk, str(seg_output_path), True)
    _log(f"  ✓ Binary segmentation written: {seg_output_path}")

    # 2. Continuous probability map output (.mha format for Grand Challenge socket)
    prob_output_dir = OUTPUT_PATH / "images" / "lesion-probability-map"
    prob_output_dir.mkdir(parents=True, exist_ok=True)
    prob_output_path = prob_output_dir / f"{output_uuid}.mha"

    prob_sitk = sitk.GetImageFromArray(prob_map)
    prob_sitk.CopyInformation(original_sitk_image)
    sitk.WriteImage(prob_sitk, str(prob_output_path), True)
    _log(f"  ✓ Probability map written: {prob_output_path}")
    _log(f"  Output write time: {time.time() - t_write:.2f}s")

    total_time = time.time() - t_start
    _log("")
    _log("══════════════════════════════════════════════════")
    _log(f"  INFERENCE COMPLETE — Total Execution Time: {total_time:.1f}s")
    _log("══════════════════════════════════════════════════")
