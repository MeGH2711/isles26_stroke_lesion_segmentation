"""
ISLES 2026 — nnU-Net Inference (Baseline 5-fold Ensemble)

This module implements the Grand Challenge inference interface:
    init_model()  — load the Baseline nnU-Net model at startup
    run(model)    — run 5-fold ensembled inference on a single T1 MRI scan

Strategy:
    1. Run nnUNetPredictor for Baseline (nnUNetTrainer / nnUNetPlans / 3d_fullres, folds 0–4)
    2. The predictor internally averages softmax across all 5 folds
    3. Argmax → binary segmentation; foreground channel → probability map

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
# Model configuration — Baseline (nnUNetTrainer / nnUNetPlans / 3d_fullres)
# ──────────────────────────────────────────────────────────────────────────────
DATASET_NAME = "Dataset001_ISLES26"
FOLDS = (0, 1, 2, 3, 4)

MODELS_CONFIG = [
    {
        "label": "baseline",
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetPlans",
        "configuration": "3d_fullres",
    },
]


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
    Load the Baseline nnU-Net model and return the predictor.

    Called once by app.py during server startup.
    """
    _show_torch_cuda_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Using device: {device}")

    # nnU-Net needs the results directory set to find model weights
    os.environ["nnUNet_results"] = str(MODEL_DIR)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictors = []
    for i, cfg in enumerate(MODELS_CONFIG):
        model_name = f"{cfg['trainer']}__{cfg['plans']}__{cfg['configuration']}"
        _log(f"")
        _log(f"Loading model {i+1}/{len(MODELS_CONFIG)}: {cfg['label']} ({model_name})")
        t_load = time.time()

        _log(f"  Creating nnUNetPredictor (tile_step_size=0.625, mirroring=off) ...")
        predictor = nnUNetPredictor(
            tile_step_size=0.625,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )

        model_folder = (
            MODEL_DIR / DATASET_NAME / model_name
        )
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
        })
        load_elapsed = time.time() - t_load
        _log(f"  ✓ {cfg['label']} loaded successfully ({load_elapsed:.1f}s)")

    # Load postprocessing if present
    pp_fns = []
    pp_kwargs = []
    _log(f"Checking for postprocessing files ...")
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
                    _log(f"  Checking: {pp_path}")
                    fns, kwargs = load_pickle(str(pp_path))
                    if fns:
                        pp_fns = fns
                        pp_kwargs = kwargs
                        _log(f"  ✓ Loaded postprocessing functions: {pp_fns}")
                        break
                    else:
                        _log(f"  (No active functions — no-op)")
            if pp_fns:
                break
    except Exception as e:
        _log(f"Note on postprocessing loading: {e}", level="WARN")

    _log(f"")
    _log(f"══════════════════════════════════════════════════")
    _log(f"  All {len(predictors)} model(s) loaded. Ready for inference.")
    _log(f"══════════════════════════════════════════════════")
    return {
        "predictors": predictors,
        "pp_fns": pp_fns,
        "pp_kwargs": pp_kwargs,
    }


def run(model):
    """
    Run 5-fold ensembled inference on a single T1 MRI scan.
    """
    t_start = time.time()
    _log(f"")
    _log(f"══════════════════════════════════════════════════")
    _log(f"  INFERENCE STARTED (Baseline 5-fold)")
    _log(f"══════════════════════════════════════════════════")

    if isinstance(model, dict):
        predictors = model["predictors"]
        pp_fns = model.get("pp_fns", [])
        pp_kwargs = model.get("pp_kwargs", [])
    else:
        predictors = model
        pp_fns = []
        pp_kwargs = []

    # ──────────────────────────────────────────────────────────────────────
    # 1. Locate the input T1 MRI scan
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

    # Read the original image to preserve metadata
    _log(f"[Step 2/6] Reading and preprocessing image ...")
    t_read = time.time()
    original_sitk_image = sitk.ReadImage(input_image_path)

    # ──────────────────────────────────────────────────────────────────────
    # 2. Load image as numpy array for in-memory prediction
    # ──────────────────────────────────────────────────────────────────────
    image_npy = sitk.GetArrayFromImage(original_sitk_image).astype(np.float32)
    image_npy = image_npy[np.newaxis]  # (1, z, y, x)

    spacing = original_sitk_image.GetSpacing()  # (x, y, z)
    image_properties = {"spacing": list(spacing)}

    voxels = image_npy.shape[1] * image_npy.shape[2] * image_npy.shape[3]
    read_elapsed = time.time() - t_read
    _log(f"  Image shape: {image_npy.shape} ({voxels:,} voxels)")
    _log(f"  Spacing: {spacing}")
    _log(f"  Image loaded in {read_elapsed:.2f}s")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Run prediction and collect softmax probabilities
    # ──────────────────────────────────────────────────────────────────────
    all_softmax = []

    for pred_idx, pred_info in enumerate(predictors):
        label = pred_info["label"]
        predictor = pred_info["predictor"]
        _log(f"")
        _log(f"[Step 3/6] Running inference: {label} ({len(FOLDS)} folds) ...")

        t_pred = time.time()
        seg, softmax = predictor.predict_single_npy_array(
            input_image=image_npy,
            image_properties=image_properties,
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=True,
        )

        all_softmax.append(softmax)
        elapsed = time.time() - t_pred
        _log(f"  ✓ {label} inference complete: softmax {softmax.shape} ({elapsed:.1f}s)")
        _progress_bar(pred_idx + 1, len(predictors), label="models done")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Ensemble: average softmax probabilities
    # ──────────────────────────────────────────────────────────────────────
    _log(f"")
    _log(f"[Step 4/6] Ensembling predictions ...")
    t_ensemble = time.time()
    if len(all_softmax) == 1:
        ensemble_softmax = all_softmax[0]
        _log(f"  Single model — using softmax directly: {ensemble_softmax.shape}")
    else:
        _log(f"  Averaging {len(all_softmax)} softmax arrays ...")
        ensemble_softmax = np.mean(all_softmax, axis=0)
        _log(f"  Ensemble softmax shape: {ensemble_softmax.shape}")
    _log(f"  ✓ Ensembling done ({time.time() - t_ensemble:.2f}s)")

    # ──────────────────────────────────────────────────────────────────────
    # 5. Generate binary segmentation (argmax) and probability map
    # ──────────────────────────────────────────────────────────────────────
    _log(f"")
    _log(f"[Step 5/6] Generating segmentation and probability map ...")
    t_post = time.time()

    binary_seg = np.argmax(ensemble_softmax, axis=0).astype(np.uint8)

    # Apply postprocessing if defined
    if pp_fns:
        _log(f"  Applying postprocessing to binary segmentation ...")
        try:
            from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing
            binary_seg = apply_postprocessing(binary_seg, pp_fns, pp_kwargs)
            _log(f"  ✓ Postprocessing applied successfully")
        except Exception as e:
            _log(f"  Warning: Failed to apply postprocessing: {e}", level="WARN")

    # Probability map = foreground class probability (class index 1)
    prob_map = ensemble_softmax[1].astype(np.float32)

    fg_voxels = int(np.sum(binary_seg == 1))
    _log(f"  Binary seg: unique={np.unique(binary_seg)}, foreground={fg_voxels:,} voxels")
    _log(f"  Prob map: min={prob_map.min():.4f}, max={prob_map.max():.4f}, mean={prob_map.mean():.4f}")
    _log(f"  ✓ Segmentation generated ({time.time() - t_post:.2f}s)")

    # ──────────────────────────────────────────────────────────────────────
    # 6. Write outputs as MetaImage (.mha) files
    # ──────────────────────────────────────────────────────────────────────
    _log(f"")
    _log(f"[Step 6/6] Writing output files ...")
    t_write = time.time()
    output_uuid = str(uuid.uuid4())

    # Binary segmentation output (.mha format for Grand Challenge socket)
    seg_output_dir = OUTPUT_PATH / "images" / "stroke-lesion-segmentation"
    seg_output_dir.mkdir(parents=True, exist_ok=True)
    seg_output_path = seg_output_dir / f"{output_uuid}.mha"

    seg_sitk = sitk.GetImageFromArray(binary_seg)
    seg_sitk.CopyInformation(original_sitk_image)
    sitk.WriteImage(seg_sitk, str(seg_output_path), True)
    _log(f"  ✓ Segmentation written: {seg_output_path}")

    # Probability map output (.mha format for Grand Challenge socket)
    prob_output_dir = OUTPUT_PATH / "images" / "lesion-probability-map"
    prob_output_dir.mkdir(parents=True, exist_ok=True)
    prob_output_path = prob_output_dir / f"{output_uuid}.mha"

    prob_sitk = sitk.GetImageFromArray(prob_map)
    prob_sitk.CopyInformation(original_sitk_image)
    sitk.WriteImage(prob_sitk, str(prob_output_path), True)
    _log(f"  ✓ Probability map written: {prob_output_path}")
    _log(f"  Output write time: {time.time() - t_write:.2f}s")

    total_time = time.time() - t_start
    _log(f"")
    _log(f"══════════════════════════════════════════════════")
    _log(f"  INFERENCE COMPLETE — Total: {total_time:.1f}s")
    _log(f"══════════════════════════════════════════════════")
