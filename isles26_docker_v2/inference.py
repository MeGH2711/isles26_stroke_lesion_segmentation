"""
ISLES 2026 — nnU-Net Inference (Baseline + ResEnc 10-fold Ensemble)

This module implements the Grand Challenge inference interface:
    init_model()  — load both Baseline and ResEnc nnU-Net models at startup
    run(model)    — run 10-fold ensembled inference on a single T1 MRI scan

Ensemble strategy:
    1. Run nnUNetPredictor for Baseline (nnUNetTrainer / nnUNetPlans / 3d_fullres, folds 0–4)
    2. Run nnUNetPredictor for ResEnc (nnUNetTrainerWithLogging / nnUNetResEncUNetMPlans / 3d_fullres, folds 0–4)
    3. Average softmax probability maps across all 10 folds
    4. Argmax → binary segmentation; foreground channel → probability map

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
# Model configuration — 10-fold Ensemble (Baseline + ResEnc)
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
    {
        "label": "resenc",
        "trainer": "nnUNetTrainerWithLogging",
        "plans": "nnUNetResEncUNetMPlans",
        "configuration": "3d_fullres",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Postprocessing — voxel-count threshold to suppress spurious FP components
# ──────────────────────────────────────────────────────────────────────────────
# Distinct from nnU-Net's "keep only largest component" logic (which you found
# actively harmful for multi-component lesions). This only drops components
# below a minimum size, leaving all other components — including small but
# real lesions — untouched.
#
# IMPORTANT: 8 is a placeholder starting point. Tune this against your own
# held-out folds (sweep e.g. 2/4/8/16/32 voxels) before trusting it on the
# leaderboard — too high will start deleting real small lesions, which is
# exactly the failure mode you're trying to avoid.
MIN_COMPONENT_VOXELS = 8


def _remove_small_components(binary_seg, min_voxel_size):
    """
    Remove connected components smaller than min_voxel_size voxels from a
    binary segmentation. Uses 26-connectivity to match the co-occurrence
    diagnostic already run on this dataset.

    Returns (cleaned_seg, num_components_removed).
    """
    labeled = sk_label(binary_seg, connectivity=3)  # connectivity=3 -> full 26-connected in 3D
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
    Load the ResEnc nnU-Net model and return the predictor.

    This is called once by app.py during server startup (before /health
    returns 200). The predictor is then reused for all /invoke calls.
    """
    _show_torch_cuda_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Using device: {device}")

    # nnU-Net needs the results directory set to find model weights
    # Grand Challenge extracts the model tarball to /opt/ml/model/
    os.environ["nnUNet_results"] = str(MODEL_DIR)

    # Import nnUNetPredictor here (after env vars are set)
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictors = []
    for i, cfg in enumerate(MODELS_CONFIG):
        model_name = f"{cfg['trainer']}__{cfg['plans']}__{cfg['configuration']}"
        _log(f"")
        _log(f"Loading model {i+1}/{len(MODELS_CONFIG)}: {cfg['label']} ({model_name})")
        t_load = time.time()

        _log(f"  Creating nnUNetPredictor (tile_step_size=0.5, mirroring=on) ...")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,         # full overlap — your test runs finished in <2min for
                                         # the full 10-fold ensemble, so there's headroom to
                                         # restore this from 0.625 without risking a timeout
            use_gaussian=True,
            use_mirroring=True,         # re-enabled flip-based TTA; helps small/boundary regions
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

    Called each time the /invoke endpoint is hit. Reads input from /input,
    runs inference with the ResEnc model, and writes outputs to /output.
    """
    t_start = time.time()
    _log(f"")
    _log(f"══════════════════════════════════════════════════")
    _log(f"  INFERENCE STARTED")
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
    # 1. Locate the input T1 MRI scan (supports .mha, .nii.gz, .nii, .mhd, etc.)
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
        # Fallback: search recursively anywhere under /input
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


    # Read the original image to preserve metadata (spacing, origin, direction)
    _log(f"[Step 2/6] Reading and preprocessing image ...")
    t_read = time.time()
    original_sitk_image = sitk.ReadImage(input_image_path)

    # ──────────────────────────────────────────────────────────────────────
    # 2. Load image as numpy array for in-memory prediction
    #    (avoids writing/reading multi-GB .npz files to disk)
    # ──────────────────────────────────────────────────────────────────────
    # SimpleITK reads in (z, y, x) order, which is what nnU-Net expects
    image_npy = sitk.GetArrayFromImage(original_sitk_image).astype(np.float32)
    # nnU-Net expects shape (channels, z, y, x) for single-modality
    image_npy = image_npy[np.newaxis]  # (1, z, y, x)

    # nnU-Net only needs 'spacing' in image_properties (in SimpleITK x,y,z order)
    spacing = original_sitk_image.GetSpacing()  # (x, y, z)
    image_properties = {"spacing": list(spacing)}

    voxels = image_npy.shape[1] * image_npy.shape[2] * image_npy.shape[3]
    read_elapsed = time.time() - t_read
    _log(f"  Image shape: {image_npy.shape} ({voxels:,} voxels)")
    _log(f"  Spacing: {spacing}")
    _log(f"  Image loaded in {read_elapsed:.2f}s")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Run prediction and collect softmax probabilities (in-memory)
    # ──────────────────────────────────────────────────────────────────────
    all_softmax = []

    for pred_idx, pred_info in enumerate(predictors):
        label = pred_info["label"]
        predictor = pred_info["predictor"]
        _log(f"")
        _log(f"[Step 3/6] Running inference: {label} (model {pred_idx+1}/{len(predictors)}, {len(FOLDS)} folds) ...")
        _log(f"  Preprocessing + sliding-window inference in progress ...")

        t_pred = time.time()

        # predict_single_npy_array returns (segmentation, probabilities)
        # when output_file_truncated=None and save_or_return_probabilities=True
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
    # 4. Ensemble: average softmax probabilities (if multiple models)
    # ──────────────────────────────────────────────────────────────────────
    _log(f"")
    _log(f"[Step 4/6] Ensembling predictions ...")
    t_ensemble = time.time()
    if len(all_softmax) == 1:
        ensemble_softmax = all_softmax[0]
        _log(f"  Single model — using softmax directly: {ensemble_softmax.shape}")
    else:
        _log(f"  Averaging {len(all_softmax)} softmax arrays ...")
        ensemble_softmax = np.mean(all_softmax, axis=0)  # shape: (n_classes, D, H, W)
        _log(f"  Ensemble softmax shape: {ensemble_softmax.shape}")
    _log(f"  ✓ Ensembling done ({time.time() - t_ensemble:.2f}s)")

    # ──────────────────────────────────────────────────────────────────────
    # 5. Generate binary segmentation (argmax) and probability map
    # ──────────────────────────────────────────────────────────────────────
    _log(f"")
    _log(f"[Step 5/6] Generating segmentation and probability map ...")
    t_post = time.time()

    binary_seg = np.argmax(ensemble_softmax, axis=0).astype(np.uint8)

    # Apply nnU-Net's own postprocessing if defined (currently a no-op — none
    # of your trained configs had active postprocessing functions)
    if pp_fns:
        _log(f"  Applying postprocessing to binary segmentation ...")
        try:
            from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing
            binary_seg = apply_postprocessing(binary_seg, pp_fns, pp_kwargs)
            _log(f"  ✓ Postprocessing applied successfully")
        except Exception as e:
            _log(f"  Warning: Failed to apply postprocessing: {e}", level="WARN")

    # Voxel-count FP suppression — drop only tiny spurious components,
    # leave real small lesions alone (see MIN_COMPONENT_VOXELS comment above)
    _log(f"  Removing components smaller than {MIN_COMPONENT_VOXELS} voxels ...")
    binary_seg, n_removed = _remove_small_components(binary_seg, MIN_COMPONENT_VOXELS)
    if n_removed:
        _log(f"  ✓ Removed {n_removed} small component(s) below {MIN_COMPONENT_VOXELS}-voxel threshold")
    else:
        _log(f"  No components below threshold — nothing removed")

    # Probability map = foreground class probability (class index 1)
    prob_map = ensemble_softmax[1].astype(np.float32)

    fg_voxels = int(np.sum(binary_seg == 1))
    _log(f"  Binary seg: unique={np.unique(binary_seg)}, foreground={fg_voxels:,} voxels")
    _log(f"  Prob map: min={prob_map.min():.4f}, max={prob_map.max():.4f}, mean={prob_map.mean():.4f}")
    _log(f"  ✓ Segmentation generated ({time.time() - t_post:.2f}s)")

    # ──────────────────────────────────────────────────────────────────────
    # 6. Write outputs as MetaImage (.mha) files with preserved spatial metadata
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