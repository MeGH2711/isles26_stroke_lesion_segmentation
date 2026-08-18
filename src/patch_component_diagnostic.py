"""
patch_component_diagnostic.py

Answers the question that decides what to do next with the instance-aware
loss: "What fraction of ACTUAL training patches contain 2+ distinct GT
lesion components?"

Why this matters: InstanceDiceLoss only re-weights small vs. large lesions
WITHIN a single training patch (connected-component labeling happens per
crop, not per whole case). If most patches only ever contain a single
component, per-component averaging collapses to ordinary Dice for those
patches - the loss would have had nothing to act on for most of training,
which would explain why fold 0 and fold 2 showed no real SMALL-lesion
improvement despite 250 epochs of training.

This pulls patches from nnU-Net's OWN training dataloader (via
nnUNetTrainerInstanceLoss), not a reimplemented sampling routine - so the
oversample_foreground_percent=0.55 behavior, patch size, etc. are exactly
what training actually used, not an approximation.

============================== HOW TO RUN ==============================
    cd to wherever nnUNetv2 is importable (same venv as training), then:

    python patch_component_diagnostic.py

Takes a few minutes - it's just pulling batches from the dataloader, no
GPU forward/backward passes happen.
==========================================================================
"""

import os

# Set these BEFORE importing nnunetv2 - it reads them at import time.
# Matches the "environment variable persistence" hazard from your training
# sessions - setting them here avoids depending on the shell session's state.
os.environ['nnUNet_raw'] = r"C:\Users\Admin\Desktop\meghpatel\isles26\data\nnUNet_raw"
os.environ['nnUNet_preprocessed'] = r"C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_preprocessed"
os.environ['nnUNet_results'] = r"C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results"

import numpy as np
import torch
from scipy.ndimage import label as cc_label
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.training.nnUNetTrainer.variants.instance_trainer import nnUNetTrainerInstanceLoss


# ============================================================================
# CONFIG
# ============================================================================

DATASET_NAME = "Dataset001_ISLES26"
CONFIGURATION = "3d_fullres"
FOLD = 0                      # which fold's dataloader/split to sample from
NUM_BATCHES_TO_SAMPLE = 200   # batch_size=2 per your plans -> ~400 patches
MIN_COMPONENT_VOXELS = 3      # matches InstanceDiceLoss's default - filters label noise

PREPROCESSED_DIR = os.path.join(
    os.environ['nnUNet_preprocessed'], DATASET_NAME
)
PLANS_JSON_PATH = os.path.join(PREPROCESSED_DIR, "nnUNetPlans.json")
DATASET_JSON_PATH = os.path.join(PREPROCESSED_DIR, "dataset.json")


def main():
    print("=" * 70)
    print("Patch component co-occurrence diagnostic")
    print("=" * 70)
    print(f"Loading plans from: {PLANS_JSON_PATH}")
    print(f"Loading dataset.json from: {DATASET_JSON_PATH}")

    plans = load_json(PLANS_JSON_PATH)
    dataset_json = load_json(DATASET_JSON_PATH)

    # Your installed nnU-Net version pops config flags directly out of the
    # plans dict in __init__ (continue_training, at minimum) rather than
    # taking them as constructor args. We're not resuming anything here, so
    # False is correct.
    plans.setdefault("continue_training", False)

    # CPU device - we're not running the network, just pulling patches from
    # the dataloader, so no need to touch the GPU for this.
    trainer = nnUNetTrainerInstanceLoss(
        plans=plans,
        configuration=CONFIGURATION,
        fold=FOLD,
        dataset_json=dataset_json,
        device=torch.device('cpu'),
    )
    trainer.initialize()  # builds network (unused here), optimizer, and loss

    # dataloaders aren't built inside initialize() - they're built by
    # get_dataloaders(), normally called just before the training loop starts.
    # We just want the training generator directly.
    dl, _dl_val = trainer.get_dataloaders()

    print(f"Sampling {NUM_BATCHES_TO_SAMPLE} batches "
          f"(patch_size={trainer.configuration_manager.patch_size}, "
          f"oversample_foreground_percent={trainer.oversample_foreground_percent})...")

    component_counts_per_patch = []
    component_counts_per_patch_26conn = []
    conn26_structure = np.ones((3, 3, 3), dtype=int)  # full 26-connectivity

    for i in range(NUM_BATCHES_TO_SAMPLE):
        batch = next(dl)
        target = batch['target']
        # Deep supervision gives a list of targets per scale - index 0 is
        # full resolution, the only one InstanceDiceLoss actually uses.
        seg = target[0] if isinstance(target, (list, tuple)) else target
        seg_np = seg.numpy() if torch.is_tensor(seg) else np.asarray(seg)

        for b in range(seg_np.shape[0]):
            fg_mask = seg_np[b, 0] > 0
            if fg_mask.sum() == 0:
                component_counts_per_patch.append(0)
                continue

            labeled, n_components = cc_label(fg_mask)
            real_components = sum(
                1 for comp_id in range(1, n_components + 1)
                if (labeled == comp_id).sum() >= MIN_COMPONENT_VOXELS
            )
            component_counts_per_patch.append(real_components)

            labeled_26, n_components_26 = cc_label(fg_mask, structure=conn26_structure)
            real_components_26 = sum(
                1 for comp_id in range(1, n_components_26 + 1)
                if (labeled_26 == comp_id).sum() >= MIN_COMPONENT_VOXELS
            )
            component_counts_per_patch_26conn.append(real_components_26)

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{NUM_BATCHES_TO_SAMPLE} batches sampled")

    counts = np.array(component_counts_per_patch)
    counts_26 = np.array(component_counts_per_patch_26conn)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total patches sampled: {len(counts)}")
    print(f"  --- 6-connectivity (scipy default) ---")
    print(f"  0 components (empty patch):     {(counts == 0).sum():4d}  ({100 * (counts == 0).mean():.1f}%)")
    print(f"  Exactly 1 component:             {(counts == 1).sum():4d}  ({100 * (counts == 1).mean():.1f}%)")
    print(f"  2+ components:                   {(counts >= 2).sum():4d}  ({100 * (counts >= 2).mean():.1f}%)")
    non_empty = counts[counts > 0]
    if len(non_empty) > 0:
        print(f"  Mean components per non-empty patch: {non_empty.mean():.2f}")
        print(f"  Max components seen in a single patch: {non_empty.max()}")

    print(f"  --- 26-connectivity (full corner-connectivity - catches fragmentation) ---")
    print(f"  0 components (empty patch):     {(counts_26 == 0).sum():4d}  ({100 * (counts_26 == 0).mean():.1f}%)")
    print(f"  Exactly 1 component:             {(counts_26 == 1).sum():4d}  ({100 * (counts_26 == 1).mean():.1f}%)")
    print(f"  2+ components:                   {(counts_26 >= 2).sum():4d}  ({100 * (counts_26 >= 2).mean():.1f}%)")
    non_empty_26 = counts_26[counts_26 > 0]
    if len(non_empty_26) > 0:
        print(f"  Mean components per non-empty patch: {non_empty_26.mean():.2f}")
        print(f"  Max components seen in a single patch: {non_empty_26.max()}")
    print("=" * 70)

    pct_multi = 100 * (counts >= 2).mean()
    pct_multi_26 = 100 * (counts_26 >= 2).mean()
    drop = pct_multi - pct_multi_26

    print()
    if drop > 20:
        print(f"INTERPRETATION: 2+-component rate drops from {pct_multi:.1f}% (6-conn) to "
              f"{pct_multi_26:.1f}% (26-conn) - a {drop:.1f}pt drop. This suggests a real "
              f"chunk of the '2+ components' patches were actually ONE irregularly-shaped "
              f"lesion getting fragmented by 6-connectivity, not genuinely separate lesions. "
              f"The instance loss (built with scipy's 6-connectivity default) has likely been "
              f"treating fragments of single lesions as if they were independent instances, "
              f"which dilutes/distorts the intended signal rather than helping with true "
              f"multi-lesion imbalance. Fix: switch InstanceDiceLoss to 26-connectivity "
              f"(structure=np.ones((3,3,3))) and re-run this diagnostic to confirm the "
              f"26-conn numbers reflect genuine multi-lesion patches before retraining.")
    else:
        print(f"INTERPRETATION: 2+-component rate barely changes between 6-conn ({pct_multi:.1f}%) "
              f"and 26-conn ({pct_multi_26:.1f}%) - so this IS mostly genuine multi-lesion "
              f"co-occurrence, not a connectivity artifact. That rules out the sampling-gap "
              f"hypothesis: the loss had real material to work with. The flat SMALL-lesion "
              f"result is more likely explained by alpha_max=0.4 being too weak relative to "
              f"the regional loss, or 250 epochs not being enough for the effect to show up "
              f"in validation Dice. Worth trying a higher alpha_max (e.g. 0.7-1.0) on one "
              f"fold before anything more structural.")


if __name__ == "__main__":
    main()