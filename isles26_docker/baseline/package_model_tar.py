"""
package_model_tar.py

Packages isles26_docker/baseline/model/ into isles26_docker/baseline/model.tar.gz for Grand Challenge.
"""

import json
import os
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "model"
OUTPUT_TAR = Path(__file__).resolve().parent / "model.tar.gz"


def main():
    version_file = MODEL_DIR / "model_version.json"
    version_data = {
        "version": "1.0-baseline-only",
        "description": "Clean inference weights for ISLES26 nnU-Net Baseline (5-fold ensemble)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "5-fold Baseline (nnUNetTrainer / nnUNetPlans / 3d_fullres)",
    }
    with open(version_file, "w", encoding="utf-8") as f:
        json.dump(version_data, f, indent=2)
    print(f"Created {version_file}")

    print(f"Packaging {MODEL_DIR} into {OUTPUT_TAR} ...")
    t0 = time.time()

    with tarfile.open(OUTPUT_TAR, "w:gz") as tar:
        for item in sorted(MODEL_DIR.iterdir()):
            print(f"  Adding {item.name} ...")
            tar.add(item, arcname=item.name)

    size_mb = OUTPUT_TAR.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"\n[OK] Successfully created {OUTPUT_TAR} ({size_mb:.1f} MB) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
