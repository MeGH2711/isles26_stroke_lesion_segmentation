"""
save_docker_image.py

Saves Docker image isles26-baseline-algorithm to isles26-baseline-algorithm.tar.gz
using gzip compression.
"""

import gzip
import shutil
import subprocess
import time
from pathlib import Path

IMAGE_TAG = "isles26-baseline-algorithm"
OUTPUT_FILE = Path(__file__).resolve().parent / f"{IMAGE_TAG}.tar.gz"


def main():
    print(f"Exporting Docker image '{IMAGE_TAG}' to {OUTPUT_FILE} ...")
    t0 = time.time()

    proc = subprocess.Popen(
        ["docker", "save", IMAGE_TAG],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with gzip.open(OUTPUT_FILE, "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(proc.stdout, f_out)

    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"docker save failed with code {proc.returncode}: {err}")

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"\n[OK] Successfully saved {OUTPUT_FILE} ({size_mb:.1f} MB) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
