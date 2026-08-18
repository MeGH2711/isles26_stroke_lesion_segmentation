import subprocess
import time
import shutil
import os
from datetime import datetime
RCLONE_EXE     = r"C:\rclone\rclone.exe"
LOCAL_WEIGHTS  = r"C:\Users\Admin\Desktop\meghpatel\isles26\data\nnunet_results\Dataset001_ISLES26\nnUNetTrainerWithLogging__nnUNetResEncUNetMPlans__3d_fullres"
TEMP_DIR       = r"C:\Users\Admin\Desktop\meghpatel\isles26\sync_temp"
GDRIVE_WEIGHTS = "gdrive2:isles26-1f/nnunet_results/Dataset001_ISLES26/nnUNetTrainerWithLogging__nnUNetResEncUNetMPlans__3d_fullres"
SYNC_INTERVAL_MINUTES = 5
def print_progress_bar(current, total, bar_length=40):
    fraction = current / total if total else 1
    filled = int(bar_length * fraction)
    bar = "#" * filled + "-" * (bar_length - filled)
    print(f"\r  [{bar}] {current}/{total} files", end="", flush=True)

def make_snapshot(src, temp):
    """Copy entire nnunet_results to a temp folder first, showing a progress bar"""
    if os.path.exists(temp):
        shutil.rmtree(temp)
    print(f"  Taking local snapshot...", flush=True)

    # Count total files first so we know the denominator for the progress bar
    total_files = sum(len(files) for _, _, files in os.walk(src))
    copied = {"count": 0}

    def copy_with_progress(s, d):
        result = shutil.copy2(s, d)
        copied["count"] += 1
        print_progress_bar(copied["count"], total_files)
        return result

    shutil.copytree(src, temp, copy_function=copy_with_progress)
    print(flush=True)  # move to next line after the progress bar
    print(f"  Snapshot ready!", flush=True)
def sync(local, remote):
    process = subprocess.Popen(
        [RCLONE_EXE, "copy", local, remote,
         "--checksum",
         "--verbose",
         "--progress",
         "--stats", "5s",
         "--stats-one-line"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    return process.returncode == 0
def run():
    print("Auto-sync started. Press Ctrl+C to stop.\n")
    while True:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*50}")
        print(f"  Sync started at {ts}")
        print(f"{'='*50}\n")
        try:
            # Step 1: snapshot live files to temp folder
            make_snapshot(LOCAL_WEIGHTS, TEMP_DIR)
            # Step 2: upload from temp (safe, no active writes)
            ok = sync(TEMP_DIR, GDRIVE_WEIGHTS)
            # Step 3: cleanup temp
            shutil.rmtree(TEMP_DIR)
            ts_end = datetime.now().strftime("%H:%M:%S")
            if ok:
                print(f"\n[{ts_end}] Sync complete! Next sync in {SYNC_INTERVAL_MINUTES} min...")
            else:
                print(f"\n[{ts_end}] Sync FAILED. Retrying in {SYNC_INTERVAL_MINUTES} min...")
        except Exception as e:
            print(f"\n[ERROR] {e}")
        time.sleep(SYNC_INTERVAL_MINUTES * 60)
if __name__ == "__main__":
    run()