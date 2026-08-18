import re
import csv

INPUT_FILE = "training_log_2026_7_1_11_22_31.txt"
OUTPUT_FILE = "training_log_fold03.csv"

# Patterns to extract each field
epoch_pattern       = re.compile(r"Epoch (\d+)")
lr_pattern          = re.compile(r"Current learning rate: ([0-9.eE+\-]+)")
train_loss_pattern  = re.compile(r"train_loss\s+([0-9.\-eE+]+)")
val_loss_pattern    = re.compile(r"val_loss\s+([0-9.\-eE+]+)")
pseudo_dice_pattern = re.compile(r"Pseudo dice \[np\.float32\(([0-9.]+)\)\]")
epoch_time_pattern  = re.compile(r"Epoch time:\s+([0-9.]+) s")
ema_dice_pattern    = re.compile(r"New best EMA pseudo Dice:\s+([0-9.eE+\-]+)")

rows = []

with open(INPUT_FILE, "r") as f:
    lines = f.readlines()

# Walk through lines and collect per-epoch data
current = {}

for line in lines:
    line = line.strip()

    # Strip the timestamp prefix if present (e.g. "2026-06-25 14:51:07.608946: ")
    # so patterns match cleanly
    content = re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+:\s*", "", line)

    if epoch_pattern.search(content):
        # Save previous epoch if complete enough to record
        if current and "epoch" in current:
            rows.append(current)
        current = {"epoch": int(epoch_pattern.search(content).group(1)),
                   "ema_pseudo_dice": ""}  # default empty if no new best

    elif lr_pattern.search(content):
        current["learning_rate"] = float(lr_pattern.search(content).group(1))

    elif train_loss_pattern.search(content):
        current["train_loss"] = float(train_loss_pattern.search(content).group(1))

    elif val_loss_pattern.search(content):
        current["val_loss"] = float(val_loss_pattern.search(content).group(1))

    elif pseudo_dice_pattern.search(content):
        current["pseudo_dice"] = float(pseudo_dice_pattern.search(content).group(1))

    elif epoch_time_pattern.search(content):
        current["epoch_time_s"] = float(epoch_time_pattern.search(content).group(1))

    elif ema_dice_pattern.search(content):
        current["ema_pseudo_dice"] = float(ema_dice_pattern.search(content).group(1))

# Don't forget the last epoch
if current and "epoch" in current:
    rows.append(current)

COLUMNS = ["epoch", "learning_rate", "train_loss", "val_loss",
           "pseudo_dice", "epoch_time_s", "ema_pseudo_dice"]

with open(OUTPUT_FILE, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in COLUMNS})

print(f"Done! {len(rows)} epochs written to '{OUTPUT_FILE}'.")
