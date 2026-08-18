# Rclone Remote Backup & Sync Setup

This guide details configuring automated model weights backup to Google Drive / cloud storage using Rclone.

## 1. Installation
Download Rclone from [rclone.org](https://downloads.rclone.org/rclone-current-windows-amd64.zip) and place `rclone.exe` in `C:\rclone` or in your system PATH.

## 2. Configure Remote
```powershell
rclone config
```
1. Create a new remote: `n`
2. Name it `gdrive`
3. Select storage type `drive` (Google Drive)
4. Follow authentication prompts to complete OAuth connection.

## 3. Automated Weight Sync
The repository includes `sync_weights.py` which takes periodic local snapshots of training checkpoints in `data/nnunet_results/` and safely uploads them to the remote without interrupting ongoing training:

```bash
python sync_weights.py
```
