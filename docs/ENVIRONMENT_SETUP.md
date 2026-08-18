# Environment Setup Guide

## System Requirements
- OS: Windows 11 / Linux (Ubuntu 20.04+)
- Python 3.11
- CUDA 11.8 or 12.1+
- PyTorch 2.x

## Setup Instructions

### 1. Python Environment
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux:
source venv/bin/activate
```

### 2. Install PyTorch & Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core medical imaging packages
pip install nnunetv2 monai SimpleITK nibabel scikit-image scikit-learn matplotlib pandas
```

### 3. Configure nnU-Net Environment Variables
Set the mandatory directory environment variables for nnU-Net v2:

#### PowerShell:
```powershell
$env:nnUNet_raw          = "C:\path\to\data\nnunet_raw"
$env:nnUNet_preprocessed = "C:\path\to\data\nnunet_preprocessed"
$env:nnUNet_results      = "C:\path\to\data\nnunet_results"
```

#### Command Prompt:
```cmd
set nnUNet_raw=C:\path\to\data\nnunet_raw
set nnUNet_preprocessed=C:\path\to\data\nnunet_preprocessed
set nnUNet_results=C:\path\to\data\nnunet_results
```
