FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS isles26_ensemble_amd64
# ISLES 2026 Docker submission - Ensembled nnU-Net (Baseline + ResEnc)
#
# Base image notes:
#   - cuda12.6 is required for the T4 GPU instances used on Grand Challenge
#   - The 'runtime' image is sufficient for inference; 'devel' is only needed
#     if you compile custom CUDA kernels

# Ensures that Python output to stdout/stderr is not buffered
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Create a virtualenv that inherits the system-site-packages (i.e. torch, cuda libs)
# This avoids PEP 668 externally-managed-environment errors on newer PyTorch images
# --without-pip is needed because the base image does not ship ensurepip;
# pip is inherited from system-site-packages instead.
RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

# Install Python dependencies
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Copy the custom trainer into the nnunetv2 package so nnU-Net can find it
# We dynamically locate the nnunetv2 package directory since the Python
# version in the base image may vary
COPY --chown=user:user custom_trainers/logging_trainer.py /tmp/logging_trainer.py
RUN NNUNET_DIR=$(python -c "import nnunetv2; import os; print(os.path.dirname(nnunetv2.__file__))") && \
    cp /tmp/logging_trainer.py "${NNUNET_DIR}/training/nnUNetTrainer/variants/logging_trainer.py" && \
    echo "Installed custom trainer to: ${NNUNET_DIR}/training/nnUNetTrainer/variants/logging_trainer.py"

# Copy application code
COPY --chown=user:user app.py /opt/app/
COPY --chown=user:user inference.py /opt/app/

# Port used by the Grand Challenge HTTP server
EXPOSE 4743

# Grand Challenge algorithm metadata (MANDATORY for HTTP /invoke API method detection)
ARG BUILD_DATE=""
LABEL org.grand-challenge.api-method="invoke"
LABEL nl.grand-challenge.algorithm.description="ISLES 2026 - Ensembled nnU-Net (ResEnc M) stroke lesion segmentation"
LABEL org.opencontainers.image.created="${BUILD_DATE}"

ENTRYPOINT ["python", "-u", "app.py"]

