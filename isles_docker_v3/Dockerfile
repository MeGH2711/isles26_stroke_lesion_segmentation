FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS isles26_ensemble_amd64
# ISLES 2026 Docker submission - Optimized Ensembled nnU-Net (Baseline + ResEnc M)
#
# Base image notes:
#   - cuda12.6 is required for the GPU instances used on Grand Challenge
#   - The 'runtime' image is sufficient for inference

# Ensures that Python output to stdout/stderr is not buffered
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Create a virtualenv that inherits system-site-packages (torch, cuda libs)
RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

# Install Python dependencies
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Copy custom trainer into nnunetv2 package so nnU-Net finds it
COPY --chown=user:user custom_trainers/logging_trainer.py /tmp/logging_trainer.py
RUN NNUNET_DIR=$(python -c "import nnunetv2; import os; print(os.path.dirname(nnunetv2.__file__))") && \
    cp /tmp/logging_trainer.py "${NNUNET_DIR}/training/nnUNetTrainer/variants/logging_trainer.py" && \
    echo "Installed custom trainer to: ${NNUNET_DIR}/training/nnUNetTrainer/variants/logging_trainer.py"

# Copy application code
COPY --chown=user:user app.py /opt/app/
COPY --chown=user:user inference.py /opt/app/

# Port used by Grand Challenge HTTP server
EXPOSE 4743

# Grand Challenge algorithm metadata (MANDATORY for HTTP /invoke API method detection)
ARG BUILD_DATE=""
LABEL org.grand-challenge.api-method="invoke"
LABEL nl.grand-challenge.algorithm.description="ISLES 2026 - Optimized Weighted Ensembled nnU-Net (ResEnc M + Baseline)"
LABEL org.opencontainers.image.created="${BUILD_DATE}"

ENTRYPOINT ["python", "-u", "app.py"]
