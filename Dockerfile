FROM python:3.11-slim

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (pip cache mount speeds up rebuilds)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Verify PyTorch
RUN python -c "import torch; print(f'PyTorch {torch.__version__} on {\"cuda\" if torch.cuda.is_available() else \"cpu\"}')"

# Pre-download DINOv2 weights so cold starts don't hit HuggingFace
RUN python -c "\
from transformers import AutoModel, AutoImageProcessor; \
AutoImageProcessor.from_pretrained('facebook/dinov2-base'); \
AutoModel.from_pretrained('facebook/dinov2-base')" \
    || echo "DINOv2 pre-download skipped (no internet at build time)"

# Application code
COPY *.py *.yaml ./

# Model weights (baked into image for fast cold starts)
COPY models/ ./models/

# Runtime dirs
RUN mkdir -p input_videos /tmp/rk_output /tmp/rk_results

ENV PYTHONUNBUFFERED=1
ENV NO_DISPLAY=1
# Path to YOLO weights baked into the image. Override at runtime:
#   docker run -e MODEL_PATH=/app/models/other.pt ...
ENV MODEL_PATH=/app/models/Rk_trained_model.pt

CMD ["python", "run_wrapper.py"]
