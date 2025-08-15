FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Install system dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
    python3.12 python3-pip git-core curl build-essential cmake \
    libnccl2 libnccl-dev && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv --break-system-packages

# Setup Python environment
RUN uv venv --python 3.12 --seed --directory / --prompt workspace workspace-lib
RUN echo "source /workspace-lib/bin/activate" >> /root/.bash_profile

SHELL [ "/bin/bash", "--login", "-c" ]

# Environment variables for faster builds
ENV UV_CONCURRENT_BUILDS=4
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV UV_LINK_MODE=copy
ENV MAX_JOBS=4

RUN mkdir -p /app/libs

# Install PyTorch first (stable version)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install pre-built triton instead of building from source
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install triton==3.0.0

# Clone and install vLLM
RUN git clone -b rc1 --depth 1 https://github.com/zyongye/vllm.git /app/libs/vllm
WORKDIR /app/libs/vllm

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements/build.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install flashinfer-python==0.2.10 mcp openai_harmony "transformers[torch]"

# Prepare for vLLM build
RUN python use_existing_torch.py

# Build vLLM with optimizations
RUN --mount=type=cache,target=/root/.cache/uv \
    VLLM_USE_PRECOMPILED=1 MAX_JOBS=4 uv pip install -e . -v

COPY <<-"EOF" /app/entrypoint
#!/bin/bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export TORCH_CUDA_ARCH_LIST=8.6
source /workspace-lib/bin/activate
exec python3 -m vllm.entrypoints.openai.api_server --port 8080 "$@"
EOF

RUN chmod +x /app/entrypoint

EXPOSE 8080

ENTRYPOINT [ "/app/entrypoint" ]