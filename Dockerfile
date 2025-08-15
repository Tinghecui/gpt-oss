FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3.12 python3-pip git-core curl build-essential cmake && apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install uv --break-system-packages

RUN uv venv --python 3.12 --seed --directory / --prompt workspace workspace-lib
RUN echo "source /workspace-lib/bin/activate" >> /root/.bash_profile

SHELL [ "/bin/bash", "--login", "-c" ]

ENV UV_CONCURRENT_BUILDS=8
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV UV_LINK_MODE=copy

RUN mkdir -p /app/libs

# absolutely required
RUN git clone https://github.com/openai/triton.git /app/libs/triton
WORKDIR /app/libs/triton
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r python/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -e . --verbose --no-build-isolation
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -e python/triton_kernels --no-deps

RUN git clone -b rc1 --depth 1 https://github.com/zyongye/vllm.git /app/libs/vllm
WORKDIR /app/libs/vllm
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r requirements/build.txt
RUN --mount=type=cache,target=/root/.cache/uv uv pip install flashinfer-python==0.2.10
RUN --mount=type=cache,target=/root/.cache/uv uv pip uninstall pytorch-triton
RUN --mount=type=cache,target=/root/.cache/uv uv pip install triton==3.4.0 mcp openai_harmony "transformers[torch]"
#RUN --mount=type=cache,target=/root/.cache/uv uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# torch 2.8
RUN --mount=type=cache,target=/root/.cache/uv uv pip install torch torchvision
RUN python use_existing_torch.py
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-build-isolation -e . -v

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