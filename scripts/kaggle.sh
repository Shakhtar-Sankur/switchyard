#!/bin/bash
# switchyard on Kaggle (Settings: Accelerator "GPU T4 x2", Internet on). Paste everything from
# "== switchyard" to "== done" back into the chat.
#   RUN=kernels (default): build the CUDA kernels, run every test on the GPU, benchmark one MoE layer
set -e
RUN=${RUN:-kernels}
cd /kaggle/working 2>/dev/null || cd /tmp
rm -rf switchyard && git clone -q https://github.com/Shakhtar-Sankur/switchyard && cd switchyard
echo "== switchyard ($RUN): $(git log -1 --format='%h %s')"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=7.5 PYTHONPATH=$PWD
nvcc --version | tail -1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'peer access 0<->1:', torch.cuda.device_count() > 1 and torch.cuda.can_device_access_peer(0, 1))"
pip install -q transformers safetensors 2>&1 | tail -1 || true

echo "== build"
python -c "from switchyard import kernels; kernels.ext(); print('built')" 2>&1 | tail -3
echo "== tests"
python -m pytest -q tests 2>&1 | tail -15
echo "== benchmark: one OLMoE MoE layer"
python bench/kernels.py
echo "== done"
