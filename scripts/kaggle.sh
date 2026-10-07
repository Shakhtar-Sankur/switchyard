#!/bin/bash
# switchyard on Kaggle (Settings: Accelerator "GPU T4 x2", Internet on). Paste everything from
# "== switchyard" to "== done" back into the chat.
#   RUN=all (default): kernels, expert parallelism, OLMoE-1B-7B serving and load balance (~35 minutes)
#   RUN=kernels | ep | serve | balance: one part
set -e
RUN=${RUN:-all}
cd /kaggle/working 2>/dev/null || cd /tmp
rm -rf switchyard && git clone -q https://github.com/Shakhtar-Sankur/switchyard && cd switchyard
echo "== switchyard ($RUN): $(git log -1 --format='%h %s')"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=7.5 PYTHONPATH=$PWD
nvcc --version | tail -1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'peer access 0<->1:', torch.cuda.device_count() > 1 and torch.cuda.can_device_access_peer(0, 1))"
pip install -q transformers safetensors huggingface_hub 2>&1 | tail -1 || true

echo "== build"
python -c "from switchyard import kernels; kernels.ext(); print('built')" 2>&1 | tail -3
echo "== tests"
python -m pytest -q tests 2>&1 | tail -15
if [ "$RUN" = all ] || [ "$RUN" = kernels ]; then
  echo "== benchmark: one OLMoE MoE layer on one GPU"
  python bench/kernels.py
fi
if [ "$RUN" = all ] || [ "$RUN" = ep ]; then
  echo "== benchmark: the layer with its experts over two GPUs (switchyard peer-to-peer)"
  python bench/ep.py
  echo "== benchmark: same, NCCL all_to_all as the transport"
  python bench/ep_nccl.py 2>&1 | grep '^{\|Error' || true
fi
FILTER="Loading\|it/s\]\|CUDAEvent.h\|Download\|Reconstruct\|Fetching\|clean_up_tokenization\|Token indices"
if [ "$RUN" = all ] || [ "$RUN" = serve ] || [ "$RUN" = balance ]; then
  python - <<'PY' 2>&1 | grep -v "$FILTER"
from huggingface_hub import snapshot_download
snapshot_download("allenai/OLMoE-1B-7B-0924", local_dir="/tmp/olmoe", allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("downloaded OLMoE-1B-7B")
PY
fi
if [ "$RUN" = all ] || [ "$RUN" = serve ]; then
  echo "== OLMoE-1B-7B on two T4s"
  SWITCHYARD_THREADED=1 python bench/serve.py /tmp/olmoe 32 2>&1 | grep -v "$FILTER" | tail -12
fi
if [ "$RUN" = all ] || [ "$RUN" = balance ]; then
  echo "== load balance on WikiText-2 (OLMoE-1B-7B, two T4s)"
  python bench/balance.py /tmp/olmoe 2>&1 | grep -v "$FILTER" | tail -12
fi
echo "== done"
