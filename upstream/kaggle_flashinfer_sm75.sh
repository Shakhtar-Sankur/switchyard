#!/bin/bash
# FlashInfer's paged attention on a Tesla T4 (SM75), with the version vLLM pins.
# Kaggle: Accelerator "GPU T4 x2", Internet on. Paste from "== flashinfer" to "== done".
set -e
cd /kaggle/working 2>/dev/null || cd /tmp
export PATH=/usr/local/cuda/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=7.5 TORCH_CUDA_ARCH_LIST=7.5
pip install -q ninja "flashinfer-python==0.7.0.post1" --extra-index-url https://flashinfer.ai/whl/ 2>&1 | tail -2 || true
curl -sL https://raw.githubusercontent.com/Shakhtar-Sankur/switchyard/main/upstream/flashinfer_sm75_check.py -o fi_sm75.py
echo "== flashinfer on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (kernels compile on first use: ~10-20 minutes)"
python fi_sm75.py 2>&1 | grep -v "^\s*$" | tail -80
echo "== done"
