#!/bin/bash
# FlashInfer's paged attention on a Tesla T4 (SM75), with the version vLLM pins.
# Kaggle: Accelerator "GPU T4 x2" (one T4 is enough), Internet on. Paste from "== flashinfer" to "== done".
set -e
cd /kaggle/working 2>/dev/null || cd /tmp
rm -rf switchyard && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/switchyard
export PATH=/usr/local/cuda/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=7.5 TORCH_CUDA_ARCH_LIST=7.5
# FlashInfer's JIT links its kernels with -lcuda; the image has the driver's libcuda.so.1 but no
# libcuda.so for the linker to find, so give it one.
LIBCUDA=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 /{print $NF; exit}')
[ -z "$LIBCUDA" ] && LIBCUDA=$(find / -name 'libcuda.so.1' -not -path '*/proc/*' 2>/dev/null | head -1)
mkdir -p /tmp/libcuda && ln -sf "$LIBCUDA" /tmp/libcuda/libcuda.so
export LIBRARY_PATH=/tmp/libcuda${LIBRARY_PATH:+:$LIBRARY_PATH}
pip install -q ninja "flashinfer-python==0.7.0.post1" --extra-index-url https://flashinfer.ai/whl/ 2>&1 | grep -v "requires cuda-core\|dependency conflicts" | tail -2 || true
rm -rf ~/.cache/flashinfer   # drop the failed builds of an earlier run
echo "== flashinfer: switchyard $(git -C switchyard log -1 --format=%h) on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), libcuda $LIBCUDA"
echo "   (kernels compile on first use: ~10-25 minutes)"
python switchyard/upstream/flashinfer_sm75_check.py 2>&1 | grep -v "^\s*$" | tail -70
echo "== done"
