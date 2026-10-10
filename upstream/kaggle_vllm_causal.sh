#!/bin/bash
# The fix for vLLM's FlashInfer prefill tests: plan() gets causal=True, as their reference is causal.
# On a T4 (float16): FlashInfer against float64 per sequence, then both prefill tests, as CI runs them.
#   !cd /tmp && rm -rf sy && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/switchyard sy && bash sy/upstream/kaggle_vllm_causal.sh
# About 15 minutes.
set -o pipefail
SY=$(cd "$(dirname "$0")" && pwd)
VLLM_SHA=193922d6ff5f7e213eacadd3cb7dd96612b30ed3
W=/tmp/vllm-turing
echo "== vLLM FlashInfer prefill tests, causal fix, on T4 on $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
if [ ! -d $W/vllm/.git ]; then
  rm -rf $W && mkdir -p $W && cd $W
  git init -q vllm && git -C vllm remote add origin https://github.com/vllm-project/vllm
  git -C vllm fetch -q --depth 1 origin $VLLM_SHA && git -C vllm checkout -q FETCH_HEAD
  python -c "import torch; print('torch==' + torch.__version__)" > constraints.txt
  pip install -q -c constraints.txt -r vllm/requirements/common.txt "flashinfer-python==0.7.0.post1" pytest 2>&1 | grep -v -i "dependency resolver\|incompatible\|^$" | tail -2
  touch vllm/vllm/_C_stable_libtorch.py
  printf '__version__ = version = "0.0.0+src"\n__version_tuple__ = version_tuple = (0, 0, 0)\n' > vllm/vllm/_version.py
  mkdir -p meta/vllm-0.0.0+src.dist-info && printf 'Metadata-Version: 2.1\nName: vllm\nVersion: 0.0.0+src\n' > meta/vllm-0.0.0+src.dist-info/METADATA
  printf 'import pytest\nimport torch\n\n\n@pytest.fixture(autouse=True)\ndef reset_default_torch_device():\n    yield\n    torch.set_default_device(None)\n' > meta/kernels_fixture.py
fi
cd $W
LIBCUDA=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 /{print $NF; exit}')
[ -z "$LIBCUDA" ] && LIBCUDA=$(find / -name 'libcuda.so.1' -not -path '*/proc/*' 2>/dev/null | head -1)
mkdir -p $W/libcuda && ln -sf "$LIBCUDA" $W/libcuda/libcuda.so
export LIBRARY_PATH=$W/libcuda${LIBRARY_PATH:+:$LIBRARY_PATH}
export PYTHONPATH=$W/vllm:$W/meta
cd vllm
T=tests/kernels/attention/test_flashinfer.py
python - <<'PATCH'
p = "tests/kernels/attention/test_flashinfer.py"
s = open(p).read()
# 1. the fix under test: causal=True in both prefill plan() calls, exactly as in the PR
a = "        block_size,\n        window_left=sliding_window - 1 if sliding_window is not None else -1,\n"
b = "        block_size,\n        q_data_type=dtype,\n        kv_data_type=kv_cache_dtype,\n"
assert s.count(a) == 1 and s.count(b) == 1, "test file changed upstream"
fix = "        causal=True,  # as ref_paged_attn; plan() defaults to non-causal\n"
s = s.replace(a, a.replace("        window_left", fix + "        window_left"))
s = s.replace(b, b.replace("        q_data_type", fix + "        q_data_type"))
# 2. float16 on this T4 (FlashInfer has no bfloat16 below SM80)
s = s.replace("DTYPES = [torch.bfloat16]\n", "DTYPES = (\n    [torch.bfloat16]\n"
              "    if not current_platform.is_cuda() or current_platform.has_device_capability(80)\n"
              "    else [torch.float16]\n)\n")
open(p, "w").write(s)
print("causal=True in both prefill tests; float16 below SM80")
PATCH
echo "== 1. FlashInfer (now causal) and the test's reference, each against float64"
python $SY/flashinfer_fp64_check.py 2>&1 | grep -E '^\{|Error|error' | head -12
echo "== 2. both prefill tests, as CI runs them"
python -m pytest -q -rf --tb=line -p no:cacheprovider --noconftest -p kernels_fixture "$T" -k "prefill and not num_heads2" 2>&1 | grep -E "^FAILED|passed|failed|^E " | tail -12
echo "== done: copy from '== vLLM FlashInfer prefill tests' to here and send it back"
