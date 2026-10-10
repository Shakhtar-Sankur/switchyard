#!/bin/bash
# vLLM skips test_flashinfer_prefill_with_paged_fp8_kv ("TODO: fix the accuracy issue"). Un-skipped on
# a T4, without and then with causal=True in its plan() call: is the missing causal flag the issue?
#   !cd /tmp && rm -rf sy && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/switchyard sy && bash sy/upstream/kaggle_vllm_causal_fp8.sh
# About 10 minutes.
set -o pipefail
SY=$(cd "$(dirname "$0")" && pwd)
VLLM_SHA=193922d6ff5f7e213eacadd3cb7dd96612b30ed3
W=/tmp/vllm-turing
echo "== vLLM FlashInfer fp8 prefill test, un-skipped, on T4 on $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
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
P="-q -rf --tb=line -p no:cacheprovider --noconftest -p kernels_fixture"
patch() {  # $1 = 1 to add causal=True
python - "$1" <<'PATCH'
import sys
p = "tests/kernels/attention/test_flashinfer.py"
s = open(p).read()
s = s.replace('    pytest.skip("TODO: fix the accuracy issue")\n', "")
s = s.replace("DTYPES = [torch.bfloat16]\n", "DTYPES = (\n    [torch.bfloat16]\n"
              "    if not current_platform.is_cuda() or current_platform.has_device_capability(80)\n"
              "    else [torch.float16]\n)\n")
b = "        block_size,\n        q_data_type=dtype,\n        kv_data_type=kv_cache_dtype,\n"
fix = "        causal=True,  # as ref_paged_attn; plan() defaults to non-causal\n"
if sys.argv[1] == "1" and fix not in s:
    assert s.count(b) == 1, "test file changed upstream"
    s = s.replace(b, b.replace("        q_data_type", fix + "        q_data_type"))
open(p, "w").write(s)
print("fp8 prefill test un-skipped; causal=True:", fix in s)
PATCH
}
git checkout -q -- $T
patch 0
echo "== 1. un-skipped, as on main (no causal flag)"
python -m pytest $P "$T" -k "prefill_with_paged_fp8_kv and not num_heads2" 2>&1 | grep -E "^FAILED|passed|failed|Mismatched|Greatest absolute" | tail -8
patch 1
echo "== 2. un-skipped, with causal=True"
python -m pytest $P "$T" -k "prefill_with_paged_fp8_kv and not num_heads2" 2>&1 | grep -E "^FAILED|passed|failed|Mismatched|Greatest absolute" | tail -8
echo "== done: copy from '== vLLM FlashInfer fp8 prefill test' to here and send it back"
