#!/bin/bash
# Second follow-up: compares FlashInfer and the test's float16 reference against a float64 reference,
# broken down by sequence, next to PyTorch's own float16 attention (scaled_dot_product_attention).
#   !cd /tmp && rm -rf sy && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/switchyard sy && bash sy/upstream/kaggle_vllm_turing_fp64.sh
# About 10 minutes.
set -o pipefail
SY=$(cd "$(dirname "$0")" && pwd)
VLLM_SHA=193922d6ff5f7e213eacadd3cb7dd96612b30ed3
W=/tmp/vllm-turing
echo "== vLLM FlashInfer T4 float64 check on $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
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
old = "DTYPES = [torch.bfloat16]\n"
if old in s:
    s = s.replace(old, "DTYPES = (\n    [torch.bfloat16]\n"
                  "    if not current_platform.is_cuda() or current_platform.has_device_capability(80)\n"
                  "    else [torch.float16]\n)\n")
    open(p, "w").write(s)
print("float16 below SM80")
PATCH
echo "== 1. FlashInfer and the test's float16 reference, each against float64"
python $SY/flashinfer_fp64_check.py 2>&1 | grep -E '^\{|Error|error' | head -30
echo "== done: copy from '== vLLM FlashInfer T4 float64 check' to here and send it back"
