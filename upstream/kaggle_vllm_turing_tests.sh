#!/bin/bash
# vLLM's FlashInfer kernel tests on a Tesla T4 (Turing, SM75), before and after running them in
# float16 where bfloat16 is unsupported. Kaggle, GPU T4 (one is enough), Internet on:
#   !cd /tmp && rm -rf sy && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/switchyard sy && bash sy/upstream/kaggle_vllm_turing_tests.sh
# About 40-60 minutes: FlashInfer compiles each kernel variant the first time it is used.
# The tests call FlashInfer directly; vLLM is used from source for its platform and test helpers,
# with an empty stand-in for its compiled extension, which these tests never call.
set -o pipefail
VLLM_SHA=193922d6ff5f7e213eacadd3cb7dd96612b30ed3
W=/tmp/vllm-turing && rm -rf $W && mkdir -p $W && cd $W

echo "== vLLM FlashInfer tests on $(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1)"
git init -q vllm && git -C vllm remote add origin https://github.com/vllm-project/vllm
git -C vllm fetch -q --depth 1 origin $VLLM_SHA && git -C vllm checkout -q FETCH_HEAD
echo "vllm $(git -C vllm log -1 --format='%h %cd')"

# Keep the notebook's torch: every other requirement is constrained to it.
python -c "import torch; print('torch==' + torch.__version__)" > constraints.txt
pip install -q -c constraints.txt -r vllm/requirements/common.txt "flashinfer-python==0.7.0.post1" pytest 2>&1 | grep -v -i "dependency resolver\|incompatible\|^$" | tail -3
python -c "import torch, flashinfer; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| flashinfer', flashinfer.__version__)"

# FlashInfer's JIT links with -lcuda; the image has libcuda.so.1 but no libcuda.so for the linker.
LIBCUDA=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 /{print $NF; exit}')
[ -z "$LIBCUDA" ] && LIBCUDA=$(find / -name 'libcuda.so.1' -not -path '*/proc/*' 2>/dev/null | head -1)
mkdir -p $W/libcuda && ln -sf "$LIBCUDA" $W/libcuda/libcuda.so
export LIBRARY_PATH=$W/libcuda${LIBRARY_PATH:+:$LIBRARY_PATH}

# vLLM from source: a version for platform detection, and an empty compiled-extension stand-in.
touch vllm/vllm/_C_stable_libtorch.py
printf '__version__ = version = "0.0.0+src"\n__version_tuple__ = version_tuple = (0, 0, 0)\n' > vllm/vllm/_version.py
mkdir -p meta/vllm-0.0.0+src.dist-info && printf 'Metadata-Version: 2.1\nName: vllm\nVersion: 0.0.0+src\n' > meta/vllm-0.0.0+src.dist-info/METADATA
# The root tests/conftest.py imports the whole engine (compiled build); these kernel tests need only
# the autouse fixture from tests/kernels/conftest.py, loaded here as a plugin with --noconftest.
printf 'import pytest\nimport torch\n\n\n@pytest.fixture(autouse=True)\ndef reset_default_torch_device():\n    yield\n    torch.set_default_device(None)\n' > meta/kernels_fixture.py
export PYTHONPATH=$W/vllm:$W/meta
python -c "
from vllm.platforms import current_platform as p
print('vllm platform', type(p).__name__, 'capability', p.get_device_capability(), 'has SM80:', p.has_device_capability(80))"

T=tests/kernels/attention/test_flashinfer.py
cd vllm
echo "== 1. as on main (bfloat16): the first decode test"
timeout 1800 python -m pytest -q -x --tb=line -p no:cacheprovider --noconftest -p kernels_fixture "$T" -k "test_flashinfer_decode_with_paged_kv and not num_heads2" 2>&1 \
  | grep -E "passed|failed|error|Error|^E " | head -8

echo "== 2. patched: float16 below SM80, the whole file as CI runs it (-k 'not num_heads2')"
python - <<'EOF'
p = "tests/kernels/attention/test_flashinfer.py"
s = open(p).read()
old = "DTYPES = [torch.bfloat16]\n"
new = ("DTYPES = (\n    [torch.bfloat16]\n"
       "    if not current_platform.is_cuda() or current_platform.has_device_capability(80)\n"
       "    else [torch.float16]\n)\n")
assert s.count(old) == 1, "test file changed upstream"
open(p, "w").write(s.replace(old, new))
print("patched DTYPES")
EOF
START=$(date +%s)
timeout 7200 python -m pytest -q -rfE --tb=line -p no:cacheprovider --noconftest -p kernels_fixture "$T" -k "not num_heads2" 2>&1 \
  | grep -E "^(PASSED|FAILED|ERROR)|passed|failed|error|^E " | tail -40
echo "minutes: $(( ($(date +%s) - START) / 60 ))"
echo "== done: copy from '== vLLM FlashInfer tests' to here and send it back"
