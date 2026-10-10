#!/bin/bash
# Follow-up to kaggle_vllm_turing_tests.sh: the 6 float16 prefill failures on a T4 (page size 32),
# each run alone, with the size of the mismatch, and the same cases at page size 16 for contrast.
#   !cd /tmp && rm -rf sy && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/switchyard sy && bash sy/upstream/kaggle_vllm_turing_diag.sh
# About 10-15 minutes (reuses nothing from the first run if the notebook was restarted).
set -o pipefail
VLLM_SHA=193922d6ff5f7e213eacadd3cb7dd96612b30ed3
W=/tmp/vllm-turing
echo "== vLLM FlashInfer T4 diagnosis on $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
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
P="-q --tb=short -p no:cacheprovider --noconftest -p kernels_fixture"
for id in "None-None-dtype0-32-128-num_heads0-seq_lens0" "None-30.0-dtype0-32-128-num_heads0-seq_lens0" \
          "None-30.0-dtype0-32-256-num_heads0-seq_lens0" "64-None-dtype0-32-128-num_heads0-seq_lens0" \
          "64-30.0-dtype0-32-128-num_heads0-seq_lens0" "64-30.0-dtype0-32-256-num_heads0-seq_lens0" \
          "None-None-dtype0-16-128-num_heads0-seq_lens0" "None-None-dtype0-32-256-num_heads0-seq_lens0"; do
  for run in 1 2; do
    r=$(python -m pytest $P "$T::test_flashinfer_prefill_with_paged_kv[$id]" 2>&1)
    verdict=$(echo "$r" | grep -E "^[0-9]+ (passed|failed)" | tail -1)
    detail=$(echo "$r" | grep -E "Mismatched elements|Greatest absolute|Greatest relative|Error:|error:" | head -3 | tr '\n' ' ')
    echo "$id run $run: $verdict $detail"
  done
done
echo "== done: copy from '== vLLM FlashInfer T4 diagnosis' to here and send it back"
