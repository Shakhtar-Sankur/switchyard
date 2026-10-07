// switchyard's MoE kernels (fp16, tensor cores through WMMA; sm_70 and newer).
//
//   route        router logits -> softmax (fp32) -> top-k experts and weights, one warp per token
//   count/assign tokens sorted by expert, stable (by token, then choice) and without atomics,
//                so a run is deterministic
//   grouped_gemm every expert's slice of the sorted batch in one launch; the first GEMM reads
//                its rows straight from the unsorted activations (no gather copy) and fuses
//                SwiGLU, the second scales each row by its router weight
//   combine      each token's k expert outputs summed in fp32, in expert order
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be a contiguous CUDA tensor")

// ---------------------------------------------------------------- route
// One warp per token; each lane holds up to E/32 logits. k rounds of warp argmax (ties go to
// the lower expert index).
template <int PER_LANE>
__global__ void route_kernel(const __half* __restrict__ logits, int N, int E, int k, bool norm,
                             int* __restrict__ idx, __half* __restrict__ w) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (warp >= N) return;
  const __half* row = logits + (size_t)warp * E;
  float v[PER_LANE];
  float mx = -INFINITY;
  for (int j = 0; j < PER_LANE; ++j) {
    int e = lane + 32 * j;
    v[j] = e < E ? __half2float(row[e]) : -INFINITY;
    mx = fmaxf(mx, v[j]);
  }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  float sum = 0.f;
  for (int j = 0; j < PER_LANE; ++j) {
    int e = lane + 32 * j;
    v[j] = e < E ? expf(v[j] - mx) : 0.f;
    sum += v[j];
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
  float picked[8];
  int pick_e[8];
  float total = 0.f;
  for (int r = 0; r < k; ++r) {
    float best = -1.f;
    int be = 1 << 30;
    for (int j = 0; j < PER_LANE; ++j) {
      int e = lane + 32 * j;
      if (e < E && (v[j] > best || (v[j] == best && e < be))) { best = v[j]; be = e; }
    }
    for (int o = 16; o; o >>= 1) {
      float ob = __shfl_xor_sync(0xffffffff, best, o);
      int oe = __shfl_xor_sync(0xffffffff, be, o);
      if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; }
    }
    for (int j = 0; j < PER_LANE; ++j)
      if (lane + 32 * j == be) v[j] = -1.f;  // taken
    picked[r] = best / sum;
    pick_e[r] = be;
    total += picked[r];
  }
  if (lane == 0) {
    for (int r = 0; r < k; ++r) {
      idx[(size_t)warp * k + r] = pick_e[r];
      w[(size_t)warp * k + r] = __float2half(norm ? picked[r] / total : picked[r]);
    }
  }
}

std::vector<torch::Tensor> route(torch::Tensor logits, int64_t k, bool norm) {
  CHECK(logits);
  TORCH_CHECK(logits.scalar_type() == torch::kHalf, "route: fp16 logits");
  int N = logits.size(0), E = logits.size(1);
  TORCH_CHECK(E <= 128 && k <= 8 && k <= E, "route: at most 128 experts and top-8");
  auto idx = torch::empty({N, k}, logits.options().dtype(torch::kInt32));
  auto w = torch::empty({N, k}, logits.options());
  if (N == 0) return {idx, w};
  auto stream = at::cuda::getCurrentCUDAStream();
  int threads = 256, blocks = (N * 32 + threads - 1) / threads;
  auto* l = reinterpret_cast<const __half*>(logits.data_ptr<at::Half>());
  auto* wp = reinterpret_cast<__half*>(w.data_ptr<at::Half>());
  if (E <= 32) route_kernel<1><<<blocks, threads, 0, stream>>>(l, N, E, k, norm, idx.data_ptr<int>(), wp);
  else if (E <= 64) route_kernel<2><<<blocks, threads, 0, stream>>>(l, N, E, k, norm, idx.data_ptr<int>(), wp);
  else route_kernel<4><<<blocks, threads, 0, stream>>>(l, N, E, k, norm, idx.data_ptr<int>(), wp);
  return {idx, w};
}

// ---------------------------------------------------------------- sort by expert
// One block per expert walks the flattened [N*k] choices in order; a block-wide prefix sum
// gives each match its place, so rows of an expert stay in (token, choice) order.
constexpr int SORT_THREADS = 256;

__device__ int block_exclusive_scan(int flag, int* tmp, int& total) {
  int t = threadIdx.x;
  tmp[t] = flag;
  __syncthreads();
  for (int o = 1; o < SORT_THREADS; o <<= 1) {
    int add = t >= o ? tmp[t - o] : 0;
    __syncthreads();
    tmp[t] += add;
    __syncthreads();
  }
  total = tmp[SORT_THREADS - 1];
  int excl = tmp[t] - flag;
  __syncthreads();
  return excl;
}

__global__ void count_kernel(const int* __restrict__ idx, int M, int* __restrict__ counts) {
  __shared__ int tmp[SORT_THREADS];
  int e = blockIdx.x, n = 0;
  for (int base = 0; base < M; base += SORT_THREADS) {
    int i = base + threadIdx.x, total;
    block_exclusive_scan(i < M && idx[i] == e, tmp, total);
    n += total;
  }
  if (threadIdx.x == 0) counts[e] = n;
}

__global__ void assign_kernel(const int* __restrict__ idx, const __half* __restrict__ w, int M, int k,
                              const int* __restrict__ offsets, int* __restrict__ token, __half* __restrict__ wsorted,
                              int* __restrict__ pos_of) {
  __shared__ int tmp[SORT_THREADS];
  int e = blockIdx.x, at = offsets[e];
  for (int base = 0; base < M; base += SORT_THREADS) {
    int i = base + threadIdx.x, total;
    bool hit = i < M && idx[i] == e;
    int p = block_exclusive_scan(hit, tmp, total);
    if (hit) {
      token[at + p] = i / k;
      wsorted[at + p] = w[i];
      pos_of[i] = at + p;
    }
    at += total;
  }
}

// Returns counts [E], offsets [E+1], token [M], weight [M] (sorted), pos_of [M] (row of each choice).
std::vector<torch::Tensor> sort_by_expert(torch::Tensor idx, torch::Tensor w, int64_t E) {
  CHECK(idx); CHECK(w);
  int M = idx.numel(), k = idx.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto counts = torch::empty({E}, idx.options());
  count_kernel<<<E, SORT_THREADS, 0, stream>>>(idx.data_ptr<int>(), M, counts.data_ptr<int>());
  auto offsets = torch::zeros({E + 1}, idx.options());
  offsets.slice(0, 1).copy_(counts.cumsum(0));
  auto token = torch::empty({M}, idx.options());
  auto wsorted = torch::empty({M}, w.options());
  auto pos_of = torch::empty({M}, idx.options());
  assign_kernel<<<E, SORT_THREADS, 0, stream>>>(idx.data_ptr<int>(), reinterpret_cast<const __half*>(w.data_ptr<at::Half>()), M, k,
                                               offsets.data_ptr<int>(), token.data_ptr<int>(),
                                               reinterpret_cast<__half*>(wsorted.data_ptr<at::Half>()), pos_of.data_ptr<int>());
  return {counts, offsets, token, wsorted, pos_of};
}

// ---------------------------------------------------------------- grouped GEMM
// C[r, :] = A[row(r), :] @ B[e]^T for every sorted row r of expert e, with B[e] in nn.Linear
// layout [Nfull, K]. 64x64 output tiles, 4 warps each computing 32x32 with 16x16x16 WMMA.
// The grid's x walks every expert's row tiles back to back (tile_offsets); y walks columns.
// SWIGLU: B[e] = [gate; up] stacked, the tile computes gate and up columns n0.. together
// and writes silu(gate) * up. SCALE: each output row is multiplied by scale[r].
constexpr int BM = 64, BN = 64, BK = 32, PAD = 8;

template <bool SWIGLU, bool SCALE>
__global__ void __launch_bounds__(128) grouped_gemm_kernel(
    const __half* __restrict__ A, const int* __restrict__ a_rows, const __half* __restrict__ B,
    const int* __restrict__ offsets, const int* __restrict__ tile_offsets, int E,
    int Nout, int K, const __half* __restrict__ scale, __half* __restrict__ C) {
  extern __shared__ __align__(16) unsigned char smem[];
  __half* As = reinterpret_cast<__half*>(smem);                 // [BM][BK+PAD]
  __half* Bs = As + BM * (BK + PAD);                            // [BN][BK+PAD]
  __half* Bu = Bs + BN * (BK + PAD);                            // [BN][BK+PAD] (SWIGLU)
  // After the K loop the operand tiles are dead: the fp32 epilogue reuses the same memory.
  float* Cs = reinterpret_cast<float*>(smem);                   // [BM][BN]
  float* Cu = Cs + BM * BN;                                     // [BM][BN] (SWIGLU)

  // Which expert does this row tile belong to? (binary search over tile_offsets)
  int tile = blockIdx.x, lo = 0, hi = E;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (tile_offsets[mid] <= tile) lo = mid; else hi = mid;
  }
  int e = lo;
  int row0 = offsets[e] + (tile - tile_offsets[e]) * BM, row_end = offsets[e + 1];
  int n0 = blockIdx.y * BN;
  int Nfull = SWIGLU ? 2 * Nout : Nout;
  const __half* Be = B + (size_t)e * Nfull * K;

  int warp = threadIdx.x / 32, wm = (warp / 2) * 32, wn = (warp % 2) * 32;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2], accu[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      wmma::fill_fragment(acc[i][j], 0.f);
      if (SWIGLU) wmma::fill_fragment(accu[i][j], 0.f);
    }

  // Each thread moves two 16-byte chunks of each operand per K tile (64 rows x 32 halves).
  // The next tile's chunks are loaded into registers while the current tile is multiplied.
  uint4 ra[2], rb[2], ru[2];
  auto fetch = [&](int k0) {
    for (int q = 0; q < 2; ++q) {
      int c = threadIdx.x + 128 * q, r = c / (BK / 8), kk = (c % (BK / 8)) * 8;
      int row = row0 + r, n = n0 + r;
      ra[q] = rb[q] = ru[q] = make_uint4(0, 0, 0, 0);
      if (row < row_end) {
        int src = a_rows ? a_rows[row] : row;
        ra[q] = *reinterpret_cast<const uint4*>(A + (size_t)src * K + k0 + kk);
      }
      if (n < Nout) {
        rb[q] = *reinterpret_cast<const uint4*>(Be + (size_t)n * K + k0 + kk);
        if (SWIGLU) ru[q] = *reinterpret_cast<const uint4*>(Be + (size_t)(Nout + n) * K + k0 + kk);
      }
    }
  };
  fetch(0);
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int q = 0; q < 2; ++q) {
      int c = threadIdx.x + 128 * q, r = c / (BK / 8), kk = (c % (BK / 8)) * 8;
      *reinterpret_cast<uint4*>(As + r * (BK + PAD) + kk) = ra[q];
      *reinterpret_cast<uint4*>(Bs + r * (BK + PAD) + kk) = rb[q];
      if (SWIGLU) *reinterpret_cast<uint4*>(Bu + r * (BK + PAD) + kk) = ru[q];
    }
    __syncthreads();
    if (k0 + BK < K) fetch(k0 + BK);
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], As + (wm + 16 * i) * (BK + PAD) + kk, BK + PAD);
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], Bs + (wn + 16 * j) * (BK + PAD) + kk, BK + PAD);
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      if (SWIGLU) {
        for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], Bu + (wn + 16 * j) * (BK + PAD) + kk, BK + PAD);
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 2; ++j) wmma::mma_sync(accu[i][j], a[i], b[j], accu[i][j]);
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      wmma::store_matrix_sync(Cs + (wm + 16 * i) * BN + wn + 16 * j, acc[i][j], BN, wmma::mem_row_major);
      if (SWIGLU) wmma::store_matrix_sync(Cu + (wm + 16 * i) * BN + wn + 16 * j, accu[i][j], BN, wmma::mem_row_major);
    }
  __syncthreads();
  for (int c = threadIdx.x; c < BM * BN; c += 128) {
    int r = c / BN, n = c % BN, row = row0 + r, col = n0 + n;
    if (row >= row_end || col >= Nout) continue;
    float v = Cs[c];
    if (SWIGLU) {
      // silu in fp32, rounded to fp16 like the reference's two separate fp16 ops
      float g = __half2float(__float2half(v)), up = __half2float(__float2half(Cu[c]));
      v = __half2float(__float2half(g / (1.f + expf(-g)))) * up;
    }
    if (SCALE) v = __half2float(__float2half(v)) * __half2float(scale[row]);
    C[(size_t)row * Nout + col] = __float2half(v);
  }
}

// tile_offsets [E+1]: cumulative row tiles per expert (ceil(count / 64)).
torch::Tensor grouped_gemm(torch::Tensor A, c10::optional<torch::Tensor> a_rows, torch::Tensor B,
                           torch::Tensor offsets, torch::Tensor tile_offsets, int64_t total_tiles,
                           int64_t rows, bool swiglu, c10::optional<torch::Tensor> scale) {
  CHECK(A); CHECK(B); CHECK(offsets); CHECK(tile_offsets);
  TORCH_CHECK(A.scalar_type() == torch::kHalf && B.scalar_type() == torch::kHalf, "grouped_gemm: fp16");
  int E = B.size(0), K = B.size(2), Nfull = B.size(1);
  TORCH_CHECK(A.size(1) == K && K % BK == 0, "grouped_gemm: K must match and be a multiple of 32");
  int Nout = swiglu ? Nfull / 2 : Nfull;
  auto C = torch::empty({rows, Nout}, A.options());
  if (total_tiles == 0 || rows == 0) return C;
  dim3 grid(total_tiles, (Nout + BN - 1) / BN);
  size_t operands = (size_t)(BM + BN * (swiglu ? 2 : 1)) * (BK + PAD) * sizeof(__half);
  size_t epilogue = (size_t)BM * BN * sizeof(float) * (swiglu ? 2 : 1);
  size_t sh = operands > epilogue ? operands : epilogue;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* ar = a_rows ? a_rows->data_ptr<int>() : nullptr;
  auto* a = reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
  auto* b = reinterpret_cast<const __half*>(B.data_ptr<at::Half>());
  auto* c = reinterpret_cast<__half*>(C.data_ptr<at::Half>());
  const __half* s = scale ? reinterpret_cast<const __half*>(scale->data_ptr<at::Half>()) : nullptr;
  if (swiglu) {
    auto kern = grouped_gemm_kernel<true, false>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    kern<<<grid, 128, sh, stream>>>(a, ar, b, offsets.data_ptr<int>(), tile_offsets.data_ptr<int>(), E, Nout, K, nullptr, c);
  } else if (s) {
    auto kern = grouped_gemm_kernel<false, true>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    kern<<<grid, 128, sh, stream>>>(a, ar, b, offsets.data_ptr<int>(), tile_offsets.data_ptr<int>(), E, Nout, K, s, c);
  } else {
    auto kern = grouped_gemm_kernel<false, false>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    kern<<<grid, 128, sh, stream>>>(a, ar, b, offsets.data_ptr<int>(), tile_offsets.data_ptr<int>(), E, Nout, K, nullptr, c);
  }
  return C;
}


// ---------------------------------------------------------------- grouped GEMV (decode)
// When each expert has only a few rows, the layer is a stream over expert weights. One block
// per (expert, 32 output columns); each warp owns 4 columns and reads their weight rows with
// 16-byte loads, against up to GEMV_ROWS activation rows held in shared memory (more rows are
// processed in further passes). Same epilogues as the GEMM.
constexpr int GEMV_ROWS = 8, GEMV_COLS = 32;  // 4 columns per warp: more blocks in flight

template <bool SWIGLU, bool SCALE>
__global__ void __launch_bounds__(256) grouped_gemv_kernel(
    const __half* __restrict__ A, const int* __restrict__ a_rows, const __half* __restrict__ B,
    const int* __restrict__ offsets, int Nout, int K, const __half* __restrict__ scale, __half* __restrict__ C) {
  extern __shared__ __align__(16) unsigned char smem[];
  __half* xs = reinterpret_cast<__half*>(smem);  // [GEMV_ROWS][K]
  int e = blockIdx.x, start = offsets[e], end = offsets[e + 1];
  if (start == end) return;
  int Nfull = SWIGLU ? 2 * Nout : Nout;
  const __half* Be = B + (size_t)e * Nfull * K;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int r0 = start; r0 < end; r0 += GEMV_ROWS) {
    int nr = min(GEMV_ROWS, end - r0);
    for (int c = threadIdx.x; c < GEMV_ROWS * K / 8; c += blockDim.x) {
      int r = c / (K / 8), kk = (c % (K / 8)) * 8;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < nr) {
        int src = a_rows ? a_rows[r0 + r] : r0 + r;
        v = *reinterpret_cast<const uint4*>(A + (size_t)src * K + kk);
      }
      *reinterpret_cast<uint4*>(xs + r * K + kk) = v;
    }
    __syncthreads();
    for (int j = 0; j < GEMV_COLS / 8; ++j) {
      int col = blockIdx.y * GEMV_COLS + warp * (GEMV_COLS / 8) + j;
      if (col >= Nout) break;
      float acc[GEMV_ROWS], accu[GEMV_ROWS];
      for (int r = 0; r < GEMV_ROWS; ++r) acc[r] = accu[r] = 0.f;
      const __half* wrow = Be + (size_t)col * K;
      const __half* urow = Be + (size_t)(Nout + col) * K;
#pragma unroll 2
      for (int kk = lane * 8; kk < K; kk += 256) {
        uint4 wv = *reinterpret_cast<const uint4*>(wrow + kk);
        uint4 uv = SWIGLU ? *reinterpret_cast<const uint4*>(urow + kk) : make_uint4(0, 0, 0, 0);
        const __half2* w2 = reinterpret_cast<const __half2*>(&wv);
        const __half2* u2 = reinterpret_cast<const __half2*>(&uv);
#pragma unroll
        for (int r = 0; r < GEMV_ROWS; ++r) {
          if (r >= nr) continue;
          uint4 xv = *reinterpret_cast<const uint4*>(xs + r * K + kk);
          const __half2* x2 = reinterpret_cast<const __half2*>(&xv);
          for (int q = 0; q < 4; ++q) {
            float2 xf = __half22float2(x2[q]), wf = __half22float2(w2[q]);
            acc[r] += xf.x * wf.x + xf.y * wf.y;
            if (SWIGLU) {
              float2 uf = __half22float2(u2[q]);
              accu[r] += xf.x * uf.x + xf.y * uf.y;
            }
          }
        }
      }
#pragma unroll
      for (int r = 0; r < GEMV_ROWS; ++r) {
        if (r >= nr) continue;
        for (int o = 16; o; o >>= 1) {
          acc[r] += __shfl_xor_sync(0xffffffff, acc[r], o);
          if (SWIGLU) accu[r] += __shfl_xor_sync(0xffffffff, accu[r], o);
        }
      }
      if (lane < nr) {
        float v = 0.f, u = 0.f;
#pragma unroll
        for (int r = 0; r < GEMV_ROWS; ++r)
          if (r == lane) { v = acc[r]; u = accu[r]; }
        int row = r0 + lane;
        if (SWIGLU) {
          float g = __half2float(__float2half(v)), up = __half2float(__float2half(u));
          v = __half2float(__float2half(g / (1.f + expf(-g)))) * up;
        }
        if (SCALE) v = __half2float(__float2half(v)) * __half2float(scale[row]);
        C[(size_t)row * Nout + col] = __float2half(v);
      }
    }
    __syncthreads();
  }
}

torch::Tensor grouped_gemv(torch::Tensor A, c10::optional<torch::Tensor> a_rows, torch::Tensor B,
                           torch::Tensor offsets, int64_t rows, bool swiglu, c10::optional<torch::Tensor> scale) {
  CHECK(A); CHECK(B); CHECK(offsets);
  int E = B.size(0), K = B.size(2), Nfull = B.size(1);
  TORCH_CHECK(A.size(1) == K && K % 8 == 0, "grouped_gemv: K must match and be a multiple of 8");
  int Nout = swiglu ? Nfull / 2 : Nfull;
  auto C = torch::empty({rows, Nout}, A.options());
  if (rows == 0) return C;
  dim3 grid(E, (Nout + GEMV_COLS - 1) / GEMV_COLS);
  size_t sh = (size_t)GEMV_ROWS * K * sizeof(__half);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* ar = a_rows ? a_rows->data_ptr<int>() : nullptr;
  auto* a = reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
  auto* b = reinterpret_cast<const __half*>(B.data_ptr<at::Half>());
  auto* c = reinterpret_cast<__half*>(C.data_ptr<at::Half>());
  const __half* s = scale ? reinterpret_cast<const __half*>(scale->data_ptr<at::Half>()) : nullptr;
  if (swiglu) {
    auto kern = grouped_gemv_kernel<true, false>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    kern<<<grid, 256, sh, stream>>>(a, ar, b, offsets.data_ptr<int>(), Nout, K, nullptr, c);
  } else if (s) {
    auto kern = grouped_gemv_kernel<false, true>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    kern<<<grid, 256, sh, stream>>>(a, ar, b, offsets.data_ptr<int>(), Nout, K, s, c);
  } else {
    auto kern = grouped_gemv_kernel<false, false>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
    kern<<<grid, 256, sh, stream>>>(a, ar, b, offsets.data_ptr<int>(), Nout, K, nullptr, c);
  }
  return C;
}

// ---------------------------------------------------------------- combine
// out[t] = sum over t's k choices of y[pos_of[t*k + j]], in expert order, accumulated in fp32.
__global__ void combine_kernel(const __half* __restrict__ y, const int* __restrict__ pos_of, const int* __restrict__ idx,
                               int k, int H, __half* __restrict__ out) {
  int t = blockIdx.x;
  int order[8];
  for (int j = 0; j < k; ++j) order[j] = j;
  for (int a = 1; a < k; ++a)  // insertion sort of the k choices by expert id
    for (int b = a; b > 0 && idx[t * k + order[b]] < idx[t * k + order[b - 1]]; --b) {
      int x = order[b]; order[b] = order[b - 1]; order[b - 1] = x;
    }
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float s = 0.f;
    for (int j = 0; j < k; ++j) s += __half2float(y[(size_t)pos_of[t * k + order[j]] * H + h]);
    out[(size_t)t * H + h] = __float2half(s);
  }
}

torch::Tensor combine(torch::Tensor y, torch::Tensor pos_of, torch::Tensor idx, int64_t n_tokens) {
  CHECK(y); CHECK(pos_of); CHECK(idx);
  int H = y.size(1), k = idx.size(1);
  auto out = torch::empty({n_tokens, H}, y.options());
  if (n_tokens == 0) return out;
  combine_kernel<<<n_tokens, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(y.data_ptr<at::Half>()), pos_of.data_ptr<int>(), idx.data_ptr<int>(), k, H,
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()));
  return out;
}


// ---------------------------------------------------------------- expert parallelism helpers
// Rows of src (on this GPU) picked by index, written to dst, which may live on another GPU
// (peer access enabled): the dispatch is a kernel writing straight into the peer's memory.
__global__ void gather_rows_kernel(const __half* __restrict__ src, const int* __restrict__ index, int n, int H,
                                   __half* __restrict__ dst) {
  int row = blockIdx.x;
  if (row >= n) return;
  const uint4* s = reinterpret_cast<const uint4*>(src + (size_t)index[row] * H);
  uint4* d = reinterpret_cast<uint4*>(dst + (size_t)row * H);
  for (int i = threadIdx.x; i < H / 8; i += blockDim.x) d[i] = s[i];
}

void gather_rows(torch::Tensor src, torch::Tensor index, torch::Tensor dst) {
  CHECK(src); CHECK(index); CHECK(dst);
  int n = index.numel(), H = src.size(1);
  TORCH_CHECK(H % 8 == 0 && dst.size(0) == n && dst.size(1) == H, "gather_rows: shapes");
  if (n == 0) return;
  gather_rows_kernel<<<n, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(src.data_ptr<at::Half>()), index.data_ptr<int>(), n, H,
      reinterpret_cast<__half*>(dst.data_ptr<at::Half>()));
}

// A contiguous copy between any two devices on the current stream (DMA, peer to peer).
void copy_async(torch::Tensor dst, torch::Tensor src) {
  TORCH_CHECK(dst.is_contiguous() && src.is_contiguous() && dst.nbytes() == src.nbytes(), "copy_async: shapes");
  if (src.nbytes() == 0) return;
  C10_CUDA_CHECK(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), src.nbytes(), cudaMemcpyDefault,
                                 at::cuda::getCurrentCUDAStream()));
}

bool enable_peer_access(int64_t device, int64_t peer) {
  int can = 0;
  C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&can, device, peer));
  if (!can) return false;
  int prev;
  C10_CUDA_CHECK(cudaGetDevice(&prev));
  C10_CUDA_CHECK(cudaSetDevice(device));
  cudaError_t e = cudaDeviceEnablePeerAccess(peer, 0);
  if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
  else C10_CUDA_CHECK(e);
  C10_CUDA_CHECK(cudaSetDevice(prev));
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route", &route);
  m.def("sort_by_expert", &sort_by_expert);
  m.def("grouped_gemm", &grouped_gemm);
  m.def("grouped_gemv", &grouped_gemv);
  m.def("combine", &combine);
  m.def("gather_rows", &gather_rows);
  m.def("copy_async", &copy_async);
  m.def("enable_peer_access", &enable_peer_access);
}
