// src/kernels/cuda/exl3.cu - EXL3 GPU reconstruct (docs/EXL3.md).
//
// Mirrors ExLlamaV3's reconstruct_had: decode the trellis tiles, apply the 128-point Hadamard on both
// dimensions, and fold in the suh/svh factors.  The codebook is procedural (decode_3inst), computed
// per window with no lookup table, exactly like the reference kernels.  This first version materializes
// W to fp16; the fused decode-GEMV is the follow-up (see docs/EXL3.md, "Decode optimizations").
#include "strata/kernels/exl3.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <cublas_v2.h>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int HAD = 128;

__device__ __forceinline__ float dh2f(uint16_t h) {
    unsigned sign = (unsigned)(h >> 15) << 31, exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
    if (exp == 0) {
        if (!man) f = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --exp; } man &= 0x3FF; f = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (man << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float o; __builtin_memcpy(&o, &f, 4); return o;
}

__device__ __forceinline__ uint32_t lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t imm) {
    uint32_t r = 0;
    for (int i = 0; i < 8; ++i) {
        uint32_t t = ((imm >> i) & 1) ? 0xFFFFFFFFu : 0u;
        uint32_t ta = (i & 4) ? a : ~a, tb = (i & 2) ? b : ~b, tc = (i & 1) ? c : ~c;
        r |= t & ta & tb & tc;
    }
    return r;
}

// The window for stored position `pos`: the 16 bits ending at bit (pos+1)*bits, read with ExLlamaV3's
// funnel shift over the packed uint32 words (unpack_trellis / dq).  NOT a linear LSB read - the encoder
// packs per 16-value span, MSB-first, then byte-swaps.
__device__ __forceinline__ uint16_t read_window(const uint16_t* __restrict__ tw, int pos, int bits) {
    const uint32_t* p32 = (const uint32_t*) tw;
    const int nw = bits * 256 / 32;
    const int b0 = pos * bits + bits - 16 + 256 * bits;
    const int b2 = b0 + 16;
    const int i0r = b0 >> 5, i1r = (b2 - 1) >> 5;
    const int s1 = (i1r + 1) * 32 - b2;
    const uint32_t a = p32[i0r % nw], b = p32[i1r % nw];
    const unsigned long long m = ((unsigned long long) a << 32) | (unsigned long long) b;
    return (uint16_t)((m >> s1) & 0xFFFF);
}

// mul1 codebook, folded to pure ALU.  decode_window computes h2f(bytesum(w*C)+0x6400) with the software
// fp16->fp32 helper, but that intermediate always has exponent 25 (s in [0x6400,0x67FC]) so it is exactly
// the integer 1024+(s&0x3FF); the two h2f constants fold to plain floats.  This removes the fp16 helper
// and its denormal loop from the GEMV's innermost op.
__device__ __forceinline__ float decode_mul1(uint32_t w) {
    uint32_t x = w * 0x83DCD12Du;
    uint32_t s = (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24) + 0x6400u;
    return (float)(1024 + (int)(s & 0x3FFu)) * 0.00676727294921875f - 10.3828125f;
}

__device__ __forceinline__ uint16_t decode_window(int cb, uint16_t w) {    uint32_t x = w;
    if (cb == 2) {
        x *= 0x83DCD12Du;
        uint32_t s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF) + 0x6400u;
        float h = dh2f((uint16_t)(s & 0xFFFF));
        return __half_as_ushort(__float2half_rn(__fmaf_rn(h, dh2f(0x1EEEu), dh2f(0xC931u))));
    }
    x = (cb == 1) ? x * 0xCBAC1FEDu : x * 89226354u + 64248484u;
    x = lop3(x, 0x8FFF8FFFu, 0x3B603B60u, 0x6Au);
    return __half_as_ushort(__float2half_rn(dh2f((uint16_t)(x & 0xFFFF)) + dh2f((uint16_t)(x >> 16))));
}

// One block per (nj, ki) tile grid; 256 threads, thread t decodes stored position t and writes it to
// its row-major destination inside the tile (tensor_core_perm: group g = t/8, slot = t%8).
__global__ void decode_kernel(const uint16_t* __restrict__ trellis, int ki, int nj, int words, int bits,
                              int cb, uint16_t* __restrict__ what, int n) {
    int i = blockIdx.y, j = blockIdx.x, t = threadIdx.x; (void)ki;
    const uint16_t* tw = trellis + (long)(i * nj + j) * words;
    unsigned v = read_window(tw, t, bits);
    int g = t >> 3, s = t & 7;
    int r0 = (g % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9, c0 = g / 4, c1 = c0 + 8;
    int r = (s & 3) == 0 ? r0 : (s & 3) == 1 ? r1 : (s & 3) == 2 ? r2 : r3;
    int c = s < 4 ? c0 : c1;
    what[(long)(i * 16 + r) * n + j * 16 + c] = decode_window(cb, (uint16_t)v);
}

__global__ void widen_kernel(const uint16_t* __restrict__ s, float* __restrict__ d, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = dh2f(s[i]);
}

__global__ void had_rows_kernel(float* __restrict__ w, int k, int n) {
    (void)k;
    // The vocabulary/head output dimension can exceed CUDA's grid.y limit of 65535.
    int rb = blockIdx.y, c = blockIdx.x;
    float b[HAD];
    for (int i = 0; i < HAD; ++i) b[i] = w[(long)(rb * HAD + i) * n + c];
    for (int step = 1; step < HAD; step <<= 1)
        for (int i = 0; i < HAD; ++i)
            if ((i & step) == 0) { float a = b[i], d = b[i | step]; b[i] = a + d; b[i | step] = a - d; }
    const float inv = 0.08838834764831845f;
    for (int i = 0; i < HAD; ++i) w[(long)(rb * HAD + i) * n + c] = b[i] * inv;
}

__global__ void had_cols_kernel(float* __restrict__ w, int k, int n) {
    (void)k;
    int r = blockIdx.x, cblk = blockIdx.y;
    float b[HAD];
    long base = (long)r * n + cblk * HAD;
    for (int j = 0; j < HAD; ++j) b[j] = w[base + j];
    for (int step = 1; step < HAD; step <<= 1)
        for (int j = 0; j < HAD; ++j)
            if ((j & step) == 0) { float a = b[j], d = b[j | step]; b[j] = a + d; b[j | step] = a - d; }
    const float inv = 0.08838834764831845f;
    for (int j = 0; j < HAD; ++j) w[base + j] = b[j] * inv;
}

__global__ void scale_rows_kernel(float* __restrict__ w, int n, const uint16_t* __restrict__ suh) {
    int i = blockIdx.x;
    float s = dh2f(suh[i]);
    for (int c = threadIdx.x; c < n; c += blockDim.x) w[(long)i * n + c] *= s;
}

__global__ void scale_cols_kernel(float* __restrict__ w, int k, int n, const uint16_t* __restrict__ svh) {
    int c = blockIdx.x;
    float s = dh2f(svh[c]);
    for (int r = threadIdx.x; r < k; r += blockDim.x) w[(long)r * n + c] *= s;
}

__global__ void f32_to_f16_kernel(const float* __restrict__ in, uint16_t* __restrict__ out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half_as_ushort(__float2half_rn(in[i]));
}

}  // namespace

void exl3_reconstruct_weight(const uint16_t* trellis, int ki, int nj, int bits, int cb,
                             const uint16_t* suh, const uint16_t* svh, uint16_t* out, void* stream) {
    if (ki <= 0 || nj <= 0 || (ki * 16) % HAD || (nj * 16) % HAD) {
        std::fprintf(stderr, "exl3_reconstruct_weight: dims must be multiples of %d\n", HAD);
        std::abort();
    }
    cudaStream_t st = (cudaStream_t)stream;
    int k = ki * 16, n = nj * 16, words = 256 * bits / 16;
    long nelem = (long)k * n;
    uint16_t* d_what = nullptr;
    float* d_w = nullptr;
    if (cudaMalloc(&d_what, (size_t)nelem * 2) != cudaSuccess ||
        cudaMalloc(&d_w, (size_t)nelem * 4) != cudaSuccess) {
        std::fprintf(stderr, "exl3_reconstruct_weight: cudaMalloc failed\n");
        std::abort();
    }
    auto check_launch = [](const char* name) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "exl3_reconstruct_weight %s: %s\n", name, cudaGetErrorString(e));
            std::abort();
        }
    };
    decode_kernel<<<dim3(nj, ki), 256, 0, st>>>(trellis, ki, nj, words, bits, cb, d_what, n);
    check_launch("decode_kernel");
    widen_kernel<<<(int)((nelem + 255) / 256), 256, 0, st>>>(d_what, d_w, nelem);
    check_launch("widen_kernel");
    had_rows_kernel<<<dim3(n, k / HAD), 1, 0, st>>>(d_w, k, n);
    check_launch("had_rows_kernel");
    had_cols_kernel<<<dim3(k, n / HAD), 1, 0, st>>>(d_w, k, n);
    check_launch("had_cols_kernel");
    scale_rows_kernel<<<dim3(k), 128, 0, st>>>(d_w, n, suh);
    check_launch("scale_rows_kernel");
    scale_cols_kernel<<<dim3(n), 128, 0, st>>>(d_w, k, n, svh);
    check_launch("scale_cols_kernel");
    f32_to_f16_kernel<<<(int)((nelem + 255) / 256), 256, 0, st>>>(d_w, out, nelem);
    check_launch("f32_to_f16_kernel");
    (void)cudaFree(d_what);
    (void)cudaFree(d_w);
}

namespace {

__global__ void prescale_had_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ suh,
                                    float* __restrict__ out) {
    __shared__ float b[HAD];
    int t = threadIdx.x, idx = blockIdx.x * HAD + t;
    b[t] = dh2f(x[idx]) * dh2f(suh[idx]);
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    out[idx] = b[t] * 0.08838834764831845f;
}

// One block per (output 16-column tile, k-group); 128 threads = 16 columns x 8 k-subchunks.  Thread
// (c, sub) accumulates output column c over its k-slice, the 8 partials are summed in shared, and one
// atomicAdd per column folds the block's k-group into z.  This replaced 16-thread blocks that
// atomicAdd-ed per k-tile: 16-thread blocks cap at ~24 blocks/CU (≈19% occupancy) and every k-tile
// serialized on the same z, which is what pinned the GEMV at ~11 tok/s.
// **4 OUTPUT TILES PER BLOCK, MEASURED.**  The expert FFN is 48 layers x (gate, up, down) over 10 experts
// and it was the largest compute block of a saturated token (18.5 ms/token).  Sweeping the tiles per block on
// the real layer (`strata-exl3-moe`, layer 0, 10 experts): T=1 0.534 ms/layer, T=2 0.385 (the old value),
// T=3 0.321, **T=4 0.308**, T=5 0.319, T=6 0.327, T=7 0.342, T=8 0.325.  Sweeping the k-group count up from
// (ki+15)/16 only lost (x2 0.330, x4 0.543).  So 4, and the split stays as it is: -20% on the expert FFN,
// 3.7 ms/token, on every machine.
constexpr int GEMV_T = 4;

// The dense (single-matrix) GEMV: 256 threads (16 sub-groups of 16 output columns) and `GEMV_T` output
// tiles per block, like the batched kernel, but with the matrix's geometry as scalars.
//
// **THE WINDOW GEOMETRY IS HOISTED OUT OF THE k LOOP.**  The two words a 16-bit window touches and the shift
// between them depend only on the thread's 16 row positions (`tt[r]`, fixed for the block) and `bits`, not on
// the tile the loop is walking.  Computing them per weight costs 3 multiplies and two integer modulos, and
// with ~20 weights per thread (the old 8-way k split over 128 threads) that dominated the kernel: it ran at
// 40-115 GB/s against the ~700 GB/s the card has.  Packed into one int per row: word0 | word1 << 8 | shift << 16.
__global__ void __launch_bounds__(256, 6) gemv_kernel(const float* __restrict__ xh, const uint16_t* __restrict__ trellis,
                                                      int ki, int nj, int words, int bits, int cb,
                                                      const int* __restrict__ pinv, float* __restrict__ z) {
    int c = threadIdx.x & 15, sub = threadIdx.x >> 4;
    int ns = gridDim.y * 16;
    int cid = blockIdx.y * 16 + sub;
    int k0 = (ki * cid) / ns, k1 = (ki * (cid + 1)) / ns;
    int tt[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) tt[r] = pinv[r * 16 + c];
    const int nw = bits * 256 / 32;
    int wo[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int b0 = tt[r] * bits + bits - 16 + 256 * bits;
        const int b2 = b0 + 16;
        const int i0 = b0 >> 5, i1 = (b2 - 1) >> 5;
        wo[r] = (i0 % nw) | ((i1 % nw) << 8) | (((i1 + 1) * 32 - b2) << 16);
    }
    float acc[GEMV_T];
    #pragma unroll
    for (int t = 0; t < GEMV_T; ++t) acc[t] = 0;
    for (int i = k0; i < k1; ++i) {
        const uint32_t* twp[GEMV_T];
        #pragma unroll
        for (int t = 0; t < GEMV_T; ++t) {
            int j = blockIdx.x + t * gridDim.x;
            twp[t] = (j < nj) ? (const uint32_t*) (trellis + (long) (i * nj + j) * words) : nullptr;
        }
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const float xr = xh[i * 16 + r];
            const int g = wo[r];
            #pragma unroll
            for (int t = 0; t < GEMV_T; ++t) {
                if (twp[t] == nullptr) continue;
                const uint64_t m = ((uint64_t) twp[t][g & 255] << 32) | twp[t][(g >> 8) & 255];
                const uint16_t win = (uint16_t) (m >> (g >> 16));
                const float wv = cb == 2 ? decode_mul1(win)
                                         : __half2float(__ushort_as_half(decode_window(cb, win)));
                acc[t] += xr * wv;
            }
        }
    }
    __shared__ float red[GEMV_T][16][16];
    #pragma unroll
    for (int t = 0; t < GEMV_T; ++t) red[t][sub][c] = acc[t];
    __syncthreads();
    if (sub == 0) {
        #pragma unroll
        for (int t = 0; t < GEMV_T; ++t) {
            const int j = blockIdx.x + t * gridDim.x;
            if (j >= nj) break;
            float s = 0;
            #pragma unroll
            for (int k2 = 0; k2 < 16; ++k2) s += red[t][k2][c];
            atomicAdd(&z[j * 16 + c], s);
        }
    }
}

__global__ void had_postscale_kernel(const float* __restrict__ z, const uint16_t* __restrict__ svh,
                                     uint16_t* __restrict__ y) {
    __shared__ float b[HAD];
    int t = threadIdx.x, idx = blockIdx.x * HAD + t;
    b[t] = z[idx];
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    y[idx] = __half_as_ushort(__float2half_rn(b[t] * 0.08838834764831845f * dh2f(svh[idx])));
}

void build_pinv(int* pinv) {
    for (int t = 0; t < 256; ++t) {
        int g = t >> 3, s = t & 7;
        int r0 = (g % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9, c0 = g / 4, c1 = c0 + 8;
        int r = (s & 3) == 0 ? r0 : (s & 3) == 1 ? r1 : (s & 3) == 2 ? r2 : r3;
        int c = s < 4 ? c0 : c1;
        pinv[r * 16 + c] = t;
    }
}

// The GEMV body, with the three scratch buffers supplied so a caller can reuse them across launches.
void gemv_launch(const uint16_t* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
                 int ki, int nj, int bits, int cb, uint16_t* y, float* d_xh, float* d_z, const int* d_pinv,
                 cudaStream_t st) {
    int k = ki * 16, n = nj * 16, words = 256 * bits / 16;
    prescale_had_kernel<<<k / HAD, HAD, 0, st>>>(x, suh, d_xh);
    // A k split so the block count fills the machine: at least ~256 blocks, and never more than one row per
    // sub.  gy = 1 (each sub walks ki/16 rows) is the fast case for the big matrices; the small ones (the
    // shared expert's 40 output tiles) need the split or they run on 5k threads.
    int gy = 1;
    while (gy < 16 && ((nj + GEMV_T - 1) / GEMV_T) * gy < 256 && ki >= (gy + 1) * 16) ++gy;
    (void) cudaMemsetAsync(d_z, 0, (size_t)n * 4, st);
    gemv_kernel<<<dim3((nj + GEMV_T - 1) / GEMV_T, gy), 256, 0, st>>>(d_xh, trellis, ki, nj, words, bits, cb, d_pinv, d_z);
    had_postscale_kernel<<<n / HAD, HAD, 0, st>>>(d_z, svh, y);
}

}  // namespace

namespace {
// Persistent GEMV workspace, allocated before any graph capture (cudaMalloc is illegal inside a capture).
struct GemvWs { float* xh = nullptr; float* z = nullptr; int* pinv = nullptr; uint16_t* x16 = nullptr; uint16_t* y16 = nullptr;
                int cxh = 0, cz = 0, cx16 = 0, cy16 = 0; };
GemvWs g_ws;
}  // namespace

// out[o][i] = in[i][o]: a 32x32 tiled transpose, both sides coalesced.  The reconstruct hands back the weight
// in the original basis as [in][out] (the GEMV's `y[j] = sum_i x[i] W[i][j]`), while a ggml block quant - and
// `native_mmvq` - stores a row of `n_in` per output, so the loader transposes before it quantizes.
__global__ void transpose_f16_kernel(const uint16_t* __restrict__ in, uint16_t* __restrict__ out, int rows, int cols) {
    __shared__ uint16_t tile[32][33];
    const int x = blockIdx.x * 32 + threadIdx.x;      // column in `in` == row in `out`
    const int y = blockIdx.y * 32 + threadIdx.y;      // row in `in` == column in `out`
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[(size_t) y * cols + x];
    __syncthreads();
    const int ox = blockIdx.y * 32 + threadIdx.x;     // column in `out` == row in `in`
    const int oy = blockIdx.x * 32 + threadIdx.y;     // row in `out` == column in `in`
    if (ox < rows && oy < cols) out[(size_t) oy * rows + ox] = tile[threadIdx.x][threadIdx.y];
}

void exl3_transpose_f16(const uint16_t* in, uint16_t* out, int rows, int cols, void* stream) {
    dim3 block(32, 32);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);
    transpose_f16_kernel<<<grid, block, 0, (cudaStream_t) stream>>>(in, out, rows, cols);
}

void exl3_gemv_reserve(int max_k, int max_n) {
    if (max_k > g_ws.cxh) { (void) cudaFree(g_ws.xh); g_ws.xh = nullptr; if (cudaMalloc(&g_ws.xh, (size_t) max_k * 4) != cudaSuccess) std::abort(); g_ws.cxh = max_k; }
    if (max_n > g_ws.cz) { (void) cudaFree(g_ws.z); g_ws.z = nullptr; if (cudaMalloc(&g_ws.z, (size_t) max_n * 4) != cudaSuccess) std::abort(); g_ws.cz = max_n; }
    if (max_k > g_ws.cx16) { (void) cudaFree(g_ws.x16); g_ws.x16 = nullptr; if (cudaMalloc(&g_ws.x16, (size_t) max_k * 2) != cudaSuccess) std::abort(); g_ws.cx16 = max_k; }
    if (max_n > g_ws.cy16) { (void) cudaFree(g_ws.y16); g_ws.y16 = nullptr; if (cudaMalloc(&g_ws.y16, (size_t) max_n * 2) != cudaSuccess) std::abort(); g_ws.cy16 = max_n; }
    if (g_ws.pinv == nullptr) {
        if (cudaMalloc(&g_ws.pinv, 256 * 4) != cudaSuccess) std::abort();
        int p[256]; build_pinv(p); (void) cudaMemcpy(g_ws.pinv, p, 256 * 4, cudaMemcpyHostToDevice);
    }
}

void exl3_gemv(const uint16_t* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
               int ki, int nj, int bits, int cb, uint16_t* y, void* stream) {
    if (ki <= 0 || nj <= 0 || (ki * 16) % HAD || (nj * 16) % HAD) {
        std::fprintf(stderr, "exl3_gemv: dims must be multiples of %d\n", HAD);
        std::abort();
    }
    cudaStream_t st = (cudaStream_t)stream;
    exl3_gemv_reserve(ki * 16, nj * 16);
    gemv_launch(x, suh, svh, trellis, ki, nj, bits, cb, y, g_ws.xh, g_ws.z, g_ws.pinv, st);
}

namespace {

__global__ void f32_to_f16_out_kernel(const float* __restrict__ a, uint16_t* __restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __half_as_ushort(__float2half_rn(a[i]));
}

__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ h, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = dh2f(h[i]);
}

// ---- batched (all-experts) variants: blockIdx.z = expert ----
//
// The engine runs one layer's routed experts per token; doing them one at a time left the GPU idle - each
// expert's prescale->gemv->postscale->silu->down->accum chain is a serial dependency and same-stream
// kernels do not overlap, so throughput scaled exactly linearly with expert count (0.139 ms at k=1 ->
// 1.237 ms at k=10) while the GPU sat ~98% idle.  Making the expert the z dimension lets all experts run
// concurrently inside one kernel and cuts the launch count from 14/expert to ~14/layer.

__global__ void prescale_batched_kernel(const uint16_t* __restrict__ x, long x_stride,
                                        const Exl3Mat* __restrict__ mats, float* __restrict__ xh, int k,
                                        float* __restrict__ zero, long nz, const int* input_rows = nullptr) {
    __shared__ float b[HAD];
    int e = blockIdx.y, t = threadIdx.x, idx = blockIdx.x * HAD + t;
    // **THE ZEROING RIDES ALONG WITH THE PRESCALE.**  The GEMV accumulates with atomics, so its output must be
    // zeroed first, and that was a `cudaMemsetAsync` of its own: a small kernel costs ~14 us of launch and
    // execution, measured as part of the expert FFN's 0.083 ms/layer fixed cost (a quarter of the whole).
    // The prescale already runs immediately before the GEMV on the same stream, so the zeroing is one more
    // strided store here.
    if (zero != nullptr) {
        const long stride = (long) gridDim.x * gridDim.y * HAD;
        for (long i = (long) (blockIdx.y * gridDim.x + blockIdx.x) * HAD + t; i < nz; i += stride) zero[i] = 0.0f;
    }
    const long input_row = input_rows ? input_rows[e] : e;
    b[t] = dh2f(x[input_row * x_stride + idx]) * dh2f(mats[e].suh[idx]);
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    xh[(long)e * k + idx] = b[t] * 0.08838834764831845f;
}

template <int T, bool Partials = false>
__global__ void __launch_bounds__(256, 6) gemv_batched_kernel(const float* __restrict__ xh, const Exl3Mat* __restrict__ mats,
                                    const int* __restrict__ pinv, float* __restrict__ z) {
    int e = blockIdx.z;
    int c = threadIdx.x & 15, sub = threadIdx.x >> 4;
    int ki = mats[e].ki, nj = mats[e].nj, bits = mats[e].bits, cb = mats[e].cb;
    int words = 256 * bits / 16;
    int ns = gridDim.y * 16;
    int cid = blockIdx.y * 16 + sub;
    int k0 = (ki * cid) / ns, k1 = (ki * (cid + 1)) / ns;
    const uint16_t* trellis = mats[e].trellis;
    const float* xhe = xh + (long)e * (ki * 16);
    int tt[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) tt[r] = pinv[r * 16 + c];
    // the window geometry for this thread's 16 rows: fixed for the whole k loop (see gemv_kernel)
    const int nw = bits * 256 / 32;
    int wo[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int b0 = tt[r] * bits + bits - 16 + 256 * bits;
        const int b2 = b0 + 16;
        const int i0 = b0 >> 5, i1 = (b2 - 1) >> 5;
        wo[r] = (i0 % nw) | ((i1 % nw) << 8) | (((i1 + 1) * 32 - b2) << 16);
    }
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0;
    for (int i = k0; i < k1; ++i) {
        const uint32_t* twp[T];
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            int j = blockIdx.x + t * gridDim.x;
            twp[t] = (j < nj) ? (const uint32_t*) (trellis + (long)(i * nj + j) * words) : nullptr;
        }
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            float xr = xhe[i * 16 + r];
            const int g = wo[r];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                if (!twp[t]) continue;
                const uint64_t m = ((uint64_t) twp[t][g & 255] << 32) | twp[t][(g >> 8) & 255];
                const uint16_t win = (uint16_t) (m >> (g >> 16));
                float wv = cb == 2 ? decode_mul1(win) : __half2float(__ushort_as_half(decode_window(cb, win)));
                acc[t] += xr * wv;
            }
        }
    }
    __shared__ float red[T][16][16];
    #pragma unroll
    for (int t = 0; t < T; ++t) red[t][sub][c] = acc[t];
    __syncthreads();
    if (sub == 0) {
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            int j = blockIdx.x + t * gridDim.x;
            if (j >= nj) break;
            float s = 0;
            #pragma unroll
            for (int k2 = 0; k2 < 16; ++k2) s += red[t][k2][c];
            if constexpr (Partials)
                z[((long) e * gridDim.y + blockIdx.y) * (nj * 16) + j * 16 + c] = s;
            else atomicAdd(&z[(long)e * (nj * 16) + j * 16 + c], s);
        }
    }
}

__global__ void postscale_batched_kernel(const float* __restrict__ z, const Exl3Mat* __restrict__ mats,
                                         uint16_t* __restrict__ y, int n, int parts = 1) {
    __shared__ float b[HAD];
    int e = blockIdx.y, t = threadIdx.x, idx = blockIdx.x * HAD + t;
    float value = z[(long) e * parts * n + idx];
    for (int p = 1; p < parts; ++p) value += z[((long) e * parts + p) * n + idx];
    b[t] = value;
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if ((t & step) == 0) { float a = b[t], d = b[t | step]; b[t] = a + d; b[t | step] = a - d; }
        __syncthreads();
    }
    y[(long)e * n + idx] = __half_as_ushort(__float2half_rn(b[t] * 0.08838834764831845f * dh2f(mats[e].svh[idx])));
}

__global__ void silu_batched_kernel(const uint16_t* __restrict__ g, const uint16_t* __restrict__ u,
                                    uint16_t* __restrict__ h, long total) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float gv = __half2float(__ushort_as_half(g[i]));
    float s = gv / (1.0f + expf(-gv));
    h[i] = __half_as_ushort(__float2half_rn(s * __half2float(__ushort_as_half(u[i]))));
}

__global__ void accum_out_kernel(const uint16_t* __restrict__ d, const float* __restrict__ weights,
                                 int n_experts, int n, uint16_t* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0;
    for (int e = 0; e < n_experts; ++e) s += weights[e] * __half2float(__ushort_as_half(d[(long)e * n + i]));
    out[i] = __half_as_ushort(__float2half_rn(s));
}

}  // namespace


// Mutable MoE scratch belongs to an execution context, not to the process. Element capacities
// include the route count so a 3-expert hit partition followed by 10 misses cannot overrun scratch.
namespace {
template <typename T> struct MoeBuffer {
    T* p = nullptr;
    size_t capacity = 0;
    void reserve(size_t count) {
        if (count <= capacity) return;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::overflow_error("exl3 MoE workspace size overflow");
        T* next = nullptr;
        if (cudaMalloc((void**) &next, count * sizeof(T)) != cudaSuccess)
            throw std::runtime_error("exl3 MoE workspace allocation failed");
        (void) cudaFree(p);
        p = next;
        capacity = count;
    }
    ~MoeBuffer() { if (p) (void) cudaFree(p); }
};
}

struct Exl3MoeWorkspace::Impl {
    int device = -1;
    MoeBuffer<Exl3Mat> gate, up, down;
    MoeBuffer<uint16_t> g, u, h, d;
    MoeBuffer<float> xh, z, weights, partials;
    MoeBuffer<int> pinv;
    MoeBuffer<uint16_t> group_x, group_weight;
    MoeBuffer<float> group_y, group_reconstruct;
    cublasHandle_t blas = nullptr;
    Impl() { if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("exl3: no CUDA device"); }
    ~Impl() { if (blas) (void) cublasDestroy(blas); }
};

Exl3MoeWorkspace::Exl3MoeWorkspace() : impl_(new Impl) {}
Exl3MoeWorkspace::~Exl3MoeWorkspace() {
    int previous = -1;
    (void) cudaGetDevice(&previous);
    (void) cudaSetDevice(impl_->device);
    delete impl_;
    if (previous != -1) (void) cudaSetDevice(previous);
}
Exl3MoeWorkspace::Impl& Exl3MoeWorkspace::implementation() { return *impl_; }
void Exl3MoeWorkspace::reserve(int routes, int n_in, int n_ff, int n_out, bool indexed) {
    int device = -1;
    if (routes < 1 || n_in < 1 || n_ff < 1 || n_out < 1 ||
        cudaGetDevice(&device) != cudaSuccess || device != impl_->device)
        throw std::invalid_argument("exl3: workspace geometry/device mismatch");
    auto& w = *impl_;
    const size_t r = (size_t) routes;
    w.gate.reserve(r); w.up.reserve(r); w.down.reserve(r); w.weights.reserve(r);
    w.g.reserve(r * (size_t) n_ff); w.u.reserve(r * (size_t) n_ff); w.h.reserve(r * (size_t) n_ff);
    w.d.reserve(r * (size_t) n_out);
    w.xh.reserve(r * (size_t) std::max(n_in, n_ff));
    w.z.reserve(r * (size_t) std::max(n_ff, n_out));
    if (indexed) {
        const size_t gate_parts = ((size_t) n_in + 255) / 256;
        const size_t down_parts = ((size_t) n_ff + 255) / 256;
        w.partials.reserve(r * std::max(gate_parts * (size_t) n_ff, down_parts * (size_t) n_out));
    }
    if (!w.pinv.p) {
        w.pinv.reserve(256);
        int pinv[256]; build_pinv(pinv);
        if (cudaMemcpy(w.pinv.p, pinv, sizeof pinv, cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("exl3: permutation upload failed");
    }
}

namespace {
// Standalone compatibility callers get distinct scratch for each device, stream and host thread.
// The engine uses explicitly owned workspaces, whose lifetime matches its expert stream.
Exl3MoeWorkspace& fallback_moe_workspace(cudaStream_t stream) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("exl3: no CUDA device");
    using Key = std::pair<int, uintptr_t>;
    static thread_local std::map<Key, std::unique_ptr<Exl3MoeWorkspace>> workspaces;
    auto& item = workspaces[{device, reinterpret_cast<uintptr_t>(stream)}];
    if (!item) item = std::make_unique<Exl3MoeWorkspace>();
    return *item;
}

uint16_t* moe_run_experts(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, int n_experts,
                          const uint16_t* x, cudaStream_t st, Exl3MoeWorkspace* workspace,
                          long x_stride = 0, const int* input_rows = nullptr) {
    if (!gate || !up || !down || !x || n_experts <= 0 || n_experts > 65535)
        throw std::invalid_argument("exl3: invalid FFN arguments");
    for (int e = 0; e < n_experts; ++e) {
        for (const Exl3Mat* m : {&gate[e], &up[e], &down[e]})
            if (!m->trellis || !m->suh || !m->svh || m->ki <= 0 || m->ki > INT32_MAX / 16 ||
                m->nj <= 0 || m->nj > INT32_MAX / 16 || m->ki % 8 || m->nj % 8 ||
                m->bits < 1 || m->bits > 8 || m->cb < 0 || m->cb > 2)
                throw std::invalid_argument("exl3: invalid FFN matrix geometry");
        if (gate[e].ki != gate[0].ki || gate[e].nj != gate[0].nj || up[e].ki != gate[0].ki ||
            up[e].nj != gate[0].nj || down[e].ki != gate[0].nj || down[e].nj != down[0].nj)
            throw std::invalid_argument("exl3: mixed FFN dimensions");
    }
    const int k = gate[0].ki * 16;
    const int ff = gate[0].nj * 16;
    const int n = down[0].nj * 16;
    auto& owner = workspace ? *workspace : fallback_moe_workspace(st);
    owner.reserve(n_experts, k, ff, n, input_rows != nullptr);
    auto& w = owner.implementation();
    Exl3Mat* d_gate = w.gate.p; Exl3Mat* d_up = w.up.p; Exl3Mat* d_down = w.down.p;
    const size_t msz = (size_t) n_experts * sizeof(Exl3Mat);
    (void)cudaMemcpyAsync(d_gate, gate, msz, cudaMemcpyHostToDevice, st);
    (void)cudaMemcpyAsync(d_up, up, msz, cudaMemcpyHostToDevice, st);
    (void)cudaMemcpyAsync(d_down, down, msz, cudaMemcpyHostToDevice, st);
    uint16_t* d_g = w.g.p; uint16_t* d_u = w.u.p; uint16_t* d_h = w.h.p; uint16_t* d_d = w.d.p;
    float* d_xh = w.xh.p; float* d_z = w.z.p; int* d_pinv = w.pinv.p;
    int gy_g = (gate[0].ki + 15) / 16; if (gy_g < 1) gy_g = 1;
    int gy_d = (down[0].ki + 15) / 16; if (gy_d < 1) gy_d = 1;
    // Indexed windows already expose route-level parallelism. An optional smaller K split
    // amortizes trellis-window setup and reduces atomics; keep the established path by default.
    if (input_rows) {
        static const int cap = [] {
            const char* value = std::getenv("STRATA_EXL3_GEMV_SPLIT");
            return value ? std::max(1, std::min(16, std::atoi(value))) : std::numeric_limits<int>::max();
        }();
        gy_g = std::min(gy_g, cap); gy_d = std::min(gy_d, cap);
    }
    auto project = [&](Exl3Mat* mats, int outputs, int parts, uint16_t* out) {
        const dim3 grid((outputs / 16 + GEMV_T - 1) / GEMV_T, parts, n_experts);
        if (input_rows) {
            // Each K partition has its own output. The Hadamard pass sums them in a fixed order,
            // so a route's result does not depend on the other slots or CUDA block scheduling.
            gemv_batched_kernel<GEMV_T, true><<<grid, 256, 0, st>>>(d_xh, mats, d_pinv, w.partials.p);
            postscale_batched_kernel<<<dim3(outputs / HAD, n_experts), HAD, 0, st>>>(w.partials.p, mats, out, outputs, parts);
        } else {
            gemv_batched_kernel<GEMV_T><<<grid, 256, 0, st>>>(d_xh, mats, d_pinv, d_z);
            postscale_batched_kernel<<<dim3(outputs / HAD, n_experts), HAD, 0, st>>>(d_z, mats, out, outputs);
        }
    };
    prescale_batched_kernel<<<dim3(k / HAD, n_experts), HAD, 0, st>>>(x, x_stride, d_gate, d_xh, k, d_z, (long) n_experts * ff, input_rows);
    project(d_gate, ff, gy_g, d_g);
    prescale_batched_kernel<<<dim3(k / HAD, n_experts), HAD, 0, st>>>(x, x_stride, d_up, d_xh, k, d_z, (long) n_experts * ff, input_rows);
    project(d_up, ff, gy_g, d_u);
    { long tot = (long)n_experts * ff; silu_batched_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(d_g, d_u, d_h, tot); }
    prescale_batched_kernel<<<dim3(ff / HAD, n_experts), HAD, 0, st>>>(d_h, ff, d_down, d_xh, ff, d_z, (long) n_experts * n);
    project(d_down, n, gy_d, d_d);
    return d_d;
}

}  // namespace

void exl3_moe_ffn(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, const float* weights,
                  int n_experts, const uint16_t* x, uint16_t* out, void* stream, Exl3MoeWorkspace* workspace) {
    if (n_experts <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    const int n = down[0].nj * 16;
    uint16_t* d_d = moe_run_experts(gate, up, down, n_experts, x, st, workspace);
    auto& w = (workspace ? *workspace : fallback_moe_workspace(st)).implementation();
    float* d_weights = w.weights.p;
    (void)cudaMemcpyAsync(d_weights, weights, (size_t)n_experts * 4, cudaMemcpyHostToDevice, st);
    // one kernel, not two: the accumulator's f32 buffer is never read by anything else
    accum_out_kernel<<<(n + 255) / 256, 256, 0, st>>>(d_d, d_weights, n_experts, n, out);
}

void exl3_moe_rows_part(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, int k, const uint16_t* x,
                        uint16_t* out, void* stream, Exl3MoeWorkspace* workspace) {
    if (k <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    const int n = down[0].nj * 16;
    uint16_t* d_d = moe_run_experts(gate, up, down, k, x, st, workspace);
    (void) cudaMemcpyAsync(out, d_d, (size_t)k * n * 2, cudaMemcpyDeviceToDevice, st);
}

void exl3_moe_rows_indexed(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, int routes,
                           const uint16_t* x, const int* input_rows, uint16_t* out, void* stream,
                           Exl3MoeWorkspace& workspace) {
    if (routes == 0) return;
    if (routes < 0 || routes > 65535 || !gate || !up || !down || !x || !input_rows || !out)
        throw std::invalid_argument("exl3: invalid indexed FFN arguments");
    const auto st = (cudaStream_t) stream;
    uint16_t* rows = moe_run_experts(gate, up, down, routes, x, st, &workspace,
                                     (long) gate[0].ki * 16, input_rows);
    if (cudaMemcpyAsync(out, rows, (size_t) routes * down[0].nj * 16 * sizeof(uint16_t),
                        cudaMemcpyDeviceToDevice, st) != cudaSuccess)
        throw std::runtime_error("exl3: indexed FFN output copy failed");
}

namespace {
__global__ void decode_group_weight(Exl3Mat m, float* out) {
    const int i = blockIdx.y, j = blockIdx.x, t = threadIdx.x;
    const uint16_t* tw = m.trellis + (size_t) (i * m.nj + j) * 16 * m.bits;
    const int group = t >> 3, slot = t & 7;
    const int r0 = (group % 4) * 2, c0 = group / 4;
    const int r = r0 + ((slot & 3) >= 2 ? 8 : 0) + (slot & 1);
    const int c = c0 + (slot >= 4 ? 8 : 0);
    out[(size_t) (i * 16 + r) * (m.nj * 16) + j * 16 + c] =
        dh2f(decode_window(m.cb, (uint16_t) read_window(tw, t, m.bits)));
}
// Transform 32 columns together: contiguous loads/stores instead of a one-thread block's
// 128 strided values. No allocation or global synchronisation between expert groups.
__global__ void group_had_rows(float* w, int n) {
    __shared__ float tile[HAD][32];
    const int col = blockIdx.x * 32 + threadIdx.x, rb = blockIdx.y * HAD;
    for (int r = threadIdx.y; r < HAD; r += blockDim.y) tile[r][threadIdx.x] = w[(size_t) (rb + r) * n + col];
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        for (int r = threadIdx.y; r < HAD; r += blockDim.y)
            if (!(r & step)) {
                const float a = tile[r][threadIdx.x], b = tile[r | step][threadIdx.x];
                tile[r][threadIdx.x] = a + b; tile[r | step][threadIdx.x] = a - b;
            }
        __syncthreads();
    }
    for (int r = threadIdx.y; r < HAD; r += blockDim.y)
        w[(size_t) (rb + r) * n + col] = tile[r][threadIdx.x] * 0.08838834764831845f;
}
__global__ void group_had_cols_pack(const float* w, int n, const uint16_t* suh, const uint16_t* svh, uint16_t* out) {
    __shared__ float tile[HAD];
    const int r = blockIdx.x, col = blockIdx.y * HAD + threadIdx.x, t = threadIdx.x;
    tile[t] = w[(size_t) r * n + col];
    __syncthreads();
    for (int step = 1; step < HAD; step <<= 1) {
        if (!(t & step)) {
            const float a = tile[t], b = tile[t | step];
            tile[t] = a + b; tile[t | step] = a - b;
        }
        __syncthreads();
    }
    out[(size_t) r * n + col] = __half_as_ushort(__float2half_rn(
        ((tile[t] * 0.08838834764831845f) * dh2f(suh[r])) * dh2f(svh[col])));
}
__global__ void gather_group_rows(const uint16_t* x, const int* input_rows, uint16_t* out, int rows, int n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t) rows * n) out[i] = x[(int64_t) input_rows[i / n] * n + i % n];
}
__global__ void round_group_output(const float* in, float* out, int64_t count) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __half2float(__float2half_rn(in[i]));
}
}

void exl3_moe_group_gemm(const Exl3Mat& gate, const Exl3Mat& up, const Exl3Mat& down, int rows,
                         const uint16_t* x, const int* input_rows, float* out, void* stream,
                         Exl3MoeWorkspace& workspace) {
    for (const Exl3Mat* matrix : {&gate, &up, &down})
        if (!matrix->trellis || !matrix->suh || !matrix->svh || matrix->bits < 1 || matrix->bits > 8 ||
            matrix->cb < 0 || matrix->cb > 2 || matrix->ki <= 0 || matrix->ki > INT32_MAX / 16 ||
            matrix->nj <= 0 || matrix->nj > INT32_MAX / 16 || matrix->ki % 8 || matrix->nj % 8)
            throw std::invalid_argument("exl3: invalid grouped GEMM matrix");
    if (rows <= 0 || rows > 65535 || !x || !input_rows || !out || gate.ki != up.ki || gate.nj != up.nj ||
        down.ki != gate.nj || gate.ki <= 0 || gate.nj <= 0 || down.nj <= 0)
        throw std::invalid_argument("exl3: invalid grouped GEMM geometry");
    const int k = gate.ki * 16, ff = gate.nj * 16, n = down.nj * 16;
    workspace.reserve(rows, k, ff, n);
    auto& w = workspace.implementation();
    w.group_x.reserve((size_t) rows * k);
    w.group_weight.reserve(std::max((size_t) k * ff, (size_t) ff * n));
    w.group_reconstruct.reserve(std::max((size_t) k * ff, (size_t) ff * n));
    w.group_y.reserve((size_t) rows * std::max(ff, n));
    const auto st = (cudaStream_t) stream;
    if (!w.blas && cublasCreate(&w.blas) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("exl3: cannot create grouped GEMM handle");
    if (cublasSetStream(w.blas, st) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("exl3: cannot set grouped GEMM stream");
    gather_group_rows<<<(unsigned) (((int64_t) rows * k + 255) / 256), 256, 0, st>>>(x, input_rows, w.group_x.p, rows, k);
    auto project = [&](const Exl3Mat& m, const uint16_t* input) {
        const int in = m.ki * 16, outputs = m.nj * 16;
        decode_group_weight<<<dim3(m.nj, m.ki), 256, 0, st>>>(m, w.group_reconstruct.p);
        group_had_rows<<<dim3(outputs / 32, in / HAD), dim3(32, 4), 0, st>>>(w.group_reconstruct.p, outputs);
        group_had_cols_pack<<<dim3(in, outputs / HAD), HAD, 0, st>>>(
            w.group_reconstruct.p, outputs, m.suh, m.svh, w.group_weight.p);
        const float alpha = 1.0f, beta = 0.0f;
        // Row-major W[in,out] is column-major W[out,in]; X[rows,in] is X[in,rows].
        if (cublasGemmEx(w.blas, CUBLAS_OP_N, CUBLAS_OP_N, outputs, rows, in, &alpha,
                         w.group_weight.p, CUDA_R_16F, outputs, input, CUDA_R_16F, in, &beta,
                         w.group_y.p, CUDA_R_32F, outputs, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("exl3: grouped tensor-core GEMM failed");
    };
    project(gate, w.group_x.p);
    const int64_t hidden = (int64_t) rows * ff;
    f32_to_f16_out_kernel<<<(unsigned) ((hidden + 255) / 256), 256, 0, st>>>(w.group_y.p, w.g.p, (int) hidden);
    project(up, w.group_x.p);
    f32_to_f16_out_kernel<<<(unsigned) ((hidden + 255) / 256), 256, 0, st>>>(w.group_y.p, w.u.p, (int) hidden);
    silu_batched_kernel<<<(unsigned) ((hidden + 255) / 256), 256, 0, st>>>(w.g.p, w.u.p, w.h.p, hidden);
    project(down, w.h.p);
    round_group_output<<<(unsigned) (((int64_t) rows * n + 255) / 256), 256, 0, st>>>(w.group_y.p, out, (int64_t) rows * n);
}

void exl3_moe_rows(const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down, const int* ids,
                   int k, const uint16_t* x, uint16_t* out, void* stream, Exl3MoeWorkspace* workspace) {
    (void) ids;   // the caller orders gate/up/down to match ids, so row i is expert ids[i]
    exl3_moe_rows_part(gate, up, down, k, x, out, stream, workspace);
}

// Scatter two f16 row blocks into the token's k rows (the order `moe_combine` expects): `dst[row]` takes
// block `b`'s next row, `n` values each.  One block per row: k is 10 at the real geometry.
__global__ void rows_scatter_kernel(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b,
                                    const int* __restrict__ a_row, int na, const int* __restrict__ b_row, int nb,
                                    int n, uint16_t* __restrict__ out) {
    const int i = blockIdx.x;
    const bool second = i >= na;
    const int r = second ? i - na : i;
    const int row = second ? b_row[r] : a_row[r];
    const uint16_t* src = (second ? b : a) + (size_t) r * n;
    uint16_t* dst = out + (size_t) row * n;
    for (int j = threadIdx.x; j < n; j += blockDim.x) dst[j] = src[j];
}

void exl3_rows_scatter(const uint16_t* a, const int* a_row, int na, const uint16_t* b, const int* b_row, int nb,
                       int n, uint16_t* out, void* stream) {
    if (na + nb <= 0) return;
    rows_scatter_kernel<<<na + nb, 256, 0, (cudaStream_t) stream>>>(a, b, a_row, na, b_row, nb, n, out);
}

__global__ void embed_gather_kernel(const uint16_t* __restrict__ row, long n, int bf16, float* __restrict__ out) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint16_t h = row[i];
    if (bf16) { uint32_t f = (uint32_t) h << 16; float o; __builtin_memcpy(&o, &f, 4); out[i] = o; }
    else out[i] = __half2float(*(const __half*) &h);
}

__global__ void embed_gather_rows_kernel(const uint16_t* table, int64_t n, int64_t vocab, bool bf16,
                                        const int32_t* tokens, int rows, float* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) rows * n) return;
    const int32_t token = tokens[i / n];
    float value = 0;
    if (token >= 0 && token < vocab) {
        const uint16_t h = table[(int64_t) token * n + i % n];
        value = bf16 ? __uint_as_float((uint32_t) h << 16) : dh2f(h);
    }
    out[i] = value;
}

void exl3_embed_gather_rows(const uint16_t* table, int64_t n_embd, int64_t n_vocab, bool bf16,
                            const int32_t* tokens, int rows, float* out, void* stream) {
    if (!table || !tokens || !out || n_embd <= 0 || n_vocab <= 0 || rows <= 0)
        throw std::invalid_argument("exl3: invalid embedding gather");
    embed_gather_rows_kernel<<<(unsigned) (((int64_t) rows * n_embd + 255) / 256), 256, 0,
                               (cudaStream_t) stream>>>(table, n_embd, n_vocab, bf16, tokens, rows, out);
}


__global__ void silu_mul_f32_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = g[i];
    h[i] = (gv / (1.0f + expf(-gv))) * u[i];
}

__global__ void shared_scalar_gate_kernel(float* __restrict__ out, const float* __restrict__ x,
                                          const uint16_t* __restrict__ w_ginp, int n_embd) {
    // one block: sigmoid(w_ginp . x) over n_embd, applied to every out element
    __shared__ float red[256];
    const int t = threadIdx.x;
    float s = 0;
    for (int i = t; i < n_embd; i += blockDim.x) {
        const uint16_t h = w_ginp[i];
        uint32_t f = (uint32_t) h << 16; float w; __builtin_memcpy(&w, &f, 4);
        s += w * x[i];
    }
    red[t] = s; __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) { if (t < step) red[t] += red[t + step]; __syncthreads(); }
    if (t == 0) red[0] = 1.0f / (1.0f + expf(-red[0]));
    __syncthreads();
    const float gate = red[0];
    for (int i = t; i < n_embd; i += blockDim.x) out[i] *= gate;
}

static float* g_sh_g = nullptr; static float* g_sh_u = nullptr; static float* g_sh_h = nullptr;
static int g_sh_ff = 0;

void exl3_shared_reserve(int n_ff) {
    if (n_ff <= g_sh_ff) return;
    (void) cudaFree(g_sh_g); (void) cudaFree(g_sh_u); (void) cudaFree(g_sh_h);
    g_sh_g = g_sh_u = g_sh_h = nullptr;
    if (cudaMalloc((void**) &g_sh_g, (size_t) n_ff * 4) || cudaMalloc((void**) &g_sh_u, (size_t) n_ff * 4) ||
        cudaMalloc((void**) &g_sh_h, (size_t) n_ff * 4)) std::abort();
    g_sh_ff = n_ff;
}

void exl3_shared_expert(const float* x, const Exl3Mat* gate, const Exl3Mat* up, const Exl3Mat* down,
                        const uint16_t* w_ginp, float* out, int n_embd, int n_ff, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    exl3_shared_reserve(n_ff);
    float* const d_g = g_sh_g; float* const d_u = g_sh_u; float* const d_h = g_sh_h;
    exl3_gemv_f32(x, gate->suh, gate->svh, gate->trellis, gate->ki, gate->nj, gate->bits, gate->cb, d_g, stream);
    exl3_gemv_f32(x, up->suh, up->svh, up->trellis, up->ki, up->nj, up->bits, up->cb, d_u, stream);
    silu_mul_f32_kernel<<<(unsigned)((n_ff + 255) / 256), 256, 0, st>>>(d_g, d_u, d_h, n_ff);
    exl3_gemv_f32(d_h, down->suh, down->svh, down->trellis, down->ki, down->nj, down->bits, down->cb, out, stream);
    shared_scalar_gate_kernel<<<1, 256, 0, st>>>(out, x, w_ginp, n_embd);
}

void exl3_embed_gather(const uint16_t* row, int64_t n, bool bf16, float* out, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    embed_gather_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(row, n, bf16 ? 1 : 0, out);
}


void exl3_gemv_f32(const float* x, const uint16_t* suh, const uint16_t* svh, const uint16_t* trellis,
                   int ki, int nj, int bits, int cb, float* y, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    const int k = ki * 16, n = nj * 16;
    exl3_gemv_reserve(k, n);
    f32_to_f16_out_kernel<<<(k + 255) / 256, 256, 0, st>>>(x, g_ws.x16, k);
    exl3_gemv(g_ws.x16, suh, svh, trellis, ki, nj, bits, cb, g_ws.y16, stream);
    f16_to_f32_kernel<<<(n + 255) / 256, 256, 0, st>>>(g_ws.y16, y, n);
}

}  // namespace strata::kernels
