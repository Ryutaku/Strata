// Resident EXL3 window compute benchmark at the model's actual 2560/640 geometry.
// No weight uploads in the timed loop. This is not an end-to-end engine benchmark.
#include "strata/kernels/exl3.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

using namespace strata::kernels;
static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t size) { check(cudaMalloc((void**) &p, size * sizeof(T))); }
    ~Buffer() { (void) cudaFree(p); }
    Buffer(const Buffer&) = delete;
};
struct Matrix {
    Buffer<uint16_t> tr, su, sv;
    Exl3Mat gpu;
    Matrix(int in, int out, std::mt19937& rng) : tr((size_t) in*out*6/16), su(in), sv(out) {
        std::vector<uint16_t> bits((size_t) in*out*6/16), scales((size_t) std::max(in,out));
        for (auto& v : bits) v = (uint16_t) rng();
        for (auto& v : scales) v = __half_as_ushort(__float2half(0.025f));
        check(cudaMemcpy(tr.p,bits.data(),bits.size()*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(su.p,scales.data(),in*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(sv.p,scales.data(),out*2,cudaMemcpyHostToDevice));
        gpu = {tr.p,su.p,sv.p,in/16,out/16,6,2};
    }
};
int main(int argc, char** argv) {
    const int routes = argc > 1 ? std::atoi(argv[1]) : 40;
    const int device = argc > 2 ? std::atoi(argv[2]) : 0;
    if (routes < 1 || routes > 512) return 2;
    try {
        check(cudaSetDevice(device));
        std::mt19937 rng(7319);
        // Ten experts with shared routes across four tokens. Distinct matrices avoid synthetic cache reuse.
        std::vector<Matrix*> weights;
        struct Cleanup { std::vector<Matrix*>& v; ~Cleanup() { for (auto p : v) delete p; } } cleanup{weights};
        for (int e=0;e<10;++e) {
            weights.push_back(new Matrix(2560,640,rng)); weights.push_back(new Matrix(2560,640,rng));
            weights.push_back(new Matrix(640,2560,rng));
        }
        std::vector<Exl3Mat> g(routes),u(routes),d(routes);
        std::vector<int> rows(routes);
        for (int r=0;r<routes;++r) {
            g[r]=weights[3*(r%10)]->gpu; u[r]=weights[3*(r%10)+1]->gpu; d[r]=weights[3*(r%10)+2]->gpu;
            rows[r]=r%4;
        }
        Buffer<int> dr(routes); Buffer<uint16_t> dx(4*2560),dy((size_t) routes*2560);
        check(cudaMemcpy(dr.p,rows.data(),rows.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemset(dx.p,0,4*2560*2));
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        Exl3MoeWorkspace workspace; workspace.reserve(routes,2560,640,2560);
        auto run = [&] { exl3_moe_rows_indexed(g.data(),u.data(),d.data(),routes,dx.p,dr.p,dy.p,(void*) stream,workspace); };
        for (int i=0;i<10;++i) run();
        check(cudaStreamSynchronize(stream));
        cudaEvent_t start,end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        check(cudaEventRecord(start,stream));
        for (int i=0;i<200;++i) run();
        check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
        float ms=0; check(cudaEventElapsedTime(&ms,start,end));
        std::printf("CUDA%d routes=%d split=%s resident_FFN_ms_per_layer_window=%.4f\n",device,routes,
                    std::getenv("STRATA_EXL3_GEMV_SPLIT") ? std::getenv("STRATA_EXL3_GEMV_SPLIT") : "default",ms/200);
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end)); check(cudaStreamDestroy(stream));
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
