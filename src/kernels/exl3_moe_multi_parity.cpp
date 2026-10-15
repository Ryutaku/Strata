// CPU-oracle parity for route-count growth, indexed inputs and independent execution workspaces.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace strata::kernels;
constexpr int N = 128;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
uint16_t half(float f) { const __half h = __float2half_rn(f); uint16_t v; std::memcpy(&v, &h, 2); return v; }
float fp32(uint16_t v) { __half h; std::memcpy(&h, &v, 2); return __half2float(h); }
template <class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc((void**) &p, n * sizeof(T))); }
    ~Buffer() { (void) cudaFree(p); }
    Buffer(const Buffer&) = delete;
};
struct Matrix {
    int inputs, outputs;
    std::vector<uint16_t> trellis, suh, svh, weight;
    Buffer<uint16_t> tr, su, sv;
    Exl3Mat gpu;
    Matrix(int bits, int cb, std::mt19937& rng, int in = N, int out = N) : inputs(in), outputs(out),
        trellis((size_t) (in / 16) * (out / 16) * 16 * bits), suh(in), svh(out), weight((size_t) in*out),
        tr(trellis.size()), su(in), sv(out) {
        for (auto& v : trellis) v = (uint16_t) rng();
        for (auto& v : suh) v = half(((int) (rng()%1000) - 500) * 0.0005f);
        for (auto& v : svh) v = half(((int) (rng()%1000) - 500) * 0.0005f);
        check(cudaMemcpy(tr.p, trellis.data(), trellis.size()*2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(su.p, suh.data(), inputs*2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(sv.p, svh.data(), outputs*2, cudaMemcpyHostToDevice));
        gpu = {tr.p, su.p, sv.p, inputs/16, outputs/16, bits, cb};
        cpu::exl3_reconstruct_weight(trellis.data(), inputs/16, outputs/16, bits, (cpu::Exl3Codebook) cb,
                                     suh.data(), svh.data(), weight.data());
    }
    std::vector<uint16_t> project(const uint16_t* x) const {
        std::vector<uint16_t> y(outputs);
        for (int j=0; j<outputs; ++j) {
            double s=0;
            for (int i=0; i<inputs; ++i) s += (double) fp32(x[i]) * fp32(weight[(size_t) i*outputs+j]);
            y[(size_t) j] = half((float) s);
        }
        return y;
    }
};

bool real_geometry(int device) {
    check(cudaSetDevice(device));
    std::mt19937 rng(104729);
    constexpr int width = 2560, hidden = 640;
    Matrix gate(6,2,rng,width,hidden), up(6,2,rng,width,hidden), down(6,2,rng,hidden,width);
    std::vector<uint16_t> x(4*width), ref(4*width), baseline;
    for (auto& v : x) v=half(((int)(rng()%1000)-500)*0.001f);
    for (int t=0;t<4;++t) {
        auto g=gate.project(x.data()+t*width); const auto u=up.project(x.data()+t*width);
        for (int i=0;i<hidden;++i) {const float v=fp32(g[i]);g[i]=half(v/(1+std::exp(-v))*fp32(u[i]));}
        const auto y=down.project(g.data()); std::copy(y.begin(),y.end(),ref.begin()+t*width);
    }
    Buffer<uint16_t> dx(x.size());check(cudaMemcpy(dx.p,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    Exl3MoeWorkspace workspace;
    bool ok=true;
    for (int routes : {10,40,10}) {
        std::vector<Exl3Mat> g(routes,gate.gpu),u(routes,up.gpu),d(routes,down.gpu);
        std::vector<int> rows(routes);for(int r=0;r<routes;++r)rows[r]=r%4;
        Buffer<int> dr(routes);Buffer<uint16_t> dy((size_t) routes*width);
        check(cudaMemcpy(dr.p,rows.data(),rows.size()*sizeof(int),cudaMemcpyHostToDevice));
        exl3_moe_rows_indexed(g.data(),u.data(),d.data(),routes,dx.p,dr.p,dy.p,nullptr,workspace);
        std::vector<uint16_t> y((size_t) routes*width);
        check(cudaMemcpy(y.data(),dy.p,y.size()*2,cudaMemcpyDeviceToHost));
        double num=0,den=0;
        for(size_t i=0;i<y.size();++i) {
            const double expected=fp32(ref[(size_t) rows[i/width]*width+i%width]);
            const double delta=fp32(y[i])-expected;num+=delta*delta;den+=expected*expected;
        }
        const double rel=std::sqrt(num/(den+1e-30));
        const bool identical=baseline.empty() || std::equal(baseline.begin(),baseline.end(),y.begin());
        ok=ok && std::isfinite(rel) && rel<0.02 && identical;
        if(baseline.empty())baseline.assign(y.begin(),y.begin()+10*width);
        std::printf("device=%d geometry=2560/640 routes=%d relative_error=%.5g route_count_identical=%d %s\n",
                    device,routes,rel,identical,ok?"PASS":"FAIL");
    }
    return ok;
}

bool run_device(int device) {
    check(cudaSetDevice(device));
    std::mt19937 rng(7919);
    Matrix g0(3,2,rng), u0(6,2,rng), d0(8,2,rng);
    Matrix g1(4,1,rng), u1(5,0,rng), d1(2,2,rng);
    Matrix* gate[] = {&g0,&g1}; Matrix* up[] = {&u0,&u1}; Matrix* down[] = {&d0,&d1};
    std::vector<uint16_t> x(3*N);
    for (auto& v : x) v = half(((int) (rng()%1000) - 500) * 0.001f);
    Buffer<uint16_t> dx(x.size()); check(cudaMemcpy(dx.p,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    // Distinct streams run different input-row permutations. Separate workspaces must not overwrite metadata.
    auto task = [&](int seed) {
        check(cudaSetDevice(device));
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        Exl3MoeWorkspace ws;
        bool ok=true;
        for (int routes : {3,10,2,37,53}) {
            const bool gemm = routes == 37;
            std::vector<int> tokens((size_t) routes);
            std::vector<Exl3Mat> g((size_t) routes),u((size_t) routes),d((size_t) routes);
            std::vector<float> ref((size_t) routes*N);
            for (int r=0;r<routes;++r) {
                const int e=gemm ? seed%2 : (r+seed)%2; tokens[(size_t) r]=(r*2+seed)%3;
                g[(size_t) r]=gate[e]->gpu; u[(size_t) r]=up[e]->gpu; d[(size_t) r]=down[e]->gpu;
                auto gv=gate[e]->project(x.data()+(size_t) tokens[(size_t) r]*N);
                const auto uv=up[e]->project(x.data()+(size_t) tokens[(size_t) r]*N);
                for (int i=0;i<N;++i) { const float v=fp32(gv[(size_t) i]); gv[(size_t) i]=half(v/(1+std::exp(-v))*fp32(uv[(size_t) i])); }
                const auto y=down[e]->project(gv.data());
                for (int i=0;i<N;++i) ref[(size_t) r*N+i]=fp32(y[(size_t) i]);
            }
            Buffer<int> dt((size_t) routes); Buffer<uint16_t> dy((size_t) routes*N);
            Buffer<float> df((size_t) routes*N);
            check(cudaMemcpyAsync(dt.p,tokens.data(),tokens.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
            if (gemm) exl3_moe_group_gemm(g[0],u[0],d[0],routes,dx.p,dt.p,df.p,(void*) stream,ws);
            else exl3_moe_rows_indexed(g.data(),u.data(),d.data(),routes,dx.p,dt.p,dy.p,(void*) stream,ws);
            std::vector<uint16_t> y((size_t) routes*N);
            std::vector<float> yf(y.size());
            if (gemm) check(cudaMemcpyAsync(yf.data(),df.p,yf.size()*4,cudaMemcpyDeviceToHost,stream));
            else check(cudaMemcpyAsync(y.data(),dy.p,y.size()*2,cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            double num=0,den=0;
            for (size_t i=0;i<y.size();++i) { double delta=(gemm?yf[i]:fp32(y[i]))-ref[i]; num+=delta*delta; den+=(double) ref[i]*ref[i]; }
            const double rel=std::sqrt(num/(den+1e-30));
            // CPU reconstruct rounds each weight; folded GEMV rounds only projection outputs.
            if (rel>0.02 || !std::isfinite(rel)) ok=false;
            std::printf("device=%d stream=%d routes=%d relative_error=%.5g %s\n",device,seed,routes,rel,rel<=0.02?"PASS":"FAIL");
        }
        check(cudaStreamDestroy(stream));
        return ok;
    };
    bool a=false,b=false; std::exception_ptr ea,eb;
    std::thread t0([&]{try {a=task(0);} catch (...) {ea=std::current_exception();}});
    std::thread t1([&]{try {b=task(1);} catch (...) {eb=std::current_exception();}});
    t0.join(); t1.join(); if(ea)std::rethrow_exception(ea); if(eb)std::rethrow_exception(eb);
    return real_geometry(device)&&a&&b;
}
}
int main() {
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) { std::puts("SKIP: no CUDA device"); return 77; }
    try { bool ok=true; for(int d=0;d<count;++d) ok=run_device(d)&&ok; return ok?0:1; }
    catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
