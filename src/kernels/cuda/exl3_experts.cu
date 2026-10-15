// src/kernels/cuda/exl3_experts.cu - one layer's routed EXL3 experts on the GPU (see the header).
#include "strata/kernels/exl3_experts.hpp"
#include "strata/artifact/exl3_model.hpp"
#include "strata/kernels/exl3_route_plan.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define STRATA_EXL3_PAUSE() _mm_pause()
#else
#define STRATA_EXL3_PAUSE() ((void) 0)
#endif

namespace strata::kernels {

struct Exl3ExpertStore::Impl {
    std::vector<Exl3Mat> gate, up, down;
    std::vector<void*> allocs;
    std::unique_ptr<Exl3MoeWorkspace> workspace;
    int device = -1;
    Impl() {
        if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("exl3: no CUDA device");
        workspace = std::make_unique<Exl3MoeWorkspace>();
    }
    ~Impl() {
        int previous = -1;
        (void) cudaGetDevice(&previous); (void) cudaSetDevice(device);
        for (void* p : allocs) (void) cudaFree(p);
        workspace.reset();
        if (previous >= 0) (void) cudaSetDevice(previous);
    }
};

namespace {

Exl3Mat upload(const Exl3Linear& L, std::vector<void*>& allocs, cudaStream_t st, size_t& bytes) {
    auto alloc = [&](size_t n, const void* src) -> void* {
        void* p = nullptr;
        if (cudaMalloc(&p, n) != cudaSuccess) { std::fprintf(stderr, "exl3 store: cudaMalloc failed\n"); std::abort(); }
        (void) st;
        if (cudaMemcpy(p, src, n, cudaMemcpyHostToDevice) != cudaSuccess) { std::fprintf(stderr, "exl3 store: memcpy failed\n"); std::abort(); }
        allocs.push_back(p);
        bytes += n;
        return p;
    };
    int words = 256 * L.bits / 16;
    size_t tr_bytes = (size_t) L.ki * L.nj * words * 2;
    Exl3Mat M;
    M.trellis = (const uint16_t*) alloc(tr_bytes, L.trellis);
    M.suh = (const uint16_t*) alloc((size_t) L.in() * 2, L.suh);
    M.svh = (const uint16_t*) alloc((size_t) L.out() * 2, L.svh);
    M.ki = L.ki; M.nj = L.nj; M.bits = L.bits; M.cb = L.cb;
    return M;
}

}  // namespace

Exl3ExpertStore::Exl3ExpertStore(const std::string& model_dir, int layer, int n_experts, void* stream) {
    if (layer < 0 || layer >= 48 || n_experts <= 0 || n_experts > 512)
        throw std::invalid_argument("exl3: invalid expert store geometry");
    auto pending = std::make_unique<Impl>();
    auto* state = pending.get();
    n_experts_ = n_experts;
    cudaStream_t st = (cudaStream_t) stream;
    Exl3Model m(model_dir);
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.";
    for (int e = 0; e < n_experts; ++e) {
        const std::string b = prefix + std::to_string(e);
        state->gate.push_back(upload(m.linear(b + ".gate_proj"), state->allocs, st, bytes_));
        state->up.push_back(upload(m.linear(b + ".up_proj"), state->allocs, st, bytes_));
        state->down.push_back(upload(m.linear(b + ".down_proj"), state->allocs, st, bytes_));
    }
    state->workspace->reserve(n_experts, state->gate[0].ki * 16, state->gate[0].nj * 16,
                              state->down[0].nj * 16);
    impl_ = pending.release();
}

Exl3ExpertStore::~Exl3ExpertStore() { delete impl_; }

void Exl3ExpertStore::run(const uint16_t* x, const int* ids, const float* weights, int k, uint16_t* out,
                          void* stream) const {
    if (k == 0) return;
    int device = -1;
    if (k < 0 || k > n_experts_ || !x || !ids || !weights || !out ||
        cudaGetDevice(&device) != cudaSuccess || device != impl_->device)
        throw std::invalid_argument("exl3: expert store arguments/device mismatch");
    std::vector<Exl3Mat> g(k), u(k), d(k);
    for (int i = 0; i < k; ++i) {
        int e = ids[i];
        if (e < 0 || e >= n_experts_) throw std::invalid_argument("exl3: expert index out of range");
        g[i] = impl_->gate[e];
        u[i] = impl_->up[e];
        d[i] = impl_->down[e];
    }
    exl3_moe_ffn(g.data(), u.data(), d.data(), weights, k, x, out, stream, impl_->workspace.get());
}


// ================================ streaming experts ================================

namespace {

template <typename T> struct StreamBuffer {
    T* p = nullptr;
    size_t capacity = 0;
    void reserve(size_t count) {
        if (count <= capacity) return;
        if (count > SIZE_MAX / sizeof(T)) throw std::overflow_error("exl3: stream buffer overflow");
        T* next = nullptr;
        if (cudaMalloc((void**) &next, count * sizeof(T)) != cudaSuccess)
            throw std::runtime_error("exl3: stream buffer allocation failed");
        (void) cudaFree(p); p = next; capacity = count;
    }
    ~StreamBuffer() { if (p) (void) cudaFree(p); }
};

struct MultiBuffers {
    StreamBuffer<float> input, output;
    StreamBuffer<uint16_t> x, rows;
    StreamBuffer<int> input_rows, destinations;
    float* host_output = nullptr;
    size_t host_capacity = 0;
    void reserve(int tokens, int routes, int n, bool host_input = true) {
        if (host_input) input.reserve((size_t) tokens * n);
        x.reserve((size_t) tokens * n);
        output.reserve((size_t) routes * n); rows.reserve((size_t) routes * n);
        input_rows.reserve((size_t) routes);
        destinations.reserve((size_t) routes);
        const size_t count = (size_t) routes * n;
        if (count > host_capacity) {
            float* next = nullptr;
            if (cudaMallocHost((void**) &next, count * sizeof(float)) != cudaSuccess)
                throw std::runtime_error("exl3: pinned route output allocation failed");
            if (host_output) (void) cudaFreeHost(host_output);
            host_output = next; host_capacity = count;
        }
    }
    ~MultiBuffers() { if (host_output) (void) cudaFreeHost(host_output); }
};

__global__ void f16_to_f32_rows_kernel(const uint16_t* __restrict__ h, float* __restrict__ y, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(*(const __half*) &h[i]);
}

__global__ void f32_to_f16_rows_kernel(const float* __restrict__ x, uint16_t* __restrict__ h, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = __half_as_ushort(__float2half_rn(x[i]));
}

__global__ void scatter_route_output(const float* in, const int* destinations, float* out, int rows, int n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t) rows * n) out[(int64_t) destinations[i / n] * n + i % n] = in[i];
}

}  // namespace

struct Exl3ExpertStream::Impl {
    std::string dir;
    std::unique_ptr<Exl3Model> model;
    int device = -1;
    cudaStream_t compute_stream = nullptr;
    std::unique_ptr<Exl3MoeWorkspace> workspace;
    std::unique_ptr<MultiBuffers> multi;
    Exl3RoutePlan route_plan;
    std::mutex run_mu;
    int n_embd = 2560;
    int n_layers = 48;
    int layer_lo = 0, layer_hi = 48;
    int max_k = 64;
    int cache = 0;                     // slots per layer; 0 = the per-token scratch
    uint16_t* d_x = nullptr;
    float* d_input = nullptr;
    uint16_t* d_rows = nullptr;
    float* d_out = nullptr;

    struct Slot { uint16_t* tr = nullptr; uint16_t* suh = nullptr; uint16_t* svh = nullptr;
                  size_t trb = 0, suhb = 0, svhb = 0; };
    struct Blocks { std::vector<Slot> g, u, d; std::vector<Exl3Mat> gm, um, dm; };

    Blocks scratch;                    // cache == 0: `max_k` slots, every use overwrites them
    struct Layer : Blocks {
        std::vector<void*> allocs;         // one block per (projection, plane), `cache` slots in it
        std::unordered_map<int, int> at;   // expert -> slot
        std::vector<uint32_t> tick;        // per slot: the last use, for the LRU
        int filled = 0;                    // slots handed out so far; the map's size does not track it after an eviction
        uint32_t clock = 0;
    };
    std::vector<Layer> layers;         // cache > 0: `cache` slots per layer, resident across tokens

    // ---- the host-side cache (see the header).  Per layer, like the VRAM one: the routing is layer-specific,
    // and a slot holds that layer's nine pieces at fixed offsets, so a hit is a memcpy into the staging buffer.
    std::vector<uint8_t> ram;                     // pageable: pinning this much needs a memlock limit
    std::vector<size_t> ram_base;                 // per layer: where its slots start in `ram`
    std::vector<size_t> ram_slot_bytes;           // per layer: one expert's nine pieces
    std::vector<std::array<size_t, 9>> ram_off;   // per layer: the nine piece offsets inside a slot
    int ram_slots = 0;                            // slots per layer
    struct RamLayer {
        std::unordered_map<int, int> at;          // expert -> slot
        std::vector<uint32_t> tick;               // per slot: the last use, for the LRU
        std::vector<int> key;                     // slot -> expert
        int filled = 0;
        uint32_t clock = 0;
    };
    std::vector<RamLayer> ram_layers;
    std::vector<uint8_t> ram_pinned;
    std::mutex ram_mu;
    long long ram_hits = 0, ram_misses = 0;
    int cur_layer = 0;                            // the layer `run` is on (the queue is host-side, single-threaded)

    // STRATA_EXL3_TIME=1: per-token upload/ffn/d2h milliseconds and the cache hit rate
    long long hits = 0, misses = 0;
    double up_ms = 0, ffn_ms = 0, d2h_ms = 0, call_ms = 0;
    uint64_t upload_bytes = 0;
    cudaEvent_t profile[4] = {nullptr, nullptr, nullptr, nullptr};

    // ---- the miss reads
    //
    // **A MISS IS READ WITH `pread`, NOT OUT OF THE MMAP.**  Copying a tensor out of the mmap faults every
    // 4 KiB page on the way in: measured on 60 real expert tensors, 0.3 GB/s through the mmap against
    // 10.4 GB/s with `preadv` into a reused buffer, both from the page cache.  The read is the whole of the
    // doorbell: the token graph spins while the host fetches the expert.
    //
    // The destination is pinned staging, one buffer per worker, so the kernel writes straight into memory
    // the H2D can DMA from: no pageable staging copy, and the buffer's own pages never fault again.
    struct Job { uint16_t* dst; const uint16_t* src; int fd; size_t off; size_t bytes;
                 uint8_t* ram_dst = nullptr;       // a miss: fill this slot as a side effect of the read
                 const uint8_t* ram_src = nullptr; // a hit: copy this out of RAM through the pinned staging
               };
    std::vector<Job> jobs;
    std::vector<std::thread> workers;
    // **TWO BUFFERS PER WORKER, WITH AN EVENT ON EACH.**  The DMA out of a staging buffer is asynchronous,
    // so the next pread into the same buffer would overwrite data the DMA has not read yet (that is how the
    // first version produced NaNs).  Two buffers let a worker read into one while the other drains, and the
    // event orders the reuse.
    struct Stage { uint8_t* buf[2] = {nullptr, nullptr}; cudaEvent_t done[2] = {nullptr, nullptr}; int turn = 0; };
    std::vector<Stage> stage;          // one per worker (one for the sequential fallback)
    size_t stage_bytes = 0;
    std::map<std::string, int> fds;    // shard path -> fd, opened once
    std::mutex api_mu;                 // concurrent enqueues to one stream are not safe on the null stream
    // the resident/miss split: two output blocks (the FFN of each part writes its own) and the row maps
    uint16_t* d_hit = nullptr;
    uint16_t* d_miss = nullptr;
    int* d_hit_rows = nullptr;
    int* d_miss_rows = nullptr;
    std::atomic<size_t> job_at{0};
    std::atomic<uint64_t> work_done{0};
    std::atomic<int> stage_error{0};
    std::mutex work_mu;
    std::condition_variable work_cv;
    uint64_t work_gen = 0;             // guarded by work_mu
    bool work_quit = false;            // guarded by work_mu
    int n_workers = 0;
    cudaStream_t work_stream = nullptr;   // the staging copies' own stream (see the ctor)
    cudaEvent_t stage_done = nullptr;     // recorded on it, waited on the FFN's stream

    bool staged(cudaError_t e) {
        if (e != cudaSuccess) stage_error.store((int) e, std::memory_order_relaxed);
        return e == cudaSuccess;
    }

    void stage_one(const Job& j, Stage& stg) {
        if (j.bytes == 0) return;
        const uint16_t* src = j.src;
        if (j.ram_src != nullptr && stg.buf[0] != nullptr) {
            // **A HIT GOES THROUGH THE PINNED BUFFER TOO.**  `cudaMemcpyAsync` from pageable memory is
            // synchronous: it would stall each worker for the length of the copy, which is exactly what the
            // double buffer and the events exist to avoid.
            const int b = stg.turn;
            stg.turn ^= 1;
            if (stg.done[b] != nullptr && !staged(cudaEventSynchronize(stg.done[b]))) return;
            std::memcpy(stg.buf[b], j.ram_src, j.bytes);
            const std::lock_guard<std::mutex> lk(api_mu);
            staged(cudaMemcpyAsync(j.dst, stg.buf[b], j.bytes, cudaMemcpyHostToDevice, work_stream));
            if (stg.done[b] != nullptr) staged(cudaEventRecord(stg.done[b], work_stream));
            return;
        }
#if !defined(_WIN32)
        if (j.fd >= 0 && stg.buf[0] != nullptr) {
            const int b = stg.turn;
            stg.turn ^= 1;
            if (stg.done[b] != nullptr && !staged(cudaEventSynchronize(stg.done[b]))) return;
            const ssize_t got = ::pread(j.fd, stg.buf[b], j.bytes, (off_t) j.off);
            if (got == (ssize_t) j.bytes) {
                src = (const uint16_t*) stg.buf[b];
                // the host cache is filled as a side effect of the read that had to happen anyway; the copy is
                // synchronous, so it is complete before this buffer is read into again (the double buffer's
                // event only guards the DMA)
                if (j.ram_dst != nullptr) {
                    static std::atomic<int> fdbg{0};
                    if (std::getenv("STRATA_EXL3_RAM_DEBUG") != nullptr && fdbg.fetch_add(1) < 10)
                        std::fprintf(stderr, "ram fill: %zu bytes\n", j.bytes);
                    std::memcpy(j.ram_dst, stg.buf[b], j.bytes);
                }
                {
                    // **ONE ENQUEUE AT A TIME.**  The workers read in parallel, but two threads enqueueing
                    // `cudaMemcpyAsync` on the same (legacy default) stream raced: the same run produced
                    // different tokens with 4 workers and identical ones with 1.  The copies are asynchronous,
                    // so serialising only the call costs nothing.
                    const std::lock_guard<std::mutex> lk(api_mu);
                    staged(cudaMemcpyAsync(j.dst, src, j.bytes, cudaMemcpyHostToDevice, work_stream));
                    if (stg.done[b] != nullptr) staged(cudaEventRecord(stg.done[b], work_stream));
                }
                return;
            }
            std::fprintf(stderr, "strata generate: EXL3 expert read: pread %zu B at %zu: %zd\n", j.bytes, j.off, got);
        }
#endif
        {
            if (j.ram_dst) std::memcpy(j.ram_dst, src, j.bytes);
            const std::lock_guard<std::mutex> lk(api_mu);
            staged(cudaMemcpyAsync(j.dst, src, j.bytes, cudaMemcpyHostToDevice, work_stream));
        }
    }

    // Queue one expert's nine pieces: the FILE location (for the pread), the mmap pointer (the fallback), and
    // the host cache: on a hit the pieces come out of RAM, on a miss the read fills the slot as a side effect.
    void queue_expert(int expert, const Slot* gs, const Slot* us, const Slot* ds,
                      const Exl3Linear& Lg, const Exl3Linear& Lu, const Exl3Linear& Ld) {
        const std::string b = "model.language_model.layers." + std::to_string(cur_layer) + ".mlp.experts." +
                              std::to_string(expert) + ".";
        const uint8_t* ram_src = nullptr;
        uint8_t* ram_dst = nullptr;
        const Slot* slots[] = {gs, us, ds};
        const Exl3Linear* linears[] = {&Lg, &Lu, &Ld};
        std::array<size_t, 9> sizes{};
        bool ram_layout = true;
        size_t plane_at = 0;
        for (int i = 0; i < 3; ++i) {
            const auto& L = *linears[i];
            sizes[(size_t) i * 3] = (size_t) L.ki * L.nj * 32 * L.bits;
            sizes[(size_t) i * 3 + 1] = (size_t) L.in() * 2;
            sizes[(size_t) i * 3 + 2] = (size_t) L.out() * 2;
            if (sizes[(size_t) i * 3] > slots[i]->trb || sizes[(size_t) i * 3 + 1] > slots[i]->suhb ||
                sizes[(size_t) i * 3 + 2] > slots[i]->svhb)
                throw std::runtime_error("exl3: expert exceeds allocated weight slot");
            for (int p = 0; p < 3; ++p) {
                const size_t piece = (size_t) i * 3 + p;
                if (ram_off[(size_t) cur_layer][piece] != plane_at) ram_layout = false;
                plane_at += sizes[piece];
            }
        }
        ram_layout = ram_layout && plane_at == ram_slot_bytes[(size_t) cur_layer];
        if (ram_slots > 0 && ram_layout) {
            const std::lock_guard<std::mutex> lk(ram_mu);
            RamLayer& rl = ram_layers[(size_t) cur_layer];
            const size_t base = ram_base[(size_t) cur_layer];
            const size_t slot_bytes = ram_slot_bytes[(size_t) cur_layer];
            auto it = rl.at.find(expert);
            int s = -1;
            if (it != rl.at.end()) {
                s = it->second;
                ram_src = ram.data() + base + (size_t) s * slot_bytes;
                ++ram_hits;
            } else {
                ++ram_misses;
                if (rl.filled < ram_slots) {
                    s = rl.filled++;
                } else {
                    for (int i = 0; i < ram_slots; ++i)
                        if (!ram_pinned[(size_t) i] && (s < 0 || rl.tick[(size_t) i] < rl.tick[(size_t) s])) s = i;
                    if (s >= 0) rl.at.erase(rl.key[(size_t) s]);
                }
                if (s >= 0) {
                    rl.at[expert] = s;
                    rl.key[(size_t) s] = expert;
                    ram_dst = ram.data() + base + (size_t) s * slot_bytes;
                }
            }
            if (s >= 0) { rl.tick[(size_t) s] = ++rl.clock; ram_pinned[(size_t) s] = 1; }
            static std::atomic<int> dbg{0};
            const int dn = dbg.fetch_add(1);
            if (std::getenv("STRATA_EXL3_RAM_DEBUG") != nullptr && dn % 500 == 0)
                std::fprintf(stderr, "ram: lookup %d L%d e%d %s slot %d map=%zu filled=%d\n", dn, cur_layer, expert,
                             ram_src != nullptr ? "HIT" : "miss", s, rl.at.size(), rl.filled);
        }
        const std::array<size_t, 9>& ro = ram_off[(size_t) cur_layer];
        std::string path; size_t off = 0, bytes = 0;
        int piece = 0;
        auto push = [&](uint16_t* dst, const std::string& base, const char* plane, const uint16_t* src, size_t n) {
            // STRATA_EXL3_NO_PREAD=1 reads through the mmap instead (an A/B arm for the pread path)
            static const bool no_pread = std::getenv("STRATA_EXL3_NO_PREAD") != nullptr;
            if (ram_src != nullptr) {   // a hit: no file read at all
                jobs.push_back({dst, src, -1, 0, n, nullptr, ram_src + ro[(size_t) piece]});
                ++piece;
                return;
            }
            const int fd = (!no_pread && model->locate(base + plane, path, off, bytes) && bytes == n) ? fd_of(path) : -1;
            jobs.push_back({dst, src, fd, off, n, ram_dst != nullptr ? ram_dst + ro[(size_t) piece] : nullptr});
            ++piece;
        };
        const std::string bg = b + "gate_proj", bu = b + "up_proj", bd = b + "down_proj";
        push(gs->tr, bg, ".trellis", Lg.trellis, sizes[0]);
        push(gs->suh, bg, ".suh", Lg.suh, sizes[1]);
        push(gs->svh, bg, ".svh", Lg.svh, sizes[2]);
        push(us->tr, bu, ".trellis", Lu.trellis, sizes[3]);
        push(us->suh, bu, ".suh", Lu.suh, sizes[4]);
        push(us->svh, bu, ".svh", Lu.svh, sizes[5]);
        push(ds->tr, bd, ".trellis", Ld.trellis, sizes[6]);
        push(ds->suh, bd, ".suh", Ld.suh, sizes[7]);
        push(ds->svh, bd, ".svh", Ld.svh, sizes[8]);
    }

    void prepare_expert(Blocks& blocks, int slot, int expert) {
        const std::string base = "model.language_model.layers." + std::to_string(cur_layer) + ".mlp.experts." +
                                  std::to_string(expert) + ".";
        const auto g = model->linear(base + "gate_proj");
        const auto u = model->linear(base + "up_proj");
        const auto d = model->linear(base + "down_proj");
        if (g.in() != n_embd || u.in() != n_embd || g.out() != u.out() ||
            d.in() != g.out() || d.out() != n_embd)
            throw std::runtime_error("exl3: incompatible expert FFN dimensions");
        queue_expert(expert, &blocks.g[(size_t) slot], &blocks.u[(size_t) slot], &blocks.d[(size_t) slot], g, u, d);
        auto set = [](Exl3Mat& m, const Exl3Linear& L) {
            m.ki = L.ki; m.nj = L.nj; m.bits = L.bits; m.cb = L.cb;
        };
        set(blocks.gm[(size_t) slot], g); set(blocks.um[(size_t) slot], u); set(blocks.dm[(size_t) slot], d);
    }

    int fd_of(const std::string& path) {
#if defined(_WIN32)
        (void) path;
        return -1;
#else
        auto it = fds.find(path);
        if (it != fds.end()) return it->second;
        const int fd = ::open(path.c_str(), O_RDONLY);
        fds[path] = fd;
        return fd;
#endif
    }

    void stage_parallel() {
        if (jobs.empty()) return;
        stage_error.store(0, std::memory_order_relaxed);
        if (workers.empty()) {
            for (const Job& j : jobs) stage_one(j, stage[0]);
        } else {
        job_at.store(0, std::memory_order_relaxed);
        work_done.store(0, std::memory_order_relaxed);
        {
            const std::lock_guard<std::mutex> lk(work_mu);
            ++work_gen;
        }
        work_cv.notify_all();
        while (work_done.load(std::memory_order_acquire) < (uint64_t) n_workers) STRATA_EXL3_PAUSE();
        }
        const int error = stage_error.load(std::memory_order_relaxed);
        if (error != 0) throw std::runtime_error(std::string("exl3: expert staging failed: ") +
                                                cudaGetErrorString((cudaError_t) error));
    }

    void worker_loop(int wid) {
        strata::aux_cpus::pin_current_thread();
        // CUDA's current device is thread-local; a stage on GPU1 cannot use a worker's default GPU0.
        if (cudaSetDevice(device) != cudaSuccess) {
            std::fprintf(stderr, "exl3: staging worker cannot select device %d\n", device);
            std::abort();
        }
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(work_mu);
                work_cv.wait(lk, [&] { return work_quit || work_gen != seen; });
                if (work_quit) return;
                seen = work_gen;
            }
            for (;;) {
                const size_t i = job_at.fetch_add(1, std::memory_order_relaxed);
                if (i >= jobs.size()) break;
                stage_one(jobs[i], stage[(size_t) wid]);
            }
            work_done.fetch_add(1, std::memory_order_release);
        }
    }

    ~Impl() {
        int previous = -1;
        (void) cudaGetDevice(&previous); (void) cudaSetDevice(device);
        {
            const std::lock_guard<std::mutex> lk(work_mu);
            work_quit = true;
        }
        work_cv.notify_all();
        for (std::thread& t : workers) if (t.joinable()) t.join();
        for (Stage& st : stage) {
            for (uint8_t* p : st.buf) if (p != nullptr) (void) cudaFreeHost(p);
            for (cudaEvent_t e : st.done) if (e != nullptr) (void) cudaEventDestroy(e);
        }
#if !defined(_WIN32)
        for (auto& kv : fds) if (kv.second >= 0) (void) ::close(kv.second);
#endif
        for (void* p : { (void*) d_x, (void*) d_input, (void*) d_rows, (void*) d_out, (void*) d_hit, (void*) d_miss,
                         (void*) d_hit_rows, (void*) d_miss_rows }) (void) cudaFree(p);
        if (stage_done != nullptr) (void) cudaEventDestroy(stage_done);
        for (auto e : profile) if (e) (void) cudaEventDestroy(e);
        if (work_stream != nullptr) (void) cudaStreamDestroy(work_stream);
        for (auto& v : { &scratch.g, &scratch.u, &scratch.d })
            for (Slot& s : *v) { (void) cudaFree(s.tr); (void) cudaFree(s.suh); (void) cudaFree(s.svh); }
        for (Layer& l : layers)
            for (void* p : l.allocs) (void) cudaFree(p);
        workspace.reset();
        multi.reset();
        if (compute_stream) (void) cudaStreamDestroy(compute_stream);
        if (previous >= 0) (void) cudaSetDevice(previous);
    }
};

Exl3ExpertStream::Exl3ExpertStream(const std::string& model_dir, int n_embd, int n_layers, int slots_per_layer,
                                   void* stream, size_t ram_cache_bytes, size_t reserve_device_bytes,
                                   int layer_lo, int layer_hi, int prefill_tokens) {
    if (layer_hi < 0) layer_hi = n_layers;
    if (n_embd <= 0 || n_layers <= 0 || n_layers > 48 || slots_per_layer < -1 || slots_per_layer > 512 ||
        layer_lo < 0 || layer_lo >= layer_hi || layer_hi > n_layers)
        throw std::invalid_argument("exl3: invalid expert stream geometry");
    auto pending = std::make_unique<Impl>();
    impl_ = pending.get();
    if (cudaGetDevice(&impl_->device) != cudaSuccess ||
        cudaStreamCreateWithFlags(&impl_->compute_stream, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error("exl3: cannot create expert execution context");
    impl_->workspace = std::make_unique<Exl3MoeWorkspace>();
    impl_->multi = std::make_unique<MultiBuffers>();
    if (const char* timing = std::getenv("STRATA_EXL3_TIME"); timing && timing[0] && timing[0] != '0')
        for (auto& event : impl_->profile)
            if (cudaEventCreate(&event) != cudaSuccess) throw std::runtime_error("exl3: profiling event creation failed");
    impl_->dir = model_dir;
    impl_->model = std::make_unique<Exl3Model>(model_dir);
    impl_->n_embd = n_embd;
    impl_->n_layers = n_layers;
    impl_->layer_lo = layer_lo; impl_->layer_hi = layer_hi;
    // **THE STAGING COPIES GET THEIR OWN STREAM.**  They used to share the caller's, which put them behind
    // the resident experts' FFN: the double-buffered workers then blocked on their own DMA events for the
    // length of that FFN (measured: the staging went from 24 to 37 ms/token).  With a stream of their own
    // the copies start at once, and the miss FFN waits on one event instead.
    if (cudaStreamCreateWithFlags(&impl_->work_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&impl_->stage_done, cudaEventDisableTiming) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: EXL3 expert stream: no staging stream\n");
        std::abort();
    }
    (void) stream;
    // The staging workers: enough to saturate the host's memory bandwidth on the miss reads, not more.
    const char* nw = std::getenv("STRATA_EXL3_WORKERS");
    impl_->n_workers = nw != nullptr ? std::atoi(nw) : 4;
    if (impl_->n_workers < 0 || impl_->n_workers > 32)
        throw std::invalid_argument("exl3: STRATA_EXL3_WORKERS must be in [0,32]");
    // the largest piece any layer's expert has (the bitrate can differ per layer), and one expert's nine
    // pieces per layer for the host cache's slot layout
    impl_->ram_slot_bytes.assign((size_t) n_layers, 0);
    impl_->ram_off.assign((size_t) n_layers, {});
    for (int l = layer_lo; l < layer_hi; ++l) {
        const std::string b = "model.language_model.layers." + std::to_string(l) + ".mlp.experts.0.";
        size_t off = 0;
        int piece = 0;
        for (const char* p : {"gate_proj", "up_proj", "down_proj"}) {
            Exl3Linear L = impl_->model->linear(b + p);
            const size_t trb = (size_t) L.ki * L.nj * (256 * L.bits / 16) * 2;
            impl_->stage_bytes = std::max(impl_->stage_bytes, std::max(trb, (size_t) (L.in() + L.out()) * 2));
            for (const size_t n : {trb, (size_t) L.in() * 2, (size_t) L.out() * 2}) {
                impl_->ram_off[(size_t) l][(size_t) piece++] = off;
                off += n;
            }
        }
        impl_->ram_slot_bytes[(size_t) l] = off;
    }
    if (ram_cache_bytes > 0) {
        size_t per_layer = 0;
        for (size_t n : impl_->ram_slot_bytes) per_layer += n;
        if (per_layer > 0) impl_->ram_slots = (int) std::min<size_t>(ram_cache_bytes / per_layer, 1u << 20);
        if (impl_->ram_slots > 0) {
            impl_->ram_base.assign((size_t) n_layers, 0);
            size_t at = 0;
            for (int l = layer_lo; l < layer_hi; ++l) {
                impl_->ram_base[(size_t) l] = at;
                at += (size_t) impl_->ram_slots * impl_->ram_slot_bytes[(size_t) l];
            }
            try { impl_->ram.assign(at, 0); } catch (const std::exception&) { impl_->ram_slots = 0; }
            if (impl_->ram_slots > 0) {
                impl_->ram_layers.resize((size_t) n_layers);
                for (Impl::RamLayer& rl : impl_->ram_layers) {
                    rl.tick.assign((size_t) impl_->ram_slots, 0);
                    rl.key.assign((size_t) impl_->ram_slots, -1);
                }
            }
        }
    }
    impl_->stage.resize((size_t) (impl_->n_workers > 0 ? impl_->n_workers : 1));
    for (Impl::Stage& stg : impl_->stage) {
        for (int b = 0; b < 2; ++b) {
            if (cudaMallocHost((void**) &stg.buf[b], impl_->stage_bytes) != cudaSuccess) stg.buf[b] = nullptr;
            else if (cudaEventCreateWithFlags(&stg.done[b], cudaEventDisableTiming) != cudaSuccess) stg.done[b] = nullptr;
        }
        if (!stg.buf[0] || !stg.buf[1] || !stg.done[0] || !stg.done[1]) {
            // Without an event the DMA may still read a buffer when its next pread overwrites it.
            // Fall back as one pair, rather than running half of a double-buffered stage.
            for (int b = 0; b < 2; ++b) {
                if (stg.buf[b]) (void) cudaFreeHost(stg.buf[b]);
                if (stg.done[b]) (void) cudaEventDestroy(stg.done[b]);
                stg.buf[b] = nullptr; stg.done[b] = nullptr;
            }
        }
    }
    if (impl_->n_workers > 0) {
        impl_->workers.reserve((size_t) impl_->n_workers);
        for (int i = 0; i < impl_->n_workers; ++i) impl_->workers.emplace_back([im = impl_, i] { im->worker_loop(i); });
    }
    const int mk = impl_->max_k;
    auto alloc = [](uint16_t** p, size_t bytes) { if (cudaMalloc((void**) p, bytes ? bytes : 1) != cudaSuccess) std::abort(); };
    alloc(&impl_->d_x, (size_t) n_embd * 2);
    if (cudaMalloc((void**) &impl_->d_input, (size_t) n_embd * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("exl3: input allocation failed");
    alloc(&impl_->d_rows, (size_t) mk * n_embd * 2);
    if (cudaMalloc((void**) &impl_->d_out, (size_t) mk * n_embd * 4) != cudaSuccess) std::abort();
    if (cudaMalloc((void**) &impl_->d_hit, (size_t) mk * n_embd * 2) != cudaSuccess ||
        cudaMalloc((void**) &impl_->d_miss, (size_t) mk * n_embd * 2) != cudaSuccess ||
        cudaMalloc((void**) &impl_->d_hit_rows, (size_t) mk * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &impl_->d_miss_rows, (size_t) mk * sizeof(int)) != cudaSuccess) std::abort();
    // size the per-slot weight buffers from expert 0 of layer 0 (all experts in a layer share a shape)
    const std::string b0 = "model.language_model.layers." + std::to_string(layer_lo) + ".mlp.experts.0.";
    auto setup_scratch = [&](std::vector<Impl::Slot>& slots, std::vector<Exl3Mat>& mats, const char* proj) {
        Exl3Linear L = impl_->model->linear(b0 + proj);
        size_t trb = 0, suhb = 0, svhb = 0;
        for (int l = layer_lo; l < layer_hi; ++l) {
            const auto S = impl_->model->linear("model.language_model.layers." + std::to_string(l) +
                                                ".mlp.experts.0." + proj);
            trb = std::max(trb, (size_t) S.ki * S.nj * 32 * S.bits);
            suhb = std::max(suhb, (size_t) S.in() * 2);
            svhb = std::max(svhb, (size_t) S.out() * 2);
        }
        slots.resize((size_t) mk); mats.resize((size_t) mk);
        for (int i = 0; i < mk; ++i) {
            alloc(&slots[(size_t) i].tr, trb);
            alloc(&slots[(size_t) i].suh, suhb);
            alloc(&slots[(size_t) i].svh, svhb);
            slots[(size_t) i].trb = trb; slots[(size_t) i].suhb = suhb; slots[(size_t) i].svhb = svhb;
            mats[(size_t) i].trellis = slots[(size_t) i].tr;
            mats[(size_t) i].suh = slots[(size_t) i].suh;
            mats[(size_t) i].svh = slots[(size_t) i].svh;
            mats[(size_t) i].ki = L.ki; mats[(size_t) i].nj = L.nj; mats[(size_t) i].bits = L.bits; mats[(size_t) i].cb = L.cb;
        }
    };
    setup_scratch(impl_->scratch.g, impl_->scratch.gm, "gate_proj");
    setup_scratch(impl_->scratch.u, impl_->scratch.um, "up_proj");
    setup_scratch(impl_->scratch.d, impl_->scratch.dm, "down_proj");
    // Verify graphs wait for this pool. Allocating on that callback would synchronise the device
    // behind the waiting graph, so reserve the whole 8-token, max-k window before any launch.
    int max_ff = 0;
    for (int l = layer_lo; l < layer_hi; ++l)
        max_ff = std::max(max_ff, impl_->model->linear("model.language_model.layers." + std::to_string(l) +
                                                      ".mlp.experts.0.gate_proj").out());
    impl_->workspace->reserve(8 * mk, n_embd, max_ff, n_embd, true);
    impl_->multi->reserve(8, 8 * mk, n_embd);
    if (prefill_tokens > 0) {
        // Price EXL3 prompt scratch before auto-sizing the cache; device inputs need no FP32 copy.
        impl_->multi->reserve(std::max(8, prefill_tokens), 4096, n_embd, false);
        impl_->workspace->reserve(4096, n_embd, max_ff, n_embd);
    }
    (void) stream;

    // ---- the resident cache.  `-1` sizes it from the free VRAM: one slot in every layer costs
    // `per_slot` bytes, and what is left after this instance's own buffers is the budget.  The GGUF
    // path's `--expert-cache` is a global count because its blobs are one pool; here the stream is per
    // layer, so the count is per layer.
    int slots = slots_per_layer;
    if (slots < 0) {
        size_t per_slot = 0;
        for (int l = layer_lo; l < layer_hi; ++l) {
            const std::string b = "model.language_model.layers." + std::to_string(l) + ".mlp.experts.0.";
            for (const char* p : {"gate_proj", "up_proj", "down_proj"}) {
                Exl3Linear L = impl_->model->linear(b + p);
                per_slot += (size_t) L.ki * L.nj * (256 * L.bits / 16) * 2 + (size_t) (L.in() + L.out()) * 2;
            }
        }
        size_t free_b = 0, total_b = 0;
        (void) cudaMemGetInfo(&free_b, &total_b);
        const size_t reserve = reserve_device_bytes;
        slots = (per_slot > 0 && free_b > reserve) ? (int) std::min<size_t>((free_b - reserve) / per_slot, 512) : 0;
        std::fprintf(stderr, "strata generate: EXL3 expert cache auto: %.2f GiB free -> %d slots/layer\n",
                     (double) free_b / 1073741824.0, slots);
    }
    impl_->cache = slots > 0 ? slots : 0;
    if (impl_->cache > 0) {
        impl_->layers.resize((size_t) n_layers);
        for (int l = layer_lo; l < layer_hi; ++l) {
            Impl::Layer& L = impl_->layers[(size_t) l];
            L.tick.assign((size_t) impl_->cache, 0);
            const std::string b = "model.language_model.layers." + std::to_string(l) + ".mlp.experts.0.";
            std::vector<Impl::Slot>* vs[3] = { &L.g, &L.u, &L.d };
            std::vector<Exl3Mat>* ms[3] = { &L.gm, &L.um, &L.dm };
            const char* projs[3] = { "gate_proj", "up_proj", "down_proj" };
            for (int i = 0; i < 3; ++i) {
                Exl3Linear S = impl_->model->linear(b + projs[i]);
                const size_t trb = (size_t) S.ki * S.nj * (256 * S.bits / 16) * 2;
                const size_t suhb = (size_t) S.in() * 2, svhb = (size_t) S.out() * 2;
                uint16_t *tr = nullptr, *suh = nullptr, *svh = nullptr;
                alloc(&tr, trb * (size_t) impl_->cache);
                alloc(&suh, suhb * (size_t) impl_->cache);
                alloc(&svh, svhb * (size_t) impl_->cache);
                L.allocs.push_back(tr); L.allocs.push_back(suh); L.allocs.push_back(svh);
                vs[i]->resize((size_t) impl_->cache);
                ms[i]->resize((size_t) impl_->cache);
                for (int s = 0; s < impl_->cache; ++s) {
                    Impl::Slot& sl = (*vs[i])[(size_t) s];
                    sl.tr = tr + (size_t) s * (trb / 2);
                    sl.suh = suh + (size_t) s * (suhb / 2);
                    sl.svh = svh + (size_t) s * (svhb / 2);
                    sl.trb = trb; sl.suhb = suhb; sl.svhb = svhb;
                    Exl3Mat& M = (*ms[i])[(size_t) s];
                    M.trellis = sl.tr; M.suh = sl.suh; M.svh = sl.svh;
                    M.ki = S.ki; M.nj = S.nj; M.bits = S.bits; M.cb = S.cb;
                }
            }
        }
    }
    (void) pending.release();
}

Exl3ExpertStream::~Exl3ExpertStream() { delete impl_; }

int Exl3ExpertStream::slots_per_layer() const { return impl_->cache; }
int Exl3ExpertStream::ram_slots_per_layer() const { return impl_->ram_slots; }
size_t Exl3ExpertStream::ram_bytes() const { return impl_->ram.size(); }

size_t Exl3ExpertStream::bytes() const {
    size_t n = (size_t) impl_->n_embd * 2 + (size_t) impl_->max_k * impl_->n_embd * (2 + 4);
    for (auto& s : impl_->scratch.g) n += s.trb + s.suhb + s.svhb;
    for (auto& s : impl_->scratch.u) n += s.trb + s.suhb + s.svhb;
    for (auto& s : impl_->scratch.d) n += s.trb + s.suhb + s.svhb;
    for (const Impl::Layer& l : impl_->layers)
        for (const auto& v : { &l.g, &l.u, &l.d })
            for (const Impl::Slot& s : *v) n += s.trb + s.suhb + s.svhb;
    return n;
}

bool Exl3ExpertStream::run(int layer, const float* x_f32, const int* ids, int k, float* out, void* stream) {
    if (k == 0) return true;
    if (k < 0 || k > impl_->max_k || layer < impl_->layer_lo || layer >= impl_->layer_hi ||
        x_f32 == nullptr || ids == nullptr || out == nullptr) return false;
    for (int i = 0; i < k; ++i) if (ids[i] < 0 || ids[i] >= 512) return false;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < i; ++j)
            if (ids[i] == ids[j]) return run_multi(layer, x_f32, ids, 1, k, out, stream);
    const std::lock_guard<std::mutex> run_lock(impl_->run_mu);
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(impl_->device) != cudaSuccess) return false;
    struct RestoreDevice {
        int previous;
        ~RestoreDevice() { (void) cudaSetDevice(previous); }
    } restore{previous};
    try {
    if (stream == nullptr) stream = (void*) impl_->compute_stream;
    cudaStream_t st = (cudaStream_t) stream;
    const long n = impl_->n_embd;
    static const bool timing = std::getenv("STRATA_EXL3_TIME") != nullptr;
    const auto t_call0 = std::chrono::steady_clock::now();
    const auto t_up0 = std::chrono::steady_clock::now();
    if (cudaMemcpyAsync(impl_->d_input, x_f32, (size_t) n * sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess)
        return false;
    f32_to_f16_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(impl_->d_input, impl_->d_x, n);

    impl_->jobs.clear();
    impl_->ram_pinned.assign((size_t) impl_->ram_slots, 0);
    // ---- which slots this token's k experts live in.  A cache hit costs nothing; a miss reads the
    // expert out of the mmap and uploads it into the least recently used slot of this layer.
    std::vector<Exl3Mat> gm, um, dm;                 // the no-split path (no cache): one list of k
    std::vector<Exl3Mat> hg, hu, hd, mg, mu, md;     // the split path: resident rows and miss rows
    std::vector<int> hrow, mrow;
    Impl::Layer* cache = (impl_->cache > 0 && layer >= 0 && layer < (int) impl_->layers.size() &&
                          k <= impl_->cache) ? &impl_->layers[(size_t) layer] : nullptr;
    impl_->cur_layer = layer;
    if (cache != nullptr) {
        hg.reserve((size_t) k); hu.reserve((size_t) k); hd.reserve((size_t) k);
        mg.reserve((size_t) k); mu.reserve((size_t) k); md.reserve((size_t) k);
        hrow.reserve((size_t) k); mrow.reserve((size_t) k);
        for (int i = 0; i < k; ++i) {
            const int e = ids[i];
            auto it = cache->at.find(e);
            int s = -1;
            const bool resident = it != cache->at.end();
            if (resident) {
                s = it->second;
                ++impl_->hits;
            } else {
                ++impl_->misses;
                if (cache->filled < impl_->cache) {
                    s = cache->filled++;                 // still filling: the slots are handed out in order
                } else {
                    s = (int) (std::min_element(cache->tick.begin(), cache->tick.end()) - cache->tick.begin());
                    for (auto victim = cache->at.begin(); victim != cache->at.end(); ++victim)
                        if (victim->second == s) { cache->at.erase(victim); break; }
                }
                impl_->prepare_expert(*cache, s, e);
                cache->at[e] = s;
            }
            cache->tick[(size_t) s] = ++cache->clock;
            if (resident) {
                hrow.push_back(i);
                hg.push_back(cache->gm[(size_t) s]); hu.push_back(cache->um[(size_t) s]); hd.push_back(cache->dm[(size_t) s]);
            } else {
                mrow.push_back(i);
                mg.push_back(cache->gm[(size_t) s]); mu.push_back(cache->um[(size_t) s]); md.push_back(cache->dm[(size_t) s]);
            }
        }
    } else {
        gm.resize((size_t) k); um.resize((size_t) k); dm.resize((size_t) k);
        for (int i = 0; i < k; ++i) {
            impl_->prepare_expert(impl_->scratch, i, ids[i]);
            gm[(size_t) i] = impl_->scratch.gm[(size_t) i];
            um[(size_t) i] = impl_->scratch.um[(size_t) i];
            dm[(size_t) i] = impl_->scratch.dm[(size_t) i];
        }
        // no cache: nothing is resident, so the whole FFN follows the staging (the split has nothing to run early)
        impl_->stage_parallel();
        exl3_moe_rows(gm.data(), um.data(), dm.data(), ids, k, impl_->d_x, impl_->d_rows, stream, impl_->workspace.get());
    }
    // **THE RESIDENT HALF'S FFN RUNS WHILE THE MISSES ARE READ.**  Their weights are already in VRAM, so
    // nothing has to wait for the host: the kernel is enqueued before the workers start, and the miss FFN
    // follows the staging copies on the same stream.  Without the split the whole FFN waited for the last
    // miss, which is most of the doorbell.
    if (!hrow.empty())
        exl3_moe_rows_part(hg.data(), hu.data(), hd.data(), (int) hrow.size(), impl_->d_x, impl_->d_hit, stream, impl_->workspace.get());
    impl_->stage_parallel();
    const auto t_up1 = std::chrono::steady_clock::now();
    // the miss FFN (and the scatter) read what the staging wrote: order them after it
    if (!impl_->jobs.empty()) {
        (void) cudaEventRecord(impl_->stage_done, impl_->work_stream);
        (void) cudaStreamWaitEvent(st, impl_->stage_done, 0);
    }
    if (!mrow.empty()) {
        (void) cudaMemcpyAsync(impl_->d_hit_rows, hrow.data(), hrow.size() * sizeof(int), cudaMemcpyHostToDevice, st);
        (void) cudaMemcpyAsync(impl_->d_miss_rows, mrow.data(), mrow.size() * sizeof(int), cudaMemcpyHostToDevice, st);
        exl3_moe_rows_part(mg.data(), mu.data(), md.data(), (int) mrow.size(), impl_->d_x, impl_->d_miss, stream, impl_->workspace.get());
        exl3_rows_scatter(impl_->d_hit, impl_->d_hit_rows, (int) hrow.size(), impl_->d_miss, impl_->d_miss_rows,
                          (int) mrow.size(), (int) n, impl_->d_rows, stream);
    } else if (!hrow.empty()) {
        (void) cudaMemcpyAsync(impl_->d_hit_rows, hrow.data(), hrow.size() * sizeof(int), cudaMemcpyHostToDevice, st);
        exl3_rows_scatter(impl_->d_hit, impl_->d_hit_rows, (int) hrow.size(), nullptr, nullptr, 0, (int) n,
                          impl_->d_rows, stream);
    }
    const auto t_ffn1 = std::chrono::steady_clock::now();
    f16_to_f32_rows_kernel<<<(unsigned)((n * k + 255) / 256), 256, 0, st>>>(impl_->d_rows, impl_->d_out, n * k);
    if (cudaMemcpyAsync(out, impl_->d_out, (size_t) n * k * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    const cudaError_t se = cudaStreamSynchronize(st);
    if (impl_->work_stream != nullptr) (void) cudaStreamSynchronize(impl_->work_stream);   // its copies feed this layer's FFN
    if (se != cudaSuccess) std::fprintf(stderr, "exl3 stream run(layer=%d): %s\n", layer, cudaGetErrorString(se));
    if (timing) {
        const auto t_d2h1 = std::chrono::steady_clock::now();
        impl_->up_ms += std::chrono::duration<double, std::milli>(t_up1 - t_up0).count();
        impl_->ffn_ms += std::chrono::duration<double, std::milli>(t_ffn1 - t_up1).count();
        impl_->d2h_ms += std::chrono::duration<double, std::milli>(t_d2h1 - t_ffn1).count();
        impl_->call_ms += std::chrono::duration<double, std::milli>(t_d2h1 - t_call0).count();
        if (layer == impl_->n_layers - 1) {
            const long long uses = impl_->hits + impl_->misses;
            std::fprintf(stderr, "exl3 stream: %lld experts, %.1f%% cached, host %.1f%% (%lld reads) | upload %.1f ms, "
                                 "ffn %.1f ms, d2h %.1f ms, pool %.1f ms per token\n", uses,
                         100.0 * (double) impl_->hits / (double) std::max<long long>(uses, 1),
                         100.0 * (double) impl_->ram_hits / (double) std::max<long long>(impl_->ram_hits + impl_->ram_misses, 1),
                         impl_->ram_misses, impl_->up_ms, impl_->ffn_ms, impl_->d2h_ms, impl_->call_ms);
            impl_->hits = impl_->misses = 0;
            impl_->ram_hits = impl_->ram_misses = 0;
            impl_->up_ms = impl_->ffn_ms = impl_->d2h_ms = impl_->call_ms = 0;
        }
    }
    return se == cudaSuccess;
    } catch (const std::exception& error) {
        (void) cudaStreamSynchronize(stream ? (cudaStream_t) stream : impl_->compute_stream);
        (void) cudaStreamSynchronize(impl_->work_stream);
        for (auto& layer_cache : impl_->layers) { layer_cache.at.clear(); layer_cache.filled = 0; }
        for (auto& layer_cache : impl_->ram_layers) { layer_cache.at.clear(); layer_cache.filled = 0; }
        std::fprintf(stderr, "exl3 stream(layer=%d): %s\n", layer, error.what());
        return false;
    }
}

bool Exl3ExpertStream::run_multi(int layer, const float* x_f32, const int32_t* ids, int tokens, int k,
                                float* out, void* stream) {
    return run_multi_impl(layer, x_f32, ids, tokens, k, out, stream, false);
}

bool Exl3ExpertStream::run_device(int layer, const float* x_f32, const int32_t* ids, int tokens, int k,
                                 float* out, void* stream) {
    return run_multi_impl(layer, x_f32, ids, tokens, k, out, stream, true);
}

bool Exl3ExpertStream::run_multi_impl(int layer, const float* x_f32, const int32_t* ids, int tokens, int k,
                                     float* out, void* stream, bool device_io) {
    if (tokens == 0) return true;
    if (tokens < 0 || k <= 0 || k > impl_->max_k || layer < impl_->layer_lo || layer >= impl_->layer_hi ||
        !x_f32 || !ids || !out || tokens > INT32_MAX / k) return false;
    const std::lock_guard<std::mutex> lock(impl_->run_mu);
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(impl_->device) != cudaSuccess) return false;
    struct RestoreDevice {
        int previous;
        ~RestoreDevice() { (void) cudaSetDevice(previous); }
    } restore{previous};
    const auto st = stream ? (cudaStream_t) stream : impl_->compute_stream;
    const bool timing = impl_->profile[0] != nullptr;
    const auto call_started = std::chrono::steady_clock::now();
    try {
        auto& plan = impl_->route_plan;
        plan.build(ids, tokens, k, 512);
        constexpr int tile_limit = 4096;
        const int tile = std::min(tokens * k, tile_limit);
        const int n = impl_->n_embd;
        auto& batch = *impl_->multi;
        batch.reserve(tokens, tile, n, !device_io);
        if (!device_io && cudaMemcpyAsync(batch.input.p, x_f32, (size_t) tokens * n * sizeof(float),
                                          cudaMemcpyHostToDevice, st) != cudaSuccess)
            throw std::runtime_error("exl3: multi input upload failed");
        const long input_elements = (long) tokens * n;
        f32_to_f16_rows_kernel<<<(unsigned) ((input_elements + 255) / 256), 256, 0, st>>>(
            device_io ? x_f32 : batch.input.p, batch.x.p, input_elements);
        impl_->cur_layer = layer;
        auto* cache = impl_->cache > 0 ? &impl_->layers[(size_t) layer] : nullptr;
        if (cache && !cache->at.empty()) {
            std::vector<uint8_t> resident(plan.experts.size());
            for (size_t g = 0; g < plan.experts.size(); ++g)
                resident[g] = cache->at.count(plan.experts[g]) != 0;
            plan.prioritize_resident(resident);
        }
        const int wave_size = cache ? impl_->cache : impl_->max_k;
        std::vector<Exl3Mat> gates, ups, downs, route_g, route_u, route_d;
        std::vector<int> input_rows;
        input_rows.reserve((size_t) tile);
        route_g.reserve((size_t) tile); route_u.reserve((size_t) tile); route_d.reserve((size_t) tile);
        for (size_t first = 0; first < plan.experts.size(); first += (size_t) wave_size) {
            const size_t end = std::min(first + (size_t) wave_size, plan.experts.size());
            gates.clear(); ups.clear(); downs.clear(); impl_->jobs.clear();
            impl_->ram_pinned.assign((size_t) impl_->ram_slots, 0);
            // Slots referenced by this wave cannot be evicted until its final tile has completed.
            std::vector<uint8_t> pinned(cache ? (size_t) impl_->cache : 0, 0);
            for (size_t group = first; group < end; ++group) {
                const int expert = plan.experts[group];
                int slot = (int) (group - first);
                Impl::Blocks* blocks = &impl_->scratch;
                if (cache) {
                    blocks = cache;
                    const auto hit = cache->at.find(expert);
                    if (hit != cache->at.end()) { slot = hit->second; ++impl_->hits; }
                    else {
                        ++impl_->misses;
                        if (cache->filled < impl_->cache) slot = cache->filled++;
                        else {
                            slot = -1;
                            for (int s = 0; s < impl_->cache; ++s)
                                if (!pinned[(size_t) s] && (slot < 0 || cache->tick[(size_t) s] < cache->tick[(size_t) slot]))
                                    slot = s;
                            if (slot < 0) throw std::runtime_error("exl3: all expert slots pinned");
                            for (auto it = cache->at.begin(); it != cache->at.end(); ++it)
                                if (it->second == slot) { cache->at.erase(it); break; }
                        }
                        impl_->prepare_expert(*cache, slot, expert);
                        cache->at[expert] = slot;
                    }
                    pinned[(size_t) slot] = 1;
                    cache->tick[(size_t) slot] = ++cache->clock;
                } else {
                    ++impl_->misses;
                    impl_->prepare_expert(impl_->scratch, slot, expert);
                }
                gates.push_back(blocks->gm[(size_t) slot]); ups.push_back(blocks->um[(size_t) slot]);
                downs.push_back(blocks->dm[(size_t) slot]);
            }
            if (timing) {
                for (const auto& job : impl_->jobs) impl_->upload_bytes += job.bytes;
                (void) cudaEventRecord(impl_->profile[0], impl_->work_stream);
            }
            impl_->stage_parallel();
            if (timing) (void) cudaEventRecord(impl_->profile[1], impl_->work_stream);
            if (cudaEventRecord(impl_->stage_done, impl_->work_stream) != cudaSuccess ||
                cudaStreamWaitEvent(st, impl_->stage_done, 0) != cudaSuccess)
                throw std::runtime_error("exl3: expert upload event failed");
            const int route_end = plan.offsets[end];
            size_t group = first;
            for (int route = plan.offsets[first]; route < route_end;) {
                while (route >= plan.offsets[group + 1]) ++group;
                const int count = std::min(tile, (device_io ? plan.offsets[group + 1] : route_end) - route);
                input_rows.clear(); route_g.clear(); route_u.clear(); route_d.clear();
                for (int r = route; r < route + count; ++r) {
                    while (r >= plan.offsets[group + 1]) ++group;
                    const size_t local = group - first;
                    input_rows.push_back(plan.destinations[(size_t) r] / k);
                    route_g.push_back(gates[local]); route_u.push_back(ups[local]); route_d.push_back(downs[local]);
                }
                if (cudaMemcpyAsync(batch.input_rows.p, input_rows.data(), (size_t) count * sizeof(int),
                                    cudaMemcpyHostToDevice, st) != cudaSuccess)
                    throw std::runtime_error("exl3: route upload failed");
                const long elements = (long) count * n;
                if (timing) (void) cudaEventRecord(impl_->profile[2], st);
                if (device_io && count >= 16) {
                    exl3_moe_group_gemm(route_g[0], route_u[0], route_d[0], count, batch.x.p,
                                         batch.input_rows.p, batch.output.p, (void*) st, *impl_->workspace);
                } else {
                    exl3_moe_rows_indexed(route_g.data(), route_u.data(), route_d.data(), count, batch.x.p,
                                          batch.input_rows.p, batch.rows.p, (void*) st, *impl_->workspace);
                    f16_to_f32_rows_kernel<<<(unsigned) ((elements + 255) / 256), 256, 0, st>>>(batch.rows.p, batch.output.p, elements);
                }
                if (timing) (void) cudaEventRecord(impl_->profile[3], st);
                if (device_io) {
                    if (cudaMemcpyAsync(batch.destinations.p, plan.destinations.data() + route, (size_t) count * sizeof(int),
                                        cudaMemcpyHostToDevice, st) != cudaSuccess)
                        throw std::runtime_error("exl3: destination upload failed");
                    scatter_route_output<<<(unsigned) ((elements + 255) / 256), 256, 0, st>>>(
                        batch.output.p, batch.destinations.p, out, count, n);
                    if (cudaStreamSynchronize(st) != cudaSuccess) throw std::runtime_error("exl3: device FFN failed");
                } else {
                if (cudaMemcpyAsync(batch.host_output, batch.output.p, (size_t) elements * sizeof(float),
                                    cudaMemcpyDeviceToHost, st) != cudaSuccess || cudaStreamSynchronize(st) != cudaSuccess)
                    throw std::runtime_error("exl3: multi FFN execution failed");
                for (int r = 0; r < count; ++r)
                    std::memcpy(out + (size_t) plan.destinations[(size_t) route + r] * n,
                                batch.host_output + (size_t) r * n, (size_t) n * sizeof(float));
                }
                if (timing) {
                    float ms = 0;
                    if (cudaEventElapsedTime(&ms, impl_->profile[2], impl_->profile[3]) == cudaSuccess) impl_->ffn_ms += ms;
                }
                route += count;
            }
            if (timing) {
                float ms = 0;
                if (cudaEventElapsedTime(&ms, impl_->profile[0], impl_->profile[1]) == cudaSuccess) impl_->up_ms += ms;
            }
        }
        if (timing) {
            impl_->call_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - call_started).count();
            if (layer + 1 == impl_->layer_hi) {
                std::fprintf(stderr, "exl3 CUDA%d [%d,%d) rows=%d: pool %.2f ms, upload %.2f ms (%.2f MiB), FFN %.2f ms, cache %lld/%lld\n",
                    impl_->device, impl_->layer_lo, impl_->layer_hi, tokens, impl_->call_ms, impl_->up_ms,
                    (double) impl_->upload_bytes / (1 << 20), impl_->ffn_ms, impl_->hits, impl_->hits + impl_->misses);
                impl_->call_ms = impl_->up_ms = impl_->ffn_ms = 0;
                impl_->hits = impl_->misses = 0; impl_->upload_bytes = 0;
            }
        }
        return true;
    } catch (const std::exception& e) {
        // A failed window is not reusable: uploads may have replaced only part of a cache slot.
        (void) cudaStreamSynchronize(st);
        (void) cudaStreamSynchronize(impl_->work_stream);
        for (auto& cache : impl_->layers) { cache.at.clear(); cache.filled = 0; }
        for (auto& cache : impl_->ram_layers) { cache.at.clear(); cache.filled = 0; }
        std::fprintf(stderr, "exl3 multi(layer=%d): %s\n", layer, e.what());
        return false;
    }
}

}  // namespace strata::kernels
