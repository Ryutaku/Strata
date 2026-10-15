# EXL3 CUDA validation — 2026-10-15

This fork integrates [upstream PR #1764](https://github.com/Niko1221/Strata/pull/1764),
head `4c5030fb589cce4c86892184fbead2c0271f8743`, on upstream `main` at `296b59d2` (0.1.44.1).

## Corrections

- The reconstruction's row Hadamard pass put the output dimension in `grid.y`. The real language
  head has 248,320 outputs, exceeding NVIDIA's 65,535 limit. Put outputs on `grid.x` and check every
  reconstruction launch immediately. The new 128×65,664 parity case crosses that limit.
- An explicit EXL3 `--expert-cache N` also enabled the GGUF cache and its hit callback, which expects
  a different host object. Preserve EXL3's slots per layer and disable the GGUF cache for EXL3.
- Validate tensor bounds, byte counts, integer overflow, trellis dimensions, integer bitrate and
  FP16 scale shapes before decoding. Reject unsupported biased linears and incompatible engine modes.
- Fix the MoE checker's missing CUDA runtime header and use of buffers after `cudaFree` in its timing
  loop. Check timing errors and the requested expert count.
- Correct the independent Python bitstream oracle to MSB-first uint32 order and pin a stored-word
  example. Quantize dense row chunks with start zero rather than pointers before their allocations.

## Environment and checks

Ubuntu 24.04, CUDA 12.8, `sm_89`, NVIDIA L40-24Q vGPU, 24 GiB. Real weights:
`thelastspark/Swift-Qwen3.8-Flash-Next-exl3`, 6.05 bpw, EXL3 1.5.1, including its n-gram table.
The build used Release, Ninja, CUDA enabled and `STRATA_BUILD_TESTS=ON`.

- CUDA full build succeeds.
- Python EXL3 suite: 29 tests pass, including loader and command-line rejection checks.
- CUDA CTest subset: 28/28 pass (`exl3|gr_parity|gr_multi|bf16|shared_expert`).
- CPU-only build and CTest: 17/17 pass; command-line tests needing the GPU engine are skipped there.
- Real K=6 expert and K=8 dense GEMV agree with the CPU reference (relative error about 2.9e-4).
- Real MoE layer 0, 10 of 16 experts: relative error 1.0e-3; corrected GPU timing 0.266 ms/layer.
  This is an isolated layer measurement, not whole-model throughput.
- Trellis dense path: 1,079 roles load, 301 projections execute without faults, all 22 checked
  converter transforms match.
- Full Q8 dense model, normal capture path, PLE enabled, 512-token capacity: arithmetic smoke test
  returns `42` and stops at EOS with both zero and 64 EXL3 cache slots per layer.

The full-model check is a short arithmetic regression, not a quality benchmark, a long-context test,
or evidence of a speedup over GGUF. These were the initial single-token checks; the windowed CUDA
extension below adds MTP, batched sessions and explicit layer splitting. HIP, SYCL and Windows
were not built here.

## Reproducing the full-model check

Install `tools/exl3/requirements.txt` into the test Python environment, select a GPU with enough free
VRAM, and run from the source checkout:

```sh
python tools/exl3/smoke.py --engine build-cuda12/strata --model /path/to/model --gpu 0 --cache 0
python tools/exl3/smoke.py --engine build-cuda12/strata --model /path/to/model --gpu 0 --cache 64
```

The harness never stops services or modifies the model. It checks the tokenizer round trip, decodes
the generated IDs, checks the answer and rejects accidental activation of the GGUF expert cache.
Use `--output result.json` to retain the decoded response and timings.

## Windowed CUDA extension

EXL3 is opt-in with `--exl3`; the GGUF path keeps its existing expert backend. The extension adds:

- Stable counting-sort routing, O(routes + experts), so an expert is uploaded once per window or
  prompt chunk. Outputs are scattered back to their original token/top-k positions before combining.
- Indexed multi-token FFNs with per-device/per-stream workspaces. Each K partition has a separate
  partial result; a fixed-order reduction removes atomic-scheduling differences between slot counts.
- GPU-resident prompt activations and grouped tensor-core GEMMs. Large expert groups reconstruct
  each projection once. The temporary route output is tiled at 4096 rows.
- Per-stage dense weights, expert caches and staging workers. Workers select their owning device
  and run on auxiliary CPUs. Host hand-offs support cards without peer-to-peer access.
- Reuse of the verifier's commit/rollback, slot checkpoints and cancellation. Raw BF16/F16 embedding
  gather works in the verifier, prefill and MTP prompt-KV path. The target's Q8 head can be borrowed
  by the verifier and compatible packed drafter without duplicating the full vocabulary head.
- EXL3 cache replacement stays separate from GGUF residency/adaptation. Resident groups run before
  misses, avoiding a sequential scan evicting every cached expert before its use. Prompt scratch,
  slot sessions and drafter bindings are budgeted before the auto cache consumes remaining VRAM.

### Configuration

Export the tokenizer once with `tools/exl3/tokenizer_to_pack.py --hf MODEL --out data/exl3 --check`.
Its vocabulary now uses the server's token-to-ID map contract. Existing list exports should be
regenerated before use with `serve/server.py`.

Example engine arguments for the tested 48 + 24 GiB L40 vGPU pair:

```sh
strata --serve --exl3 MODEL --ple-exl3 MODEL/ngram_embedding.safetensors \
  --expert-cache auto --prefill 16384 --max-context 262144 --kv int8 \
  --spec 4 --mtp MTP_RUNTIME --batch 2 --batch-mtp --batch-groups 1 \
  --layer-split 32 --split-device 1 --pool-workers 8 --pcie-frac 0 \
  --ple-io mmap --vram-reserve-mib 1024
```

`MTP_RUNTIME` is a compatible packed Qwen3.8 drafter containing `dense.txt`, `dense.bin`,
`experts.bin` and optionally `draft_vocab.bin`. This uses the existing packed drafter; it does
not import an EXL3 MTP checkpoint. Target expert weights remain in their original EXL3 format.
Without MTP, omit `--mtp` and `--batch-mtp`; keep `--spec T` (T >= 2), which the server requires
for its verifier windows regardless of whether a drafter is loaded.

The test config format is the server's JSON format: `args`, `cwd`, `tokenizer`, and optional `env`.
For the HTTP server also set `exe`, `model_name` and `parallel: 2`. Bind to localhost unless an API
key is configured. Useful opt-in environment settings:

- `STRATA_SPLIT_MTP_BATCH=1`: batched MTP prompt KV on the last GPU of a split.
- `STRATA_EXL3_GEMV_SPLIT=1`: fewer K partitions in indexed FFNs. Measured faster for this hardware's
  multi-token windows. A fixed value preserves the same per-route arithmetic across slot counts;
  changing it can change generated tokens. Unset retains the earlier partition count.
- `STRATA_EXL3_WORKERS=0..32`: expert upload workers per stage (default 4).
- `STRATA_EXL3_TIME=1`: upload bytes/time, FFN time and expert-cache counters per stage.

`--prefill auto[:N]` reserves owned EXL3 prompt buffers at the requested ceiling; it does not
borrow a GGUF cache. Explicit boundaries such as `--layer-split 32` are required. Raw trellis
dense diagnostic overrides (`STRATA_EXL3_DENSE=exl3` or `only:...`) are rejected with windows.
As in the existing batch scheduler, pipelined groups and batch MTP are separate scheduler modes;
use one group for MTP across concurrent slots. Multi-card boundaries are implemented generally,
but only two NVIDIA cards were exercised here.

### Validation and measurements

Ubuntu 24.04, CUDA 12.8, sm_89, EPYC 9654, NVIDIA L40-48Q + L40-24Q vGPUs, no GPU P2P.
Target: `thelastspark/Swift-Qwen3.8-Flash-Next-exl3` 6.05 bpw with PLE; INT8 KV.

- 41 affected CUDA/CPU CTests pass, covering EXL3, verifier batch/commit, GDN, GR, BF16,
  shared experts, native multi-row kernels, RoPE and INT8 KV. Synthetic EXL3 FFNs use independent
  CPU reconstruction, mixed integer bitrates/codebooks, two simultaneous streams and both GPUs.
  Actual 2560/640 geometry also checks identical results when route counts grow and shrink.
- Two 32-token batch-MTP outputs match their solo references exactly with layer split enabled.
  A separate 64-token lifecycle test checks cancellation, the unaffected slot, resuming the
  cancelled slot, the other slot's next turn and reuse of the cancelled slot. All checks pass.
  Single-GPU batching and a two-group, two-GPU pipeline also match both 32-token solo references
  with no MTP runtime loaded.
- Two independently submitted 261632-token prompts both retrieve distinct FIRST/MIDDLE/LAST
  markers. Each has 262144 context capacity and its own INT8 KV state. The first prompt reads
  while the second waits for admission; its decode overlaps the second prompt's chunked read.
  This is actual long-input validation, not only allocation of a 256K cache.
- Initial 32K + 512 generated tokens: 647 prefill / 28.5 decode tok/s. Resident-first cache order
  raises the same 8192-chunk prefill to 1209 tok/s. A 16384-chunk run with K split 1 measures
  1489 prefill / 32.1 decode tok/s, completing all 512 generated tokens.
- The two near-256K retrieval prompts with 32768 chunks measure 2629 / 1503 prefill tok/s;
  their combined request wall time is 282.2 s. These short 40-token retrieval answers do not
  establish sustained decode throughput. Their run overlapped a CPU build and is a functional
  stress result, not a controlled speed comparison.
- The final candidate repeats both 261632-token marker checks successfully with 16384 chunks,
  fixed K split 1 and two slots. Prefill is 1986 / 1185 tok/s and combined wall time is 359.4 s.
  This smaller-chunk configuration is slower for full input reads on this machine.
- A 128K prompt followed by all 512 requested tokens with 16384 chunks, K split 1 and 12
  upload workers measures 2051 prefill / 30.9 decode tok/s. More upload workers do not establish
  a general speed improvement; the runtime default remains 4.
- The actual HTTP API returns `43` and `44` for two concurrent arithmetic requests with MTP,
  two batch slots, layer split, 262144 context capacity and INT8 KV enabled.
- Isolated resident 40-route FFN at actual dimensions: 1.00/1.11 ms per layer-window with the
  original K partition count, 0.57/0.61 ms with K split 1, on GPU0/GPU1 respectively. This excludes
  weight upload, attention, dense projections, MTP and commit; it is not whole-model throughput.

The requested 3000 prefill / 150 decode tok/s targets have **not** been demonstrated. The 32K
prompt buffer on the 24 GiB card leaves effectively no resident expert cache in the tested full
configuration; smaller chunks trade prompt work for a larger decode cache. These are targeted
functional and performance checks, not a broad model-quality evaluation. Backends other than
CUDA remain unvalidated for this extension.

Reproduce with the same tokenizer and a JSON engine configuration:

```sh
python tools/exl3/bench.py --engine build-cuda12/strata --config config.json \
  --prompt-tokens 32768 --max-new 512 --output logs/perf
python tools/exl3/bench.py --engine build-cuda12/strata --config config.json \
  --prompt-tokens 261632 --max-new 128 --needle --batch --output logs/long-context
python tools/exl3/lifecycle.py --engine build-cuda12/strata --config config.json --output logs/lifecycle
python tools/exl3/api_smoke.py --config server-config.json --output logs/api
```

The harnesses retain engine logs, input/output token IDs and result JSON. The performance harness
also records the engine SHA-256. Neither harness manages services or changes the model. A `passed`
result validates completion/marker checks; it does not mean a throughput target was reached.
