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
or evidence of a speedup over GGUF. HIP, SYCL and Windows were not built here. MTP/speculative decode,
batched sessions and layer splitting remain unsupported for this EXL3 path.

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
