# Architecture

## Model

Ethoric-10M is an English base language model: d_model 256, 12 layers, 8 query / 2 key-value heads (GQA, head
dim 32), SwiGLU FFN of width 721, vocabulary 5048 with tied embeddings, context 512.

Details the engine reproduces exactly:

- RoPE with half-split rotation, applied after a per-head QK RMS-norm.
- Headwise sigmoid gate on the attention output.
- Peri-LN: RMS-norms before and after both the attention and MLP blocks.
- Canon A/C: depthwise causal convolutions (kernel 4) whose state is reset at document boundaries (EOS id 1).
- Sliding window of 16 tokens on layer 0 (ring-buffer KV cache).

## Software structure

| Unit | Role |
|---|---|
| `components/ethoric/include/ethoric.hpp` | `Model` (ETHW loader, CRC32, PSRAM relocation, vocabulary), `Runner` (one decoding step), compute kernels, `Parallel` interface |
| `components/ethoric/include/sampler.hpp` | Temperature, top-k and repetition-penalty sampling (xorshift32) |
| `main/main.cpp` | Boot sequence: map model, verify, benchmark and select kernels, self-check SIMD, choose cores, generate |
| `main/dual_core.hpp` | Second-core worker that sleeps between tokens |
| `components/ethoric/include/bpe.hpp` | Byte-level BPE encoder (text to ids) over tables embedded in the model file |
| `components/ethoric/include/mini_json.hpp` | Flat JSON parser and string escaper for the HTTP API |
| `components/ethoric/include/dns_captive.hpp` | DNS reply builder for the captive portal (every name resolves to the board) |
| `main/net_server.cpp` | WiFi (station or access point), captive portal, and the HTTP server (`POST /completion`, `GET /health`, chat page) |
| `host/host_main.cpp` | Same engine on a desktop: logits dump, perplexity, generation, benchmarks |

## Compute path

1. **Quantization** – weights are per-row int8 (attention, embedding) or group-of-32 int4 with bf16 scales
   (MLP, optional). Activations are quantized per group of 32 to int8 (W8A8 / W4A8).
2. **Kernels** – a portable C reference (`mv_a8`, `mv_a4`) defines the result. Xtensa PIE kernels
   (`mv_a8_pie`, `mv_a4_pie`, inline assembly, enabled by `ETHORIC_PIE`) must reproduce it bit-for-bit; this is
   verified at every boot (`a8_selfcheck`, `a4_selfcheck`) and on the host through `ETHORIC_PIE_EMUL`.
   Float kernels (`ref`, `u4`, `r4`, `r8`) are benchmarked at boot and the fastest is selected.
3. **Two cores** – matrix-vector products are split by output rows and attention is split by heads. Results are
   independent of the split, so single- and dual-core outputs are bit-identical.
4. **Memory** – weights stay memory-mapped in flash, or are copied to PSRAM when they fit
   (`PSRAM_RESERVE_KB` is kept free). Small float tables live in PSRAM too, leaving internal RAM for the stack.

## Optimization history

Measured decode latency on the board across successive firmware iterations (same prompt, 128 new tokens):
411, 294, 227, 150 and finally **134 ms/token**. The iterations combined, in this order of introduction, the
techniques described above: kernel selection and PSRAM relocation, W8A8 arithmetic with PIE SIMD kernels, the int4
MLP, dual-core execution, and relocating (almost) all weights into PSRAM. The last two runs both used the int4
model on two cores; they differ in how much of the model was resident in PSRAM (6.75 of 7.03 MiB at 150 ms/token,
6.98 of 7.03 MiB at 134 ms/token).

The workload is compute-bound on scalar code (6–7 cycles per multiply-accumulate); raw PSRAM reads reach about
90 MB/s and flash about 32 MB/s, so data movement only dominates once the arithmetic is SIMD.

## ESP-IDF notes

- `sdkconfig` overrides `sdkconfig.defaults`; delete it when changing defaults.
- The task watchdog is 5 s: long loops on core 0 must yield (`YIELD_MS`).
- The `model` partition (type `data`, subtype `0x40`, offset `0x110000`, size `0xB00000`) is mapped with
  `esp_partition_mmap`.
- The main component is compiled with `-O3`.
