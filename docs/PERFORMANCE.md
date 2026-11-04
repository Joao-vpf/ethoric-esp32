# Performance and quality

Board: ESP32-S3 N16R8, 240 MHz, flash QIO 80 MHz, octal PSRAM 80 MHz, ESP-IDF 6.1, context 160.

## Speed

| Firmware iteration | ms/token | tokens/s |
|---|---|---|
| Early int8 pipeline | 411 | 2.4 |
| Intermediate iterations | 294, 227 | 3.4, 4.4 |
| Int4 MLP, 2 cores, 6.75 / 7.03 MiB in PSRAM | 150 | 6.7 |
| Int4 MLP, 2 cores, 6.98 / 7.03 MiB in PSRAM | **134** | **7.5** |

Final boot log (excerpt): model 7194 KB, CRC verified in 0.46 s, weights copied to PSRAM in 0.4 s, SIMD vs C
bit-exact for int8 and int4, `1 core: 156 ms/token | 2 cores: 117 ms/token` measured at boot, 128 generated
tokens in 17.1 s.

## Quality

Measured as perplexity against the fp32 model.

| Variant | Perplexity change |
|---|---|
| Int8 weights, float activations | negligible |
| + int8 activations (W8A8, group 32) | +0.11 % |
| Int4 MLP, group 32 | +7.9 to +8.4 % (top-1 agreement with int8 about 85 %) |
| Int4 MLP, group 16 | +6.9 % |
| Int4 gate/up, int8 down | +5.7 % |
| Int4 MLP, group 8 | +4.5 % |

Int4 trades about 8 % perplexity for speed and a smaller footprint. Use `--mlp-bits 8` for the best quality, or
`--mlp4-layers` to quantize only some layers.

## Reproducing

```bash
host/build/ethoric_host model.bin --ids <ids> --ppl
host/build/ethoric_host model.bin --ids <ids> --bench
python tools/check_against_hf.py --dir ethoric_original --wq 8 --block 0
```

Host timings are not representative of the ESP32.
