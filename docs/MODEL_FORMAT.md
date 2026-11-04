# ETHW model format

Little endian. Every tensor starts at a multiple of 64 bytes so it can be read directly from memory-mapped flash.

## Versions

| Version | Addition |
|---|---|
| 1 | Per-row int8 weights with fp32 scales |
| 2 | Each row padded with zeros to a multiple of 16 bytes (leading dimension `ld`) for SIMD |
| 3 | `flags` bit 2: MLP matrices in int4; `m4mask` selects which layers use int4 |

The loader accepts all three. Old files without a vocabulary still load.

## Header (128 bytes)

| Offset | Field |
|---|---|
| 0 | Magic `ETHW` |
| 4.. | Version, model dimensions, flags, sliding-window layer mask, `rope_theta`, `norm_eps` |
| 68 | CRC32 of the body (bytes 128 to end) |
| 72 | Total file size (u64) |
| 80 | Vocabulary table offset |
| 84 | Vocabulary table size in bytes |
| 88 | `m4mask`: u32 bitmask of layers whose MLP is int4 (v3) |
| 92 | BPE tables offset (u32, 0 = absent) |
| 96 | BPE tables size in bytes (u32) |

The CRC and total size let the device detect partial or corrupted flash writes (`VERIFY_CRC`).

## Body

1. Embedding (int8, tied with the LM head) and final norm.
2. For every layer: norms, QK-norm, attention gate, Canon A/C kernels, then the matrices
   `q, k, v, o, mlp_gate, mlp_up, mlp_down`.

Each matrix is a scale array followed by the weights.

- **int8**: one fp32 scale per output row, `s = max|w| / 127`; rows padded to a multiple of 16.
- **int4**: groups of 32 columns, one bf16 scale per group (chosen by an MSE search over clipping ratios).
  Each group occupies 16 bytes: the low nibble holds column `j` (+8 offset) and the high nibble column `j+16`
  (two's complement).

## Vocabulary (optional)

`u32 offsets[V+1]` followed by the byte strings. The exporter honours the `tokenizer.json` decoder
(ByteLevel, Metaspace, ByteFallback, Replace); special tokens map to empty strings. `--check` compares the table
with the `tokenizers` library on random id sequences.

## BPE tokenizer tables (optional)

Lets the device turn text into ids (`components/ethoric/include/bpe.hpp`). Written when `tokenizer.json` uses the
supported pipeline (NFC, the three Split steps, ByteLevel without regex, plain BPE); otherwise the exporter skips them.

| Part | Content |
|---|---|
| header (16 B) | `"BPE1"`, `n_merges`, `n_classes`, reserved (u32 each) |
| `byte_id[256]` | u16 token id of each single byte after the GPT-2 byte-to-unicode mapping |
| `digit_id[1110]` | u16 ids of the added tokens `0`-`9`, `00`-`99`, `000`-`999` (index: 1 digit `d`, 2 digits `10+dd`, 3 digits `110+ddd`) |
| merges | `n_merges` x `{u32 key = (left<<16)|right, u16 rank, u16 result}`, sorted by key |
| classes | `n_classes` x u32 `(first_codepoint << 3) | class`, sorted; 0 other, 1 Lu, 2 Ll, 3 other letter, 4 M, 5 P/S, 6 N, 7 White_Space |

Limitations: input is assumed NFC-normalized; special-token strings such as `</s>` in the text are encoded as ordinary
text; Unicode classes come from Python's `unicodedata` (15.1) and may differ marginally from the tokenizers library for
very recent code points.
