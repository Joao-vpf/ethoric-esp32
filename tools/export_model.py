#!/usr/bin/env python3
"""Exports Ethoric-10M to a single binary file ("ETHW" format) for the C++ decoder.

    python export_model.py --dir ethoric_original --out ethoric_int8.bin                    # int8 weights
    python export_model.py --dir ethoric_original --out ethoric_int4.bin --mlp-bits 4       # int4 MLP

Layout summary (little endian; see docs/MODEL_FORMAT.md for the full description). Every tensor starts at a
multiple of 64 bytes from the beginning of the file so it can be read straight from memory-mapped flash.

  128-byte header: magic "ETHW", version (2, or 3 with an int4 MLP), model dimensions, flags, window mask,
                   rope_theta, norm_eps, CRC32 of the body, total size, vocabulary offset/size, int4 layer mask.
  body:            embedding (tied with the LM head), final norm, then per layer: norms, QK-norm, gate, Canon
                   kernels, and the seven matrices q, k, v, o, mlp_gate, mlp_up, mlp_down.
  vocabulary:      optional token table so the device can print text instead of ids.

Quantization:
  int8 matrices: symmetric, one scale per OUTPUT ROW, s = max|w| / 127, q = clip(rint(w / s), -127, 127).
                 Rows are zero-padded to a multiple of 16 bytes (format v2) for the SIMD kernels.
  int4 matrices: symmetric, one bf16 scale per GROUP of 32 columns, chosen by a small MSE search over clipping
                 ratios (format v3, MLP only).
The int8 scheme equals `--wq 8 --block 0` in reference_decoder.py, so perplexities are comparable.
"""
import argparse
import json
import os
import re
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_decoder import read_safetensors  # noqa: E402


def bytes_to_unicode():
    """GPT-2 byte-to-unicode table (ByteLevel): byte -> character."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _decoder_steps(dec):
    if not dec:
        return []
    if dec.get("type") == "Sequence":
        out = []
        for d in dec.get("decoders", []):
            out += _decoder_steps(d)
        return out
    return [dec]


def build_vocab_table(tj: dict, vocab: int):
    """List of `vocab` byte strings: what each id appends to the text. Follows the tokenizer.json "decoder"
    (ByteLevel, Metaspace, ByteFallback, Replace); special tokens become empty."""
    model = tj.get("model", {})
    table = [b""] * vocab
    pieces = {}
    for tok, i in (model.get("vocab") or {}).items():
        if 0 <= i < vocab:
            pieces[i] = tok
    special = set()
    for at in tj.get("added_tokens", []):
        i = at["id"]
        if 0 <= i < vocab:
            pieces[i] = at["content"]
            if at.get("special"):
                special.add(i)
    steps = _decoder_steps(tj.get("decoder"))
    kinds = [d.get("type") for d in steps]
    u2b = {c: b for b, c in bytes_to_unicode().items()}
    for i in range(vocab):
        t = pieces.get(i)
        if t is None or i in special:
            continue
        raw = None
        for d in steps:
            ty = d.get("type")
            if ty == "Replace":
                pat = d.get("pattern", {})
                if "String" in pat:
                    t = t.replace(pat["String"], d.get("content", ""))
            elif ty == "Metaspace":
                rep = d.get("replacement", d.get("str_rep", "\u2581"))
                t = t.replace(rep, " ")
            elif ty == "ByteFallback":
                m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", t)
                if m:
                    raw = bytes([int(m.group(1), 16)])
            elif ty == "ByteLevel":
                try:
                    raw = bytes(u2b[c] for c in t)
                except KeyError:
                    raw = t.encode("utf-8")
        table[i] = raw if raw is not None else t.encode("utf-8")
    return table, kinds


def pack_vocab(table):
    offs, pos = [], 0
    for b in table:
        offs.append(pos)
        pos += len(b)
    offs.append(pos)
    return struct.pack(f"<{len(offs)}I", *offs) + b"".join(table)


def check_vocab(dirpath, table, n=300, seed=0):
    """Compares against tok.decode() (the `tokenizers` library) on random id sequences."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("  (tokenizers not installed: --check skipped)")
        return True
    import random
    tok = Tokenizer.from_file(os.path.join(dirpath, "tokenizer.json"))
    rnd = random.Random(seed)
    bad = 0
    for _ in range(n):
        ids = [rnd.randrange(len(table)) for _ in range(rnd.randrange(1, 12))]
        ref = tok.decode(ids)
        mine = b"".join(table[i] for i in ids).decode("utf-8", errors="replace")
        # the library may strip a leading space (Strip/Metaspace); compare without it
        if ref != mine and ref.lstrip() != mine.lstrip():
            bad += 1
            if bad <= 3:
                print(f"  MISMATCH ids={ids}\n    lib ={ref!r}\n    here={mine!r}")
    print(f"  --check: {n - bad}/{n} sequences equal to tok.decode")
    return bad == 0


# ---------------------------------------------------------------------------------------------------------------------
# On-device BPE tokenizer tables (optional "BPE1" section, see docs/MODEL_FORMAT.md)
# ---------------------------------------------------------------------------------------------------------------------
BPE_SPLIT1 = (r" ?(?:\p{L}\p{M}*)+['\u2019](?:\p{L}\p{M}*)+| ?(?:\p{Lu}\p{M}*)+(?=(?:\p{Lu}\p{M}*)(?:\p{Ll}\p{M}*))"
              r"| ?(?:\p{Lu}\p{M}*)?(?:\p{Ll}\p{M}*)+| ?(?:\p{Lu}\p{M}*)+| ?(?:[\p{P}\p{S}]\p{M}*)+|[^\S\r\n]*[\n\r]+|[^\S\r\n]+")
BPE_SPLIT2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d|n't)"
BPE_SPLIT3 = r"\p{N}{1,3}"
WHITE_SPACE = set(range(0x09, 0x0E)) | {0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000} | set(range(0x2000, 0x200B))


def bpe_unsupported_reason(tj):
    """None if tokenizer.json uses exactly the pipeline implemented by bpe.hpp, otherwise a human-readable reason."""
    m, pre = tj.get("model", {}), tj.get("pre_tokenizer") or {}
    if m.get("type") != "BPE" or m.get("dropout") or m.get("continuing_subword_prefix") or m.get("end_of_word_suffix") \
            or m.get("byte_fallback") or m.get("ignore_merges"):
        return "model is not a plain byte-level BPE"
    if (tj.get("normalizer") or {}).get("type") not in (None, "NFC"):
        return "unsupported normalizer"
    parts = pre.get("pretokenizers", [])
    if pre.get("type") != "Sequence" or len(parts) != 4:
        return "pre_tokenizer is not the expected Sequence of 4 steps"
    want = (BPE_SPLIT1, BPE_SPLIT2, BPE_SPLIT3)
    for part, rx in zip(parts[:3], want):
        if part.get("type") != "Split" or part.get("behavior") != "Isolated" or part.get("invert") \
                or part.get("pattern", {}).get("Regex") != rx:
            return "pre_tokenizer Split steps differ from the implemented ones"
    if parts[3].get("type") != "ByteLevel" or parts[3].get("use_regex") or parts[3].get("add_prefix_space"):
        return "pre_tokenizer ByteLevel step differs"
    return None


def unicode_class_table():
    """Sorted u32 entries (first_codepoint << 3 | class); classes: 0 other, 1 Lu, 2 Ll, 3 other letter, 4 M, 5 P/S, 6 N, 7 space."""
    import unicodedata

    def cls(c):
        if c in WHITE_SPACE:
            return 7
        cat = unicodedata.category(chr(c))
        if cat == "Lu":
            return 1
        if cat == "Ll":
            return 2
        if cat in ("Lt", "Lm", "Lo"):
            return 3
        if cat[0] == "M":
            return 4
        if cat[0] in "PS":
            return 5
        if cat[0] == "N":
            return 6
        return 0
    out, prev = [], -1
    for c in range(0x110000):
        k = cls(c)
        if k != prev:
            out.append((c << 3) | k)
            prev = k
    return out


def build_bpe_section(tj):
    """Serializes the BPE tables for the on-device encoder."""
    vocab, b2u = tj["model"]["vocab"], bytes_to_unicode()
    byte_id = [vocab.get(b2u[b], 0xFFFF) for b in range(256)]
    assert 0xFFFF not in byte_id, "vocabulary lacks some single-byte tokens"
    added = {t["content"]: t["id"] for t in tj["added_tokens"] if not t["special"]}
    digits = [str(d) for d in range(10)] + [f"{d:02d}" for d in range(100)] + [f"{d:03d}" for d in range(1000)]
    assert all(d in added for d in digits), "tokenizer.json lacks the added tokens for 1-3 digit strings"
    merges, seen = [], set()
    for r, mg in enumerate(tj["model"]["merges"]):
        a, b = mg if isinstance(mg, list) else mg.split(" ")
        key = (vocab[a] << 16) | vocab[b]
        if key not in seen:
            seen.add(key)
            merges.append((key, r, vocab[a + b]))
    merges.sort()
    classes = unicode_class_table()
    assert max(vocab.values()) < 0xFFFF and len(tj["model"]["merges"]) < 0xFFFF
    blob = b"BPE1" + struct.pack("<III", len(merges), len(classes), 0)
    blob += struct.pack("<256H", *byte_id) + struct.pack("<1110H", *[added[d] for d in digits])
    blob += b"".join(struct.pack("<IHH", *m) for m in merges)
    blob += struct.pack(f"<{len(classes)}I", *classes)
    return blob


def check_bpe(dir_, tj, text_n=300):
    """Compares the reference encoder built from the tables with the `tokenizers` library on random ASCII text."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("  (tokenizers not installed: BPE --check skipped)")
        return True
    import random
    import tokenizer_ref
    lib = Tokenizer.from_file(os.path.join(dir_, "tokenizer.json"))
    rng, bad = random.Random(0), 0
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789      \n.,;:!?'\"-()"
    for _ in range(text_n):
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 60)))
        a, b = lib.encode(t).ids, tokenizer_ref.encode(tj, t)
        if a != b:
            bad += 1
            print(f"  MISMATCH {t!r}\n    lib ={a}\n    here={b}")
    print(f"  --check: {text_n - bad}/{text_n} random texts encode identically to the tokenizers library")
    return bad == 0


def bf16_round(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 (round to nearest even), returned as uint16."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def quant_int4_groups(w: np.ndarray, search: bool = True):
    """Symmetric int4 per GROUP of 32 columns (bf16 scale). Returns (q int8 [rows,ng,32] in [-8,7], bf16 scale uint16
    [rows,ng]). Each group scale is picked among max/7 * {1,...,0.55} by the smallest squared error (outlier clipping)."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    rows, cols = w.shape
    ld = (cols + 31) // 32 * 32
    if ld != cols:
        w = np.concatenate([w, np.zeros((rows, ld - cols), np.float32)], axis=1)
    g = w.reshape(rows, ld // 32, 32)
    amax = np.abs(g).max(axis=-1)
    best_err = np.full(amax.shape, np.inf, np.float64)
    best_q = np.zeros(g.shape, np.int8)
    best_s = np.ones(amax.shape, np.uint16)
    for m in (np.linspace(1.0, 0.55, 10) if search else [1.0]):
        sb = bf16_round(np.where(amax > 0, amax / 7.0 * m, 1.0).astype(np.float32))
        sf = bf16_to_f32(sb)
        q = np.clip(np.rint(g / sf[..., None]), -8, 7).astype(np.int8)
        err = ((q * sf[..., None] - g) ** 2).sum(-1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_q = np.where(better[..., None], q, best_q)
        best_s = np.where(better, sb, best_s)
    return best_q, best_s


def pack_int4(q: np.ndarray) -> np.ndarray:
    """[rows,ng,32] int4 -> [rows, ng*16] bytes: byte j = column j in the low nibble (+8) and column j+16 in the high nibble."""
    lo = (q[..., :16].astype(np.int16) + 8).astype(np.uint8)
    hi = (q[..., 16:].astype(np.int16) & 0xF).astype(np.uint8)
    return (lo | (hi << 4)).reshape(q.shape[0], -1)


def dequant_int4(q: np.ndarray, sb: np.ndarray, cols: int) -> np.ndarray:
    return (q * bf16_to_f32(sb)[..., None]).reshape(q.shape[0], -1)[:, :cols]


def quant_rows(w: np.ndarray):
    w = np.ascontiguousarray(w, dtype=np.float32)
    s = np.abs(w).max(axis=1, keepdims=True) / np.float32(127.0)
    s[s == 0] = 1.0
    q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
    return q, s[:, 0].astype(np.float32)


class Writer:
    def __init__(self, f):
        self.f = f
        self.sizes = {}

    def align(self):
        pad = (-self.f.tell()) % 64
        if pad:
            self.f.write(b"\0" * pad)

    def raw(self, arr: np.ndarray, tag: str):
        self.align()
        b = np.ascontiguousarray(arr).tobytes()
        self.f.write(b)
        self.sizes[tag] = self.sizes.get(tag, 0) + len(b)

    def f32(self, arr, tag="fp32"):
        self.raw(np.asarray(arr, dtype=np.float32), tag)

    def mat4(self, w, tag="int4"):
        q, sb = quant_int4_groups(w)
        self.raw(sb, "scales_bf16")
        self.raw(pack_int4(q), tag)

    def mat(self, w, tag="int8"):
        q, s = quant_rows(w)
        pad = (-q.shape[1]) % 16          # format v2: every row occupies a multiple of 16 bytes (zeros at the end)
        if pad:
            q = np.concatenate([q, np.zeros((q.shape[0], pad), np.int8)], axis=1)
        self.raw(s, "scales")
        self.raw(q, tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="ethoric_original")
    ap.add_argument("--out", default="ethoric_int8.bin")
    ap.add_argument("--mlp-bits", type=int, choices=(4, 8), default=8,
                    help="4 = MLP matrices in int4 (groups of 32, bf16 scale): the whole model fits in PSRAM")
    ap.add_argument("--mlp4-layers", default="all",
                    help="with --mlp-bits 4: MLP layers stored in int4, e.g. 0,1,2,3 (default: all). The others stay int8")
    ap.add_argument("--no-vocab", action="store_true", help="do not embed the vocabulary (old format)")
    ap.add_argument("--no-bpe", action="store_true", help="do not embed the BPE tables (the host must tokenize prompts)")
    ap.add_argument("--check", action="store_true", help="verify the vocabulary against the tokenizers library")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.dir, "config.json")))
    g = lambda *ks: next(cfg[k] for k in ks if k in cfg)
    d, nl = int(g("d_model", "hidden_size")), int(g("n_layer", "num_hidden_layers"))
    nh, nkv = int(g("n_head", "num_attention_heads")), int(g("n_kv_head", "num_key_value_heads"))
    hd, ffn = int(cfg["head_dim"]), int(g("ffn_hidden", "intermediate_size"))
    vocab, max_seq = int(cfg["vocab_size"]), int(g("max_seq_len", "max_position_embeddings"))
    canon = cfg.get("canon") or {}
    kernel = int(canon.get("kernel", 4))
    window = int(cfg.get("attn_window", 0) or 0)
    eos = int(cfg.get("eos_token_id") or 0)
    doc_masking = bool(cfg.get("doc_masking", True))
    canon_doc = doc_masking and bool(canon.get("doc_masking", True))
    assert cfg.get("attn_gate") == "headwise" and cfg.get("qk_norm") == "per_head"
    assert set(canon.get("positions", [])) == {"A", "C"} and cfg.get("tie_word_embeddings", True)
    assert nl <= 32, "window_mask has 32 bits"
    wmask = 0
    for i in cfg.get("attn_window_layers") or []:
        wmask |= 1 << int(i)

    m4mask = 0
    if a.mlp_bits == 4:
        m4mask = (1 << nl) - 1 if a.mlp4_layers == "all" else sum(1 << int(t) for t in a.mlp4_layers.split(",") if t != "")
        assert m4mask, "--mlp4-layers is empty"
    W = read_safetensors(os.path.join(a.dir, "model.safetensors"))
    with open(a.out, "wb") as f:
        hdr = struct.pack("<4sI11iIIff", b"ETHW", 3 if a.mlp_bits == 4 else 2, d, nl, nh, nkv, hd, ffn, vocab, max_seq, kernel, window, eos,
                          int(doc_masking) | (int(canon_doc) << 1) | ((a.mlp_bits == 4) << 2), wmask,
                          float(cfg["rope_theta"]), float(g("norm_eps", "rms_norm_eps")))
        f.write(hdr + b"\0" * (128 - len(hdr)))
        w = Writer(f)
        w.mat(W["model.embed_tokens.weight"], "embed_int8")
        w.f32(W["model.final_norm.weight"])
        for i in range(nl):
            p = f"model.layers.{i}."
            for n in ("attn_in_norm", "attn_out_norm", "mlp_in_norm", "mlp_out_norm"):
                w.f32(W[p + n + ".weight"])
            w.f32(W[p + "attn.q_norm.weight"])
            w.f32(W[p + "attn.k_norm.weight"])
            w.f32(W[p + "attn.gate_proj.weight"])
            w.f32(W[p + "attn.gate_proj.bias"])
            w.f32(W[p + "canon_a.weight"][:, 0, :])
            w.f32(W[p + "canon_c.weight"][:, 0, :])
            for n in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                if a.mlp_bits == 4 and n.startswith("mlp.") and (m4mask >> i) & 1:
                    w.mat4(W[p + n + ".weight"])
                else:
                    w.mat(W[p + n + ".weight"])
        voff = vbytes = 0
        tj_path = os.path.join(a.dir, "tokenizer.json")
        if not a.no_vocab and os.path.exists(tj_path):
            table, kinds = build_vocab_table(json.load(open(tj_path, encoding="utf-8")), vocab)
            w.align()
            voff = f.tell()
            blob = pack_vocab(table)
            f.write(blob)
            vbytes = len(blob)
            print(f"vocabulary: {vbytes:,} bytes, decoder={kinds}, {sum(1 for b in table if not b)} empty tokens")
            if a.check and not check_vocab(a.dir, table):
                sys.exit("vocabulary does not match tok.decode: do not use this file for text")
        boff = bbytes = 0
        if not a.no_bpe and os.path.exists(tj_path):
            tj = json.load(open(tj_path, encoding="utf-8"))
            why = bpe_unsupported_reason(tj)
            if why:
                print(f"BPE tables skipped: {why}")
            else:
                w.align()
                boff = f.tell()
                blob = build_bpe_section(tj)
                f.write(blob)
                bbytes = len(blob)
                print(f"BPE tables: {bbytes:,} bytes")
                if a.check and not check_bpe(a.dir, tj):
                    sys.exit("BPE tables do not reproduce the tokenizers library: do not rely on on-device tokenization")
        total = f.tell()
    # total size and body CRC32 (bytes 128..end) go in the header so the ESP32 can detect a partial flash write
    with open(a.out, "r+b") as f:
        f.seek(128)
        crc = zlib.crc32(f.read()) & 0xFFFFFFFF
        f.seek(68)
        f.write(struct.pack("<IQII", crc, total, voff, vbytes))
        f.seek(88)
        f.write(struct.pack("<III", m4mask, boff, bbytes))
    print(f"{a.out}: {total:,} bytes ({total / 2**20:.2f} MiB), body crc32 = {crc:08x}")
    for k, v in w.sizes.items():
        print(f"  {k:12s} {v:>10,} bytes")
    print(f"  (the rest is header and alignment padding: {total - sum(w.sizes.values()):,} bytes)")


if __name__ == "__main__":
    main()
