#!/usr/bin/env python3
"""NumPy reference decoder for Ethoric-10M (no torch), faithful to the original modeling code.

Block (Peri-LN + Canon):
    h = RMSNorm_attn_in(x);  h = h + CanonA(h)
    x = x + RMSNorm_attn_out(Attention(h))
    h = RMSNorm_mlp_in(x);   h = h + CanonC(h)
    x = x + RMSNorm_mlp_out(SwiGLU(h))

Attention: GQA (8 query heads, 2 KV heads), per-head QK-norm BEFORE RoPE, half-split (non-interleaved) RoPE,
a per-head sigmoid gate (computed from h after Canon A) applied before o_proj, and a 16-token local window on
the layers listed in `attn_window_layers`. The embedding is tied (lm_head = embed_tokens).

It decodes one token at a time (KV cache + Canon state), which is exactly what the ESP32 does. Document
semantics (EOS closes a document) follow the full Hugging Face forward pass: the document id of a token is the
number of EOS tokens before it.

Usage:
    python reference_decoder.py --dir ethoric_original --prompt "The climate change" --max-new 40
    python reference_decoder.py --dir ethoric_original --prompt "..." --wq 8      # simulate int8 weights
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

# --------------------------------------------------------------------------- safetensors

_DT = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8,
       "I32": np.int32, "I64": np.int64}


def read_safetensors(path: str) -> dict:
    """Minimal reader: 8 bytes (u64 LE) with the JSON header size, the JSON header, then the tensor data."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        raw = np.asarray(mm[base + a: base + b])
        dt = meta["dtype"]
        if dt == "BF16":
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=_DT[dt])
            if dt == "F16":
                arr = arr.astype(np.float32)
        out[name] = np.array(arr.reshape(meta["shape"]))
    return out


# --------------------------------------------------------------------------- (simulated) quantization

def fake_quant(w: np.ndarray, bits: int = 8, block: int = 0) -> np.ndarray:
    """Quantizes and dequantizes (symmetric). block=0: one scale per output row;
    block=N: one scale per block of N inputs (zero-padded: 721 is not a multiple of 32/128)."""
    out, inn = w.shape
    if block <= 0:
        block = inn
    nb = -(-inn // block)
    pad = nb * block - inn
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    wb = wp.reshape(out, nb, block)
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(wb).max(-1, keepdims=True) / qmax
    s[s == 0] = 1.0
    q = np.clip(np.rint(wb / s), -qmax, qmax)
    return (q * s).reshape(out, nb * block)[:, :inn].astype(np.float32)


# --------------------------------------------------------------------------- operators

def rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps)) * w


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Half-split rotation: (x1, x2) -> (x1*cos - x2*sin, x2*cos + x1*sin). x: [..., hd]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


# --------------------------------------------------------------------------- model

class Layer:
    pass


class State:
    """KV cache + the two Canon states of every layer."""

    def __init__(self, m: "Ethoric"):
        L, k1 = m.n_layer, m.kernel - 1
        self.k = [np.zeros((m.max_seq, m.n_kv, m.hd), np.float32) for _ in range(L)]
        self.v = [np.zeros((m.max_seq, m.n_kv, m.hd), np.float32) for _ in range(L)]
        self.kdoc = [np.zeros(m.max_seq, np.int64) for _ in range(L)]
        self.ca = [np.zeros((k1, m.d), np.float32) for _ in range(L)]
        self.cc = [np.zeros((k1, m.d), np.float32) for _ in range(L)]
        self.ca_doc = [np.full(k1, -1, np.int64) for _ in range(L)]
        self.cc_doc = [np.full(k1, -1, np.int64) for _ in range(L)]
        self.pos = 0
        self.doc = 0


class Ethoric:
    def __init__(self, model_dir: str, wq: int = 0, block: int = 0):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        g = lambda *ks: next(cfg[k] for k in ks if k in cfg)
        self.d = int(g("d_model", "hidden_size"))
        self.n_layer = int(g("n_layer", "num_hidden_layers"))
        self.n_head = int(g("n_head", "num_attention_heads"))
        self.n_kv = int(g("n_kv_head", "num_key_value_heads"))
        self.hd = int(cfg["head_dim"])
        self.ffn = int(g("ffn_hidden", "intermediate_size"))
        self.max_seq = int(g("max_seq_len", "max_position_embeddings"))
        self.theta = float(cfg["rope_theta"])
        self.eps = float(g("norm_eps", "rms_norm_eps"))
        self.vocab = int(cfg["vocab_size"])
        self.eos = int(cfg.get("eos_token_id") or 0)
        canon = cfg.get("canon") or {}
        self.kernel = int(canon.get("kernel", 4))
        self.doc_masking = bool(cfg.get("doc_masking", True))
        self.canon_doc_masking = self.doc_masking and bool(canon.get("doc_masking", True))
        self.window = int(cfg.get("attn_window", 0) or 0)
        self.window_layers = set(cfg.get("attn_window_layers") or [])
        assert cfg.get("attn_gate") == "headwise", "only the headwise gate is implemented"
        assert cfg.get("qk_norm") == "per_head", "only per-head QK-norm is implemented"
        assert set(canon.get("positions", [])) == {"A", "C"} and canon.get("enabled", True)
        assert cfg.get("tie_word_embeddings", True)
        self.rep = self.n_head // self.n_kv
        self.scale = np.float32(1.0 / math.sqrt(self.hd))

        W = read_safetensors(os.path.join(model_dir, "model.safetensors"))
        q = (lambda w: fake_quant(w, wq, block)) if wq else (lambda w: w)
        self.embed = q(W["model.embed_tokens.weight"]).astype(np.float32)
        self.final_norm = W["model.final_norm.weight"]
        self.layers = []
        for i in range(self.n_layer):
            p = f"model.layers.{i}."
            L = Layer()
            L.attn_in_norm = W[p + "attn_in_norm.weight"]
            L.attn_out_norm = W[p + "attn_out_norm.weight"]
            L.mlp_in_norm = W[p + "mlp_in_norm.weight"]
            L.mlp_out_norm = W[p + "mlp_out_norm.weight"]
            L.q_norm = W[p + "attn.q_norm.weight"]
            L.k_norm = W[p + "attn.k_norm.weight"]
            L.q = q(W[p + "attn.q_proj.weight"])
            L.k = q(W[p + "attn.k_proj.weight"])
            L.v = q(W[p + "attn.v_proj.weight"])
            L.o = q(W[p + "attn.o_proj.weight"])
            L.gate_w = W[p + "attn.gate_proj.weight"]     # small (8x256): kept in floating point
            L.gate_b = W[p + "attn.gate_proj.bias"]
            L.canon_a = W[p + "canon_a.weight"][:, 0, :]  # [d, k]
            L.canon_c = W[p + "canon_c.weight"][:, 0, :]
            L.mlp_gate = q(W[p + "mlp.gate_proj.weight"])
            L.mlp_up = q(W[p + "mlp.up_proj.weight"])
            L.mlp_down = q(W[p + "mlp.down_proj.weight"])
            L.window = self.window if i in self.window_layers else None
            self.layers.append(L)

        half = self.hd // 2
        freqs = 1.0 / (np.float32(self.theta) ** (np.arange(half, dtype=np.float32) / np.float32(half)))
        ang = np.outer(np.arange(self.max_seq, dtype=np.float32), freqs.astype(np.float32))
        self.cos, self.sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    # ---- Canon: causal depthwise convolution (kernel k) as a residual branch, with document masking
    def _canon(self, w, h, state, sdoc, doc):
        win = np.vstack([state, h[None]])                  # [k, d], oldest first
        wdoc = np.concatenate([sdoc, [doc]])
        keep = (wdoc == doc).astype(np.float32)
        out = (win * w.T * keep[:, None]).sum(0)           # tap i multiplies win[i]
        state[:-1] = state[1:]
        state[-1] = h
        sdoc[:-1] = sdoc[1:]
        sdoc[-1] = doc
        return out

    def new_state(self) -> State:
        return State(self)

    def step(self, tok: int, st: State) -> np.ndarray:
        """Processes ONE token and returns the logits [vocab]."""
        pos = st.pos
        if pos >= self.max_seq:
            raise ValueError(f"maximum context of {self.max_seq} tokens exceeded")
        adoc = st.doc if self.doc_masking else 0
        cdoc = st.doc if self.canon_doc_masking else 0
        H, Hkv, hd, rep = self.n_head, self.n_kv, self.hd, self.rep
        cos, sin = self.cos[pos], self.sin[pos]
        x = self.embed[tok].astype(np.float32).copy()
        for i, L in enumerate(self.layers):
            h = rms(x, L.attn_in_norm, self.eps)
            h = h + self._canon(L.canon_a, h, st.ca[i], st.ca_doc[i], cdoc)
            qv = (L.q @ h).reshape(H, hd)
            kv = (L.k @ h).reshape(Hkv, hd)
            vv = (L.v @ h).reshape(Hkv, hd)
            qv, kv = rms(qv, L.q_norm, self.eps), rms(kv, L.k_norm, self.eps)   # before RoPE
            qv, kv = rope(qv, cos, sin), rope(kv, cos, sin)
            st.k[i][pos], st.v[i][pos], st.kdoc[i][pos] = kv, vv, adoc
            lo = 0 if L.window is None else max(0, pos - L.window + 1)
            K, V = st.k[i][lo:pos + 1], st.v[i][lo:pos + 1]             # [n, Hkv, hd]
            keep = st.kdoc[i][lo:pos + 1] == adoc
            s = np.einsum("grd,ngd->grn", qv.reshape(Hkv, rep, hd), K) * self.scale
            s = np.where(keep[None, None, :], s, -np.inf)
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s)
            p = p / p.sum(-1, keepdims=True)
            o = np.einsum("grn,ngd->grd", p, V).reshape(H, hd)          # head j uses KV head j // rep
            o = o * sigmoid(L.gate_w @ h + L.gate_b)[:, None]           # headwise gate
            x = x + rms(L.o @ o.reshape(-1), L.attn_out_norm, self.eps)

            h = rms(x, L.mlp_in_norm, self.eps)
            h = h + self._canon(L.canon_c, h, st.cc[i], st.cc_doc[i], cdoc)
            m = L.mlp_down @ (silu(L.mlp_gate @ h) * (L.mlp_up @ h))
            x = x + rms(m, L.mlp_out_norm, self.eps)
        x = rms(x, self.final_norm, self.eps)
        logits = self.embed @ x
        st.pos += 1
        if tok == self.eos:
            st.doc += 1     # EOS closes the document; the next token already belongs to the following one
        return logits

    # ---- utilities
    def forward_logits(self, ids) -> np.ndarray:
        st = self.new_state()
        return np.stack([self.step(int(t), st) for t in ids])

    def perplexity(self, ids) -> float:
        lg = self.forward_logits(ids[:-1]).astype(np.float64)
        lg -= lg.max(-1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        nll = -lp[np.arange(len(ids) - 1), np.asarray(ids[1:])]
        return float(np.exp(nll.mean()))

    def generate(self, ids, max_new=32, top_k=0, temperature=1.0, seed=0, stop_eos=True):
        rng = np.random.default_rng(seed)
        st = self.new_state()
        logits = None
        for t in ids:
            logits = self.step(int(t), st)
        out = list(ids)
        for _ in range(max_new):
            if st.pos >= self.max_seq:
                break
            if top_k > 0:
                lg = logits.astype(np.float64) / max(temperature, 1e-6)
                kth = np.sort(lg)[-top_k]
                lg = np.where(lg < kth, -np.inf, lg)
                p = np.exp(lg - lg.max())
                nxt = int(rng.choice(len(p), p=p / p.sum()))
            else:
                nxt = int(np.argmax(logits))
            out.append(nxt)
            if stop_eos and nxt == self.eos:
                break
            logits = self.step(nxt, st)
        return out


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="ethoric_original")
    ap.add_argument("--prompt", default="The climate change")
    ap.add_argument("--ids", help="comma-separated ids (instead of --prompt)")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=0, help="0 = greedy")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--wq", type=int, default=0, choices=[0, 4, 8], help="simulate int8/int4 weights")
    ap.add_argument("--block", type=int, default=0, help="0 = one scale per row; N = one per block of N")
    a = ap.parse_args()

    m = Ethoric(a.dir, wq=a.wq, block=a.block)
    tok = None
    if a.ids:
        ids = [int(t) for t in a.ids.split(",")]
    else:
        from tokenizers import Tokenizer       # pip install tokenizers
        tok = Tokenizer.from_file(os.path.join(a.dir, "tokenizer.json"))
        ids = tok.encode(a.prompt).ids
    out = m.generate(ids, a.max_new, a.top_k, a.temperature)
    print("ids:", out)
    if tok is not None:
        print(tok.decode(out))


if __name__ == "__main__":
    main()
