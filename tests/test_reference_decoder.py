#!/usr/bin/env python3
"""Self-test of reference_decoder.py without torch.

Builds a tiny random model (same tensor names as Ethoric-10M) and compares the incremental decoder
(KV cache + Canon state) with a LITERAL transcription of the Hugging Face code run as a full pass
(dense mask, padded convolution, document ids via cumsum) in float64.
Covers: GQA, local window, Canon with document masking (EOS in the middle), gate, QK-norm,
half-split RoPE, tied embedding, and an "odd" FFN size.

    python tests/test_reference_decoder.py
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import reference_decoder as en  # noqa: E402

CFG = dict(vocab_size=50, d_model=32, n_layer=3, n_head=4, n_kv_head=2, head_dim=8, ffn_hidden=23,
           max_seq_len=40, rope_theta=10000.0, norm_eps=1e-5, qk_norm="per_head", peri_ln=True,
           attn_gate="headwise", attn_window=4, attn_window_layers=[0, 2],
           canon={"enabled": True, "kernel": 4, "positions": ["A", "C"], "doc_masking": True},
           doc_masking=True, eos_token_id=1, pad_token_id=0, tie_word_embeddings=True)


def make_weights(rng, c):
    d, H, Hkv, hd, F = c["d_model"], c["n_head"], c["n_kv_head"], c["head_dim"], c["ffn_hidden"]
    n = lambda *s, sd=0.3: (rng.standard_normal(s) * sd).astype(np.float32)
    nw = lambda k: (1 + 0.2 * rng.standard_normal(k)).astype(np.float32)
    W = {"model.embed_tokens.weight": n(c["vocab_size"], d, sd=0.8), "model.final_norm.weight": nw(d)}
    for i in range(c["n_layer"]):
        p = f"model.layers.{i}."
        W[p + "attn.q_proj.weight"] = n(H * hd, d)
        W[p + "attn.k_proj.weight"] = n(Hkv * hd, d)
        W[p + "attn.v_proj.weight"] = n(Hkv * hd, d)
        W[p + "attn.o_proj.weight"] = n(d, H * hd)
        W[p + "attn.q_norm.weight"] = nw(hd)
        W[p + "attn.k_norm.weight"] = nw(hd)
        W[p + "attn.gate_proj.weight"] = n(H, d)
        W[p + "attn.gate_proj.bias"] = n(H, sd=1.0)
        for nm in ("attn_in_norm", "attn_out_norm", "mlp_in_norm", "mlp_out_norm"):
            W[p + nm + ".weight"] = nw(d)
        W[p + "canon_a.weight"] = n(d, 1, 4, sd=0.6)
        W[p + "canon_c.weight"] = n(d, 1, 4, sd=0.6)
        W[p + "mlp.gate_proj.weight"] = n(F, d)
        W[p + "mlp.up_proj.weight"] = n(F, d)
        W[p + "mlp.down_proj.weight"] = n(d, F)
    return W


def write_safetensors(path, W):
    header, off, blobs = {}, 0, []
    for k, v in W.items():
        b = np.ascontiguousarray(v, dtype=np.float32).tobytes()
        header[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(header).encode()
    h += b" " * ((-len(h)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


# ---- literal transcription of the HF code, float64, full pass ----------------------

def ref_forward(W, c, ids):
    f = lambda a: np.asarray(a, np.float64)
    T = len(ids)
    d, H, Hkv, hd, eps = c["d_model"], c["n_head"], c["n_kv_head"], c["head_dim"], c["norm_eps"]
    ids = np.asarray(ids)
    is_eos = (ids == c["eos_token_id"]).astype(np.int64)
    doc = np.cumsum(is_eos) - is_eos                     # document_ids()
    pos = np.arange(T)

    def rmsn(x, w):
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * f(w)

    half = hd // 2
    freqs = 1.0 / (c["rope_theta"] ** (np.arange(half) / half))
    ang = np.outer(np.arange(c["max_seq_len"]), freqs)
    cos, sin = np.cos(ang)[pos], np.sin(ang)[pos]       # [T, half]

    def apply_rope(x):                                   # x [H, T, hd]; half-split
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def canon(h, w):                                     # Canon.forward with doc_ids
        k = w.shape[-1]
        xt = np.pad(h.T, ((0, 0), (k - 1, 0)))           # [d, T+k-1]
        dpad = np.concatenate([np.full(k - 1, -1), doc])
        out = np.zeros((d, T))
        for i in range(k):
            seg = xt[:, i:i + T]
            keep = (dpad[i:i + T] == doc)[None, :]
            out += f(w)[:, 0, i][:, None] * seg * keep
        return out.T

    x = f(W["model.embed_tokens.weight"])[ids]
    for i in range(c["n_layer"]):
        p = f"model.layers.{i}."
        h = rmsn(x, W[p + "attn_in_norm.weight"])
        h = h + canon(h, W[p + "canon_a.weight"])
        q = (h @ f(W[p + "attn.q_proj.weight"]).T).reshape(T, H, hd).transpose(1, 0, 2)
        k = (h @ f(W[p + "attn.k_proj.weight"]).T).reshape(T, Hkv, hd).transpose(1, 0, 2)
        v = (h @ f(W[p + "attn.v_proj.weight"]).T).reshape(T, Hkv, hd).transpose(1, 0, 2)
        q, k = rmsn(q, W[p + "attn.q_norm.weight"]), rmsn(k, W[p + "attn.k_norm.weight"])
        q, k = apply_rope(q), apply_rope(k)
        k, v = np.repeat(k, H // Hkv, axis=0), np.repeat(v, H // Hkv, axis=0)   # repeat_interleave
        allowed = pos[None, :] <= pos[:, None]
        allowed &= doc[:, None] == doc[None, :]
        if i in c["attn_window_layers"]:
            allowed &= (pos[:, None] - pos[None, :]) < c["attn_window"]
        s = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)
        s = np.where(allowed[None], s, -np.inf)
        s = s - s.max(-1, keepdims=True)
        pr = np.exp(s)
        pr /= pr.sum(-1, keepdims=True)
        out = pr @ v                                                              # [H, T, hd]
        gate = 1 / (1 + np.exp(-(h @ f(W[p + "attn.gate_proj.weight"]).T + f(W[p + "attn.gate_proj.bias"]))))
        out = out * gate.T[:, :, None]
        out = out.transpose(1, 0, 2).reshape(T, H * hd)
        x = x + rmsn(out @ f(W[p + "attn.o_proj.weight"]).T, W[p + "attn_out_norm.weight"])
        h = rmsn(x, W[p + "mlp_in_norm.weight"])
        h = h + canon(h, W[p + "canon_c.weight"])
        g, u = h @ f(W[p + "mlp.gate_proj.weight"]).T, h @ f(W[p + "mlp.up_proj.weight"]).T
        m = (g / (1 + np.exp(-g)) * u) @ f(W[p + "mlp.down_proj.weight"]).T
        x = x + rmsn(m, W[p + "mlp_out_norm.weight"])
    x = rmsn(x, W["model.final_norm.weight"])
    return x @ f(W["model.embed_tokens.weight"]).T


def main():
    rng = np.random.default_rng(0)
    W = make_weights(rng, CFG)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        json.dump(CFG, open(os.path.join(td, "config.json"), "w"))
        write_safetensors(os.path.join(td, "model.safetensors"), W)
        m = en.Ethoric(td)

        for name, ids in [
            ("no EOS", [5, 9, 3, 7, 2, 8, 11, 4, 6, 10, 12, 13, 14, 15, 16, 17]),
            ("EOS in the middle (documents)", [5, 9, 3, 1, 7, 2, 8, 1, 11, 4, 6, 10, 1, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
            ("EOS at the start", [1, 4, 5, 6, 1, 1, 7, 8, 9, 10, 11, 12]),
        ]:
            ref = ref_forward(W, CFG, ids)
            got = m.forward_logits(ids)
            err = float(np.abs(ref - got).max())
            agree = float((ref.argmax(-1) == got.argmax(-1)).mean())
            status = "OK" if err < 1e-4 else "FAILED"
            ok &= err < 1e-4
            print(f"[{status}] {name}: max|dlogit| = {err:.2e}, argmax equal = {agree:.0%}")

        # counter-test: without the document mask the result MUST differ (proves the test is sensitive)
        ids = [5, 9, 3, 1, 7, 2, 8, 1, 11, 4, 6, 10, 12, 13]
        m2 = en.Ethoric(td)
        m2.doc_masking = m2.canon_doc_masking = False
        diff = float(np.abs(m2.forward_logits(ids) - ref_forward(W, CFG, ids)).max())
        print(f"[info] without the doc mask the error rises to {diff:.2e} (expected >> 1e-4)")
        ok &= diff > 1e-3

        # greedy: numpy must pick the full-pass argmax at every step
        ids = [5, 9, 3, 7, 2, 8]
        out = m.generate(ids, max_new=12, stop_eos=False)
        good = all(int(np.argmax(ref_forward(W, CFG, out[:j])[-1])) == out[j] for j in range(len(ids), len(out)))
        print(f"[{'OK' if good else 'FAILED'}] greedy generation = full-pass argmax ({len(out) - len(ids)} tokens)")
        ok &= good

        # simulated int8: small error in the logits
        mq = en.Ethoric(td, wq=8)
        ids = [5, 9, 3, 7, 2, 8, 11, 4, 6, 10, 12, 13, 14, 15, 16, 17]
        lg32 = m.forward_logits(ids)
        dq = float(np.abs(mq.forward_logits(ids) - lg32).max())
        rel = dq / float(lg32.std())
        print(f"[info] per-row int8: max|dlogit| vs fp32 = {dq:.2e} ({rel:.1%} of the logit standard deviation)")
        ok &= rel < 0.25     # only a sanity check that the simulation does not blow up; the real measure is perplexity on the real model

        # safetensors reader
        r = en.read_safetensors(os.path.join(td, "model.safetensors"))
        same = all(np.array_equal(r[k], W[k]) for k in W)
        print(f"[{'OK' if same else 'FAILED'}] safetensors reader returns the exact tensors")
        ok &= same
    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
