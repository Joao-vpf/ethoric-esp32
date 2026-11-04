#!/usr/bin/env python3
"""Tests the on-device BPE encoder (bpe.hpp) through host/host_main.cpp.

  * ASCII text: the C++ encoder must give the same ids as the independent reference (tools/tokenizer_ref.py).
  * Any text, including non-ASCII: concatenating the token bytes must reproduce the input exactly.

Uses a small synthetic tokenizer with the same pre-tokenizer pipeline as Ethoric's. Set ETHORIC_TOKENIZER to a real
tokenizer.json to run the same checks against it (the model is a tiny random one with a matching vocabulary size).

    python tests/test_tokenizer.py
"""
import collections
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import export_model as ex  # noqa: E402
import test_reference_decoder as st  # noqa: E402
import tokenizer_ref as tr  # noqa: E402

CORPUS = ("the lighthouse stood at the edge of the cliff and every night its beam swept across the dark water "
          "water is made of hydrogen and oxygen it is what we are they will not go there 2019 2020 ") * 4


def synthetic_tokenizer(n_merges=120):
    b2u = tr.bytes_to_unicode()
    vocab = {"<s>": 0, "</s>": 1, "<pad>": 2}
    for b in range(256):
        vocab[b2u[b]] = len(vocab)
    words = [tuple(b2u[b] for b in p.encode()) for p in tr.pretokenize(CORPUS)]
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for w in words:
            pairs.update(zip(w, w[1:]))
        if not pairs:
            break
        a, b = max(pairs, key=lambda k: (pairs[k], k))
        merges.append([a, b])
        vocab[a + b] = len(vocab)
        new = []
        for w in words:
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == (a, b):
                    out.append(a + b); i += 2
                else:
                    out.append(w[i]); i += 1
            new.append(tuple(out))
        words = new
    added = [{"id": i, "content": c, "special": True} for i, c in enumerate(("<s>", "</s>", "<pad>"))]
    for d in [str(x) for x in range(10)] + [f"{x:02d}" for x in range(100)] + [f"{x:03d}" for x in range(1000)]:
        added.append({"id": len(vocab) + len(added) - 3, "content": d, "special": False})
    return {"added_tokens": added, "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": ex.BPE_SPLIT1}, "behavior": "Isolated", "invert": False},
                {"type": "Split", "pattern": {"Regex": ex.BPE_SPLIT2}, "behavior": "Isolated", "invert": False},
                {"type": "Split", "pattern": {"Regex": ex.BPE_SPLIT3}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
            "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "dropout": None, "vocab": vocab, "merges": merges}}


def check(tj, label, exe, td):
    n_ids = max(max(tj["model"]["vocab"].values()), max(a["id"] for a in tj["added_tokens"])) + 1
    cfg = dict(st.CFG, vocab_size=n_ids)
    d = os.path.join(td, label)
    os.makedirs(d)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    st.write_safetensors(os.path.join(d, "model.safetensors"), st.make_weights(np.random.default_rng(1), cfg))
    json.dump(tj, open(os.path.join(d, "tokenizer.json"), "w"))
    binf = os.path.join(d, "m.bin")
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "export_model.py"), "--dir", d, "--out", binf],
                       capture_output=True, text=True)
    if r.returncode or "BPE tables:" not in r.stdout:
        print(r.stdout, r.stderr)
        return False

    def enc(t):
        o = subprocess.run([exe, binf, "--prompt", t, "--encode"], capture_output=True, text=True)
        return [int(x) for x in o.stdout.split(",")] if o.stdout.strip() else []

    ok = True
    rng = random.Random(7)
    alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789      \n\t.,;:!?'\"-()/_@#$%"
    texts = ["Don't stop 2019, HTTPServer ok!\n  x", "I'll   go  .\r\n\r\n 123456 7", "UPPER lower MiXed a'b it's WE'RE", ""]
    texts += ["".join(rng.choice(alpha) for _ in range(rng.randint(1, 80))) for _ in range(300)]
    bad = [t for t in texts if enc(t) != tr.encode(tj, t)]
    ok &= not bad
    print(f"[{'OK' if not bad else 'FAILED'}] {label}: C++ encoder == reference on {len(texts)} ASCII texts"
          + (f" (first mismatch {bad[0]!r})" if bad else ""))

    table, _ = ex.build_vocab_table(tj, n_ids)
    uni = ["café naïve résumé", "日本語のテキスト", "emoji \U0001F600 test",
           "x²y ٣٤ 12٣45", "é mixed", "tab\there  \n\n end.", "\x01ctl\x02 chars"] + texts[:50]
    bad = [t for t in uni if b"".join(table[i] for i in enc(t)).decode("utf-8", "replace") != t]
    ok &= not bad
    print(f"[{'OK' if not bad else 'FAILED'}] {label}: non-ASCII and control text round-trips through the vocabulary")
    return ok


def main():
    td = tempfile.mkdtemp()
    exe = os.path.join(td, "host")
    subprocess.run(["g++", "-O2", "-std=c++17", "-pthread", "-I", os.path.join(ROOT, "components", "ethoric", "include"),
                    "-o", exe, os.path.join(ROOT, "host", "host_main.cpp")], check=True)
    tj = synthetic_tokenizer()
    ok = ex.bpe_unsupported_reason(tj) is None
    print(f"[{'OK' if ok else 'FAILED'}] synthetic tokenizer accepted by the exporter")
    ok &= check(tj, "synthetic", exe, td)
    real = os.environ.get("ETHORIC_TOKENIZER")
    if real:
        ok &= check(json.load(open(real, encoding="utf-8")), "real", exe, td)
    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
