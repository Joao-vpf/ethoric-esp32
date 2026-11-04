#!/usr/bin/env python3
"""Compares reference_decoder.py against the original Ethoric-10M implementation from Hugging Face.

    python tools/check_against_hf.py --dir ethoric_original
    python tools/check_against_hf.py --dir ethoric_original --text-file sample.txt --wq 8
    python tools/check_against_hf.py --dir ethoric_original --wq 8 --block 0     # perplexity cost of int8

Checks: (1) tokenizer ids, (2) full-pass logits, (3) perplexity, (4) greedy generation.
With --wq N it also reports the perplexity cost of (simulated) weight quantization against HF fp32.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_decoder as en

DEFAULT_TEXT = (
    "The lighthouse stood at the edge of the cliff, and every night its beam swept across the dark water. "
    "The keeper climbed the stairs at dusk, lit the lamp, and watched the ships pass. "
    "Water is made of hydrogen and oxygen. Two plus two is four, and three times five is fifteen. "
    "The climate of a place depends on its distance from the sea, its height above it, and the winds that cross it."
)


def log_softmax(lg):
    lg = lg.astype(np.float64)
    lg = lg - lg.max(-1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(-1, keepdims=True))


def ppl_from_logits(lg, ids):
    lp = log_softmax(lg[:-1])
    return float(np.exp(-lp[np.arange(len(ids) - 1), np.asarray(ids[1:])].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="ethoric_original")
    ap.add_argument("--text-file")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--gen", type=int, default=40, help="tokens of greedy generation")
    ap.add_argument("--wq", type=int, default=0, choices=[0, 4, 8])
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3, help="fp32 logit tolerance")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text = open(a.text_file, encoding="utf-8").read() if a.text_file else DEFAULT_TEXT
    tok = AutoTokenizer.from_pretrained(a.dir)
    hf = AutoModelForCausalLM.from_pretrained(a.dir, trust_remote_code=True, torch_dtype=torch.float32).eval()
    ids = tok(text).input_ids[: a.max_tokens]
    print(f"{len(ids)} tokens; first: {ids[:12]}  eos={hf.config.eos_token_id}")
    ok = True

    # (1) tokenizer: the `tokenizers` library (used to build the device vocabulary) must give the same ids
    try:
        from tokenizers import Tokenizer
        t2 = Tokenizer.from_file(os.path.join(a.dir, "tokenizer.json")).encode(text).ids[: a.max_tokens]
        same = t2 == ids
        print(f"[{'OK' if same else 'DIFFER'}] tokenizer.json (lib tokenizers) vs AutoTokenizer: "
              f"{'identical ids' if same else f'{t2[:12]} vs {ids[:12]}'}")
        ok &= same
    except ImportError:
        print("[skipped] pip install tokenizers to compare the tokenizer")

    # HF: full pass, no cache
    with torch.no_grad():
        lg_hf = hf(torch.tensor([ids])).logits[0].float().numpy()

    # (2) logits fp32
    m = en.Ethoric(a.dir)
    t0 = time.time()
    lg = m.forward_logits(ids)
    dt = time.time() - t0
    err = float(np.abs(lg - lg_hf).max())
    agree = float((lg.argmax(-1) == lg_hf.argmax(-1)).mean())
    good = err < a.tol and agree > 0.99
    print(f"[{'OK' if good else 'FAILED'}] logits fp32: max|d| = {err:.2e}, mean|d| = {np.abs(lg - lg_hf).mean():.2e}, "
          f"argmax equal = {agree:.1%}  ({dt / len(ids) * 1e3:.1f} ms/token in numpy)")
    ok &= good

    # (3) perplexity
    p_hf, p_np = ppl_from_logits(lg_hf, ids), ppl_from_logits(lg, ids)
    good = abs(p_np / p_hf - 1) < 1e-3
    print(f"[{'OK' if good else 'FAILED'}] perplexity: HF = {p_hf:.4f}, numpy = {p_np:.4f}")
    ok &= good

    # (4) greedy generation (prompt = first 8 tokens)
    prompt = ids[:8]
    with torch.no_grad():
        out_hf = hf.generate(torch.tensor([prompt]), max_new_tokens=a.gen, do_sample=False,
                             eos_token_id=hf.config.eos_token_id)[0].tolist()
    out_np = m.generate(prompt, max_new=a.gen)
    n = min(len(out_hf), len(out_np))
    first = next((i for i in range(n) if out_hf[i] != out_np[i]), None)
    good = first is None and len(out_hf) == len(out_np)
    print(f"[{'OK' if good else 'DIFFER'}] greedy generation of {a.gen} tokens: "
          f"{'identical' if good else f'first difference at index {first}'}")
    print("   text:", tok.decode(out_np))
    ok &= good

    # (5) cost of simulated quantization
    if a.wq:
        mq = en.Ethoric(a.dir, wq=a.wq, block=a.block)
        lgq = mq.forward_logits(ids)
        pq = ppl_from_logits(lgq, ids)
        print(f"[info] int{a.wq} (block={a.block or 'row'}): perplexity {pq:.4f} vs fp32 {p_hf:.4f} "
              f"({pq / p_hf - 1:+.2%}); argmax equal to fp32 = {(lgq.argmax(-1) == lg_hf.argmax(-1)).mean():.1%}")

    print("\nALL OK" if ok else "\nDIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
