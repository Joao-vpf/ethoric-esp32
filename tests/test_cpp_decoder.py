#!/usr/bin/env python3
"""Tests the C++ decoder (components/ethoric/include/ethoric.hpp) against reference_decoder.py with simulated int8.

Uses the tiny random model from test_reference_decoder.py (GQA, window of 4 so the KV ring wraps, Canon
with EOS in the middle). Exports it with tools/export_model.py, compiles host/host_main.cpp, runs it, and
compares logits, perplexity and greedy generation, plus kernels, threads, relocation, W8A8, int4, mixed
layers, the embedded vocabulary and file-integrity checks.

    python tests/test_cpp_decoder.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
EXPORTER = os.path.join(TOOLS, "export_model.py")
sys.path.insert(0, HERE)
sys.path.insert(0, TOOLS)
import reference_decoder as en  # noqa: E402
import test_reference_decoder as st  # noqa: E402
import export_model as ex  # noqa: E402


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        print(r.stdout, r.stderr)
        sys.exit(f"command failed: {' '.join(cmd)}")
    return r


def main():
    ok = True
    rng = np.random.default_rng(1)
    W = st.make_weights(rng, st.CFG)
    with tempfile.TemporaryDirectory() as td:
        json.dump(st.CFG, open(os.path.join(td, "config.json"), "w"))
        st.write_safetensors(os.path.join(td, "model.safetensors"), W)
        binf, exe = os.path.join(td, "m.bin"), os.path.join(td, "host")

        r = run([sys.executable, "-I", EXPORTER, "--dir", td, "--out", binf])
        print(r.stdout.strip().splitlines()[0])
        run(["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-pthread", "-I", os.path.join(ROOT, "components", "ethoric", "include"),
             "-o", exe, os.path.join(ROOT, "host", "host_main.cpp")])

        mq = en.Ethoric(td, wq=8, block=0)
        cases = {
            "no EOS (window ring wraps)": [5, 9, 3, 7, 2, 8, 11, 4, 6, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            "EOS in the middle (documents)": [5, 9, 3, 1, 7, 2, 8, 1, 11, 4, 6, 10, 1, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "EOS at the start": [1, 4, 5, 6, 1, 1, 7, 8, 9, 10, 11, 12],
        }
        for name, ids in cases.items():
            lout = os.path.join(td, "l.bin")
            run([exe, binf, "--ids", ",".join(map(str, ids)), "--logits-out", lout])
            got = np.fromfile(lout, dtype=np.float32).reshape(len(ids), -1)
            ref = mq.forward_logits(ids)
            err = float(np.abs(got - ref).max())
            agree = float((got.argmax(-1) == ref.argmax(-1)).mean())
            good = err < 2e-3 and agree == 1.0
            ok &= good
            print(f"[{'OK' if good else 'FAILED'}] {name}: max|dlogit| C++ vs numpy-int8 = {err:.2e}, argmax equal = {agree:.0%}")

        # perplexity
        ids = cases["no EOS (window ring wraps)"]
        r = run([exe, binf, "--ids", ",".join(map(str, ids)), "--ppl"])
        pc = float(r.stdout.split()[1])
        pn = mq.perplexity(ids)
        good = abs(pc / pn - 1) < 1e-4
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] perplexity: C++ = {pc:.4f}, numpy-int8 = {pn:.4f}")

        # greedy generation (both sides stop at EOS, so the token counts match)
        prompt = [5, 9, 3, 7, 2, 8]
        r = run([exe, binf, "--ids", ",".join(map(str, prompt)), "--new", "14"])
        out_c = [int(t) for t in r.stdout.split()[1:]]
        out_n = mq.generate(prompt, max_new=14, stop_eos=True)
        good = out_c == out_n
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] greedy generation: {'identical' if good else f'{out_c} vs {out_n}'}")

        # context smaller than the model's (the ESP32 uses this to save RAM)
        r = run([exe, binf, "--ids", ",".join(map(str, prompt)), "--new", "10", "--max-ctx", "16"])
        out_c16 = [int(t) for t in r.stdout.split()[1:]]
        good = out_c16 == out_n[:len(out_c16)]
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] --max-ctx 16 gives the same text ({len(out_c16)} ids)")

        # a truncated file must be rejected, not crash
        raw = open(binf, "rb").read()
        trunc = os.path.join(td, "trunc.bin")
        open(trunc, "wb").write(raw[:5000])
        r = subprocess.run([exe, trunc, "--ids", "1,2"], capture_output=True, text=True)
        good = r.returncode != 0 and "invalid" in r.stderr
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] truncated file is rejected")

        # CRC: an intact file passes; a flipped bit in the body is detected; so is a partial write (rest 0xFF)
        r = subprocess.run([exe, binf, "--verify"], capture_output=True, text=True)
        good = r.returncode == 0 and "crc32: ok" in r.stdout
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] crc32 of the intact file matches")
        bad = bytearray(raw)
        bad[len(bad) // 2] ^= 0x01
        open(os.path.join(td, "bad.bin"), "wb").write(bad)
        r = subprocess.run([exe, os.path.join(td, "bad.bin"), "--verify"], capture_output=True, text=True)
        good = r.returncode == 3 and "BAD" in r.stdout
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] a flipped bit in the body is detected")
        part = raw[: len(raw) * 2 // 3] + b"\xff" * (len(raw) - len(raw) * 2 // 3)   # erased flash after the cut
        open(os.path.join(td, "part.bin"), "wb").write(part)
        r = subprocess.run([exe, os.path.join(td, "part.bin"), "--verify"], capture_output=True, text=True)
        good = r.returncode == 3 and "BAD" in r.stdout
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] partial write (rest 0xFF) is detected")

        # the exporter's crc32 == zlib.crc32 of the body (header offset 68)
        import struct, zlib
        crc_hdr, total = struct.unpack_from("<IQ", raw, 68)
        good = crc_hdr == (zlib.crc32(raw[128:]) & 0xFFFFFFFF) and total == len(raw)
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] header: crc32 and total size ({total} bytes) are consistent")

        # ---- kernels / threads / relocation: same logits (threads and relocation: bit-exact)
        ids = cases["EOS in the middle (documents)"]
        outs = {}
        for tag, extra in {"ref": ["--kernel", "ref"], "u4": ["--kernel", "u4"], "r4": ["--kernel", "r4"],
                           "r8": ["--kernel", "r8"], "r4+thr": ["--kernel", "r4", "--threads", "2"],
                           "r4+reloc": ["--kernel", "r4", "--reloc-kb", "8"],
                           "r4+reloc+thr": ["--kernel", "r4", "--reloc-kb", "8", "--threads", "2"],
                           "r4+reloc-all+thr": ["--kernel", "r4", "--reloc-kb", "9999", "--threads", "2"]}.items():
            lo = os.path.join(td, f"k_{tag}.bin")
            run([exe, binf, "--ids", ",".join(map(str, ids)), "--logits-out", lo] + extra)
            outs[tag] = np.fromfile(lo, dtype=np.float32)
        for tag in ("u4", "r4", "r8"):
            e = float(np.abs(outs[tag] - outs["ref"]).max())
            good = e < 1e-3
            ok &= good
            print(f"[{'OK' if good else 'FAILED'}] kernel {tag} vs ref: max|dlogit| = {e:.1e}")
        for tag in ("r4+thr", "r4+reloc", "r4+reloc+thr", "r4+reloc-all+thr"):
            good = np.array_equal(outs[tag], outs["r4"])
            ok &= good
            print(f"[{'OK' if good else 'FAILED'}] {tag} bit-identical to r4")

        # ---- int8 activations (W8A8, groups of 32), format v2 with padded rows (ffn=23 -> ld=32)
        for tag, extra in {"aq": [], "aq+thr": ["--threads", "2"], "aq+reloc": ["--reloc-kb", "8"],
                           "aq+reloc+thr": ["--reloc-kb", "8", "--threads", "2"]}.items():
            lo = os.path.join(td, f"a_{tag}.bin")
            run([exe, binf, "--ids", ",".join(map(str, ids)), "--logits-out", lo, "--aq", "32"] + extra)
            outs[tag] = np.fromfile(lo, dtype=np.float32)
        lo = os.path.join(td, "a_emul.bin")
        r = run([exe, binf, "--ids", ",".join(map(str, ids)), "--logits-out", lo, "--aq", "32", "--pie-emul"])
        outs["emul"] = np.fromfile(lo, dtype=np.float32)
        good = np.array_equal(outs["emul"], outs["aq"]) and "selfcheck pie-emul: gate 1 down 1 embed 1" in r.stdout
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] SIMD kernel logic (emulated in C) identical to mv_a8, also with padded ld")
        e = float(np.abs(outs["aq"] - outs["r4"]).max()) / float(outs["r4"].std())
        good = e < 0.25
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] W8A8 (aq 32) vs float activations: max|dlogit|/std = {e:.3f}")
        for tag in ("aq+thr", "aq+reloc", "aq+reloc+thr"):
            good = np.array_equal(outs[tag], outs["aq"])
            ok &= good
            print(f"[{'OK' if good else 'FAILED'}] {tag} bit-identical to aq")

        # ---- int4 MLP (format v3): export --mlp-bits 4; C++ against numpy using the SAME dequantized weights
        bin4 = os.path.join(td, "m4.bin")
        run([sys.executable, "-I", EXPORTER, "--dir", td, "--out", bin4, "--mlp-bits", "4", "--no-vocab"])
        W4 = dict(W)
        for i in range(st.CFG["n_layer"]):
            for n in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{i}.mlp.{n}.weight"
                qg, sb = ex.quant_int4_groups(W[key])
                W4[key] = ex.dequant_int4(qg, sb, W[key].shape[1]).astype(np.float32)
        td4 = os.path.join(td, "d4"); os.makedirs(td4)
        json.dump(st.CFG, open(os.path.join(td4, "config.json"), "w"))
        st.write_safetensors(os.path.join(td4, "model.safetensors"), W4)
        mq4 = en.Ethoric(td4, wq=8, block=0)
        ref4 = mq4.forward_logits(ids)
        for tag, extra in {"int4": [], "int4+thr": ["--threads", "2"], "int4+reloc": ["--reloc-kb", "8"],
                           "int4+emul": ["--pie-emul"]}.items():
            lo = os.path.join(td, f"i4_{tag}.bin")
            r = run([exe, bin4, "--ids", ",".join(map(str, ids)), "--logits-out", lo] + extra)
            outs[tag] = np.fromfile(lo, dtype=np.float32)
            if tag == "int4+emul":
                good = "selfcheck pie-emul int4: gate 1 up 1 down 1" in r.stdout
                ok &= good
                print(f"[{'OK' if good else 'FAILED'}] int4: SIMD kernel (emulated) == C reference, bit-exact, on all 3 matrices")
        got = outs["int4"].reshape(len(ids), -1)
        e = float(np.abs(got - ref4).max()) / float(ref4.std())
        good = e < 0.25
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] int4 (C++ W4A8) vs numpy with the same dequantized weights: max|dlogit|/std = {e:.3f}")
        for tag in ("int4+thr", "int4+reloc", "int4+emul"):
            good = np.array_equal(outs[tag], outs["int4"])
            ok &= good
            print(f"[{'OK' if good else 'FAILED'}] {tag} bit-identical to int4")

        # mixed layers: only layer 1 in int4 (the others int8), same dequantized weights
        binm = os.path.join(td, "m4mix.bin")
        run([sys.executable, "-I", EXPORTER, "--dir", td, "--out", binm, "--mlp-bits", "4", "--mlp4-layers", "1", "--no-vocab"])
        Wm = dict(W)
        for n in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.1.mlp.{n}.weight"
            qg, sb = ex.quant_int4_groups(W[key])
            Wm[key] = ex.dequant_int4(qg, sb, W[key].shape[1]).astype(np.float32)
        tdm = os.path.join(td, "dm"); os.makedirs(tdm)
        json.dump(st.CFG, open(os.path.join(tdm, "config.json"), "w"))
        st.write_safetensors(os.path.join(tdm, "model.safetensors"), Wm)
        refm = en.Ethoric(tdm, wq=8, block=0).forward_logits(ids)
        lo = os.path.join(td, "mix.bin")
        run([exe, binm, "--ids", ",".join(map(str, ids)), "--logits-out", lo])
        gotm = np.fromfile(lo, dtype=np.float32).reshape(len(ids), -1)
        e = float(np.abs(gotm - refm).max()) / float(refm.std())
        good = e < 0.25
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] mixed layers (only layer 1 in int4) vs numpy: max|dlogit|/std = {e:.3f}")

        # ---- embedded vocabulary (synthetic tokenizer.json, ByteLevel decoder)
        b2u = ex.bytes_to_unicode()
        enc = lambda bs: "".join(b2u[b] for b in bs)
        V = st.CFG["vocab_size"]
        vocab = {"<pad>": 0, "<eos>": 1}
        words = [b"hello", b" world", b" the", "\u00e9".encode()[:1], "\u00e9".encode()[1:], b"\n", b".", " \u20ac".encode()]
        for i in range(2, V):
            vocab[enc(words[i % len(words)]) + ("" if i < 2 + len(words) else str(i))] = i
        tj = {"model": {"type": "BPE", "vocab": vocab, "merges": []},
              "added_tokens": [{"id": 0, "content": "<pad>", "special": True}, {"id": 1, "content": "<eos>", "special": True}],
              "decoder": {"type": "ByteLevel"}}
        json.dump(tj, open(os.path.join(td, "tokenizer.json"), "w"))
        binv = os.path.join(td, "mv.bin")
        run([sys.executable, "-I", EXPORTER, "--dir", td, "--out", binv])
        rb = subprocess.run([exe, binv, "--ids", "5,9,3,7", "--new", "12", "--text", "--verify"], capture_output=True)
        assert rb.returncode == 0
        r = type("R", (), {"stdout": rb.stdout.decode("utf-8", errors="replace")})
        outl = r.stdout.split("\n")
        gen = [int(t) for t in [l for l in outl if l.startswith("ids:")][0].split()[1:]][4:]
        txt = r.stdout[r.stdout.index("text: ") + 6:]
        txt = txt[:-1] if txt.endswith("\n") else txt
        table, _ = ex.build_vocab_table(tj, V)
        want = b"".join(table[i] for i in gen).decode("utf-8", errors="replace")
        good = txt == want and "crc32: ok" in r.stdout
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] vocabulary in the .bin: C++ text == exporter table ({want!r})")
        good = table[0] == b"" and table[1] == b"" and table[8] == b"hello" and table[9] == b" world" and \
            table[3] + table[4] == "\u00e9".encode() and table[7] == " \u20ac".encode()
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] ByteLevel table: empty specials, UTF-8 bytes split over 2 tokens join back")
        # Metaspace + ByteFallback (sentencepiece style)
        tj2 = {"model": {"vocab": {"<unk>": 0, "\u2581hi": 1, "<0x0A>": 2, "a": 3}},
               "added_tokens": [{"id": 0, "content": "<unk>", "special": True}],
               "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
                                                           {"type": "ByteFallback"}]}}
        t2, _ = ex.build_vocab_table(tj2, 4)
        good = t2 == [b"", b" hi", b"\n", b"a"]
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] Replace+ByteFallback table (sentencepiece)")
        # an old file (without vocabulary) still loads
        run([sys.executable, "-I", EXPORTER, "--dir", td, "--out", binv, "--no-vocab"])
        r = subprocess.run([exe, binv, "--ids", "5,9", "--new", "3", "--text"], capture_output=True, text=True)
        good = r.returncode == 0 and "text:" not in r.stdout and "no embedded vocabulary" in r.stderr
        ok &= good
        print(f"[{'OK' if good else 'FAILED'}] .bin without vocabulary still loads")

    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
