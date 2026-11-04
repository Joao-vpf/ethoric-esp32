"""Independent reference encoder for Ethoric's tokenizer.json (ASCII text only).

It translates the pre-tokenizer regexes into Python `re` syntax (Unicode classes reduced to their ASCII members) and runs
a textbook BPE over the merge list. The on-device C++ encoder (components/ethoric/include/bpe.hpp) is tested against it.
For the real thing use the `tokenizers` library; `export_model.py --check` compares against it when installed.
"""
import re

L, LU, LL, M = "[A-Za-z]", "[A-Z]", "[a-z]", "[̀-ͯ]"
PS = r"[!-/:-@\[-`{-~]"
SPLIT1 = re.compile(
    rf" ?(?:{L}{M}*)+['’](?:{L}{M}*)+"
    rf"| ?(?:{LU}{M}*)+(?=(?:{LU}{M}*)(?:{LL}{M}*))"
    rf"| ?(?:{LU}{M}*)?(?:{LL}{M}*)+"
    rf"| ?(?:{LU}{M}*)+"
    rf"| ?(?:{PS}{M}*)+"
    rf"|[^\S\r\n]*[\n\r]+"
    rf"|[^\S\r\n]+")
SPLIT2 = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d|n't)")
SPLIT3 = re.compile(r"[0-9]{1,3}")


def isolate(pattern, pieces):
    """Split with behavior "Isolated": matches become their own pieces, text between them stays as pieces."""
    out = []
    for p in pieces:
        pos = 0
        for m in pattern.finditer(p):
            if m.start() > pos:
                out.append(p[pos:m.start()])
            out.append(m.group())
            pos = m.end()
        if pos < len(p):
            out.append(p[pos:])
    return out


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def pretokenize(text):
    return isolate(SPLIT3, isolate(SPLIT2, isolate(SPLIT1, [text])))


def encode(tj, text):
    if not text.isascii():
        raise ValueError("reference encoder handles ASCII only")
    vocab = tj["model"]["vocab"]
    ranks = {}
    for r, m in enumerate(tj["model"]["merges"]):
        pair = tuple(m) if isinstance(m, list) else tuple(m.split(" "))
        ranks.setdefault(pair, r)
    added = {t["content"]: t["id"] for t in tj["added_tokens"] if not t["special"]}
    b2u = bytes_to_unicode()
    ids = []
    for piece in pretokenize(text):
        if piece in added and piece.isdigit():
            ids.append(added[piece])
            continue
        sym = [b2u[b] for b in piece.encode("utf-8")]
        while len(sym) > 1:
            cand = [(ranks[(sym[i], sym[i + 1])], i) for i in range(len(sym) - 1) if (sym[i], sym[i + 1]) in ranks]
            if not cand:
                break
            _, i = min(cand)
            sym[i:i + 2] = [sym[i] + sym[i + 1]]
        ids.extend(vocab[s] for s in sym)
    return ids
