// On-device byte-level BPE encoder (text -> token ids) for Ethoric's tokenizer.
//
// The tables live in an optional section of the ETHW file (written by tools/export_model.py, see docs/MODEL_FORMAT.md):
//   u8  magic[4] = "BPE1"; u32 n_merges; u32 n_classes; u32 reserved;
//   u16 byte_id[256]       token id of each single byte (after the GPT-2 byte-to-unicode mapping)
//   u16 digit_id[1110]     ids of the added tokens "0".."9", "00".."99", "000".."999" (index: 1-digit d,
//                          2-digit 10+dd, 3-digit 110+ddd)
//   { u32 key; u16 rank; u16 result } merges[n_merges], sorted by key = (left_id << 16) | right_id
//   u32 classes[n_classes] sorted by start: (first_codepoint << 3) | class, covering every code point
// Class codes: 0 other, 1 Lu, 2 Ll, 3 other letter (Lt/Lm/Lo), 4 M, 5 P or S, 6 N, 7 White_Space.
//
// Pre-tokenization reproduces the tokenizer.json pipeline (checked by the exporter, which refuses to embed the tables
// for any other pipeline):
//   1. Split (Isolated) with the seven-alternative regex over letters, marks, punctuation/symbols and whitespace
//   2. Split (Isolated) on English contractions ('s 't 're 've 'm 'll 'd n't, ASCII, case-insensitive)
//   3. digits: runs of ASCII digits are cut into chunks of up to 3, each one a single added token; other Unicode
//      digits are cut into chunks of up to 3 and go through BPE
// Known limitation: the input is assumed to be NFC-normalized already, and special-token strings such as "</s>" in
// the text are treated as ordinary text.
#pragma once
#include <stdint.h>
#include <string.h>

#include <vector>

namespace ethoric {

class BpeTokenizer {
 public:
  bool init(const uint8_t* p, uint32_t n) {
    if (!p || n < kHeader || memcmp(p, "BPE1", 4) != 0) return false;
    n_merges_ = rd32(p + 4);
    n_cls_ = rd32(p + 8);
    const uint64_t need = uint64_t(kHeader) + 8ull * n_merges_ + 4ull * n_cls_;
    if (need > n || n_cls_ == 0) return false;
    byte_id_ = p + 16;
    digit_id_ = p + 16 + 512;
    merges_ = p + kHeader;
    cls_ = merges_ + 8ull * n_merges_;
    return true;
  }
  bool ready() const { return merges_ != nullptr; }

  // Encodes UTF-8 `s` into ids. Returns the number of ids, or -1 if `cap` is too small.
  int encode(const char* s, size_t len, int* out, int cap) const {
    std::vector<uint32_t> cp;      // code points (0xFFFFFFFF for an invalid byte)
    std::vector<uint32_t> off;     // byte offset of each code point (+ the end)
    std::vector<uint8_t> cl;       // class of each code point
    cp.reserve(len);
    off.reserve(len + 1);
    cl.reserve(len);
    decode_utf8(reinterpret_cast<const uint8_t*>(s), len, &cp, &off, &cl);
    Ctx cx{cp.data(), off.data(), cl.data(), int(cp.size()), reinterpret_cast<const uint8_t*>(s), out, cap, 0, false};

    int gap = -1;
    for (int i = 0; i < cx.n;) {
      const int m = match_pretoken(cx, i);
      if (m > 0) {
        if (gap >= 0) { split_contractions(cx, gap, i); gap = -1; }
        split_contractions(cx, i, i + m);
        i += m;
      } else {
        if (gap < 0) gap = i;
        ++i;
      }
    }
    if (gap >= 0) split_contractions(cx, gap, cx.n);
    return cx.overflow ? -1 : cx.count;
  }

 private:
  static const uint32_t kHeader = 16 + 512 + 2220;
  const uint8_t* byte_id_ = nullptr;
  const uint8_t* digit_id_ = nullptr;
  const uint8_t* merges_ = nullptr;
  const uint8_t* cls_ = nullptr;
  uint32_t n_merges_ = 0, n_cls_ = 0;

  struct Ctx {
    const uint32_t* cp;
    const uint32_t* off;
    const uint8_t* cl;
    int n;
    const uint8_t* text;
    int* out;
    int cap;
    int count;
    bool overflow;
  };

  static uint32_t rd32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
  static uint16_t rd16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }

  // ---- code point classes -----------------------------------------------------------------------------------------
  uint8_t classify(uint32_t c) const {
    if (c == 0xFFFFFFFFu) return 0;
    uint32_t lo = 0, hi = n_cls_;   // last entry whose start <= c
    while (hi - lo > 1) {
      const uint32_t mid = (lo + hi) / 2;
      if ((rd32(cls_ + 4ull * mid) >> 3) <= c) lo = mid; else hi = mid;
    }
    return uint8_t(rd32(cls_ + 4ull * lo) & 7u);
  }

  void decode_utf8(const uint8_t* s, size_t n, std::vector<uint32_t>* cp, std::vector<uint32_t>* off,
                   std::vector<uint8_t>* cl) const {
    size_t i = 0;
    while (i < n) {
      const uint8_t b = s[i];
      uint32_t c = 0xFFFFFFFFu;
      size_t l = 1;
      if (b < 0x80) { c = b; }
      else if (b >= 0xC2 && b < 0xE0 && i + 1 < n && (s[i + 1] & 0xC0) == 0x80) {
        c = (uint32_t(b & 0x1F) << 6) | (s[i + 1] & 0x3F); l = 2;
      } else if (b >= 0xE0 && b < 0xF0 && i + 2 < n && (s[i + 1] & 0xC0) == 0x80 && (s[i + 2] & 0xC0) == 0x80) {
        c = (uint32_t(b & 0x0F) << 12) | (uint32_t(s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F); l = 3;
        if (c < 0x800 || (c >= 0xD800 && c < 0xE000)) { c = 0xFFFFFFFFu; l = 1; }
      } else if (b >= 0xF0 && b < 0xF5 && i + 3 < n && (s[i + 1] & 0xC0) == 0x80 && (s[i + 2] & 0xC0) == 0x80 &&
                 (s[i + 3] & 0xC0) == 0x80) {
        c = (uint32_t(b & 0x07) << 18) | (uint32_t(s[i + 1] & 0x3F) << 12) | (uint32_t(s[i + 2] & 0x3F) << 6) |
            (s[i + 3] & 0x3F);
        l = 4;
        if (c < 0x10000 || c > 0x10FFFF) { c = 0xFFFFFFFFu; l = 1; }
      }
      cp->push_back(c);
      off->push_back(uint32_t(i));
      cl->push_back(classify(c));
      i += l;
    }
    off->push_back(uint32_t(n));
  }

  // ---- step 1: the pre-tokenizer regex ----------------------------------------------------------------------------
  static bool is_letter(uint8_t k) { return k >= 1 && k <= 3; }
  static bool hws(const Ctx& x, int i) { return x.cl[i] == 7 && x.cp[i] != '\n' && x.cp[i] != '\r'; }

  // (?:\p{L}\p{M}*)+ starting at i: returns the end, or i if there is no letter
  static int letter_run(const Ctx& x, int i) {
    int j = i;
    while (j < x.n && is_letter(x.cl[j])) {
      ++j;
      while (j < x.n && x.cl[j] == 4) ++j;
    }
    return j;
  }

  // Length of the match starting at i, 0 if no alternative matches.
  static int match_pretoken(const Ctx& x, int i) {
    const int n = x.n;
    const int p = (x.cp[i] == ' ') ? i + 1 : i;   // optional leading space (the first alternatives)
    if (p < n) {
      // 1. letters ['’] letters
      int j = letter_run(x, p);
      if (j > p && j < n && (x.cp[j] == '\'' || x.cp[j] == 0x2019)) {
        const int k = letter_run(x, j + 1);
        if (k > j + 1) return k - i;
      }
      // 2. uppercase run followed by (Lu Ll): everything except the last uppercase unit
      if (x.cl[p] == 1) {
        int ends_prev = p, e = p;   // e = end of the whole uppercase run; ends_prev = end before its last unit
        int units = 0;
        while (e < n && x.cl[e] == 1) {
          ends_prev = e;
          ++e;
          while (e < n && x.cl[e] == 4) ++e;
          ++units;
        }
        if (units >= 2 && e < n && x.cl[e] == 2) return ends_prev - i;
      }
      // 3. optional uppercase unit followed by lowercase units
      {
        int q = p;
        if (x.cl[q] == 1) {
          ++q;
          while (q < n && x.cl[q] == 4) ++q;
        }
        if (q < n && x.cl[q] == 2) {
          while (q < n && x.cl[q] == 2) {
            ++q;
            while (q < n && x.cl[q] == 4) ++q;
          }
          return q - i;
        }
      }
      // 4. uppercase run
      if (x.cl[p] == 1) {
        int q = p;
        while (q < n && x.cl[q] == 1) {
          ++q;
          while (q < n && x.cl[q] == 4) ++q;
        }
        return q - i;
      }
      // 5. punctuation / symbols
      if (x.cl[p] == 5) {
        int q = p;
        while (q < n && x.cl[q] == 5) {
          ++q;
          while (q < n && x.cl[q] == 4) ++q;
        }
        return q - i;
      }
    }
    // 6./7. horizontal whitespace followed by newlines, or just horizontal whitespace
    int j = i;
    while (j < n && hws(x, j)) ++j;
    if (j < n && (x.cp[j] == '\n' || x.cp[j] == '\r')) {
      while (j < n && (x.cp[j] == '\n' || x.cp[j] == '\r')) ++j;
      return j - i;
    }
    return j - i;
  }

  // ---- step 2: contractions ---------------------------------------------------------------------------------------
  static uint32_t lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

  static int contraction_len(const Ctx& x, int i, int end) {
    const uint32_t a = x.cp[i];
    const uint32_t b = i + 1 < end ? lower(x.cp[i + 1]) : 0;
    const uint32_t c = i + 2 < end ? lower(x.cp[i + 2]) : 0;
    if (a == '\'') {
      if (b == 's' || b == 't' || b == 'm' || b == 'd') return 2;
      if ((b == 'r' && c == 'e') || (b == 'v' && c == 'e') || (b == 'l' && c == 'l')) return 3;
    } else if (lower(a) == 'n' && b == '\'' && c == 't') {
      return 3;
    }
    return 0;
  }

  void split_contractions(Ctx& x, int a, int b) const {
    int seg = a;
    for (int i = a; i < b;) {
      const int l = contraction_len(x, i, b);
      if (l) {
        split_digits(x, seg, i);
        split_digits(x, i, i + l);
        i += l;
        seg = i;
      } else {
        ++i;
      }
    }
    split_digits(x, seg, b);
  }

  // ---- step 3: digits ---------------------------------------------------------------------------------------------
  void emit(Ctx& x, int id) const {
    if (x.count >= x.cap) { x.overflow = true; return; }
    x.out[x.count++] = id;
  }

  void split_digits(Ctx& x, int a, int b) const {
    int seg = a;
    int j = a;
    while (j < b) {
      if (x.cl[j] != 6) { ++j; continue; }
      bps(x, seg, j);
      const bool ascii = x.cp[j] < 128;
      int k = j, cnt = 0, val = 0;
      while (k < b && cnt < 3 && x.cl[k] == 6 && (x.cp[k] < 128) == ascii) {
        if (ascii) val = val * 10 + int(x.cp[k] - '0');
        ++k;
        ++cnt;
      }
      if (ascii) {
        const int idx = cnt == 1 ? val : cnt == 2 ? 10 + val : 110 + val;
        emit(x, rd16(digit_id_ + 2 * idx));
      } else {
        bps(x, j, k);
      }
      j = seg = k;
    }
    bps(x, seg, b);
  }

  // ---- BPE on one pre-token ---------------------------------------------------------------------------------------
  bool lookup(uint32_t key, uint16_t* rank, uint16_t* res) const {
    uint32_t lo = 0, hi = n_merges_;
    while (lo < hi) {
      const uint32_t mid = (lo + hi) / 2;
      const uint32_t k = rd32(merges_ + 8ull * mid);
      if (k == key) {
        *rank = rd16(merges_ + 8ull * mid + 4);
        *res = rd16(merges_ + 8ull * mid + 6);
        return true;
      }
      if (k < key) lo = mid + 1; else hi = mid;
    }
    return false;
  }

  void bps(Ctx& x, int a, int b) const {
    if (a >= b) return;
    std::vector<uint16_t> sym;
    for (uint32_t i = x.off[a]; i < x.off[b]; ++i) sym.push_back(rd16(byte_id_ + 2 * x.text[i]));
    for (;;) {
      uint32_t best = 0xFFFFFFFFu;
      size_t at = 0;
      uint16_t bres = 0;
      for (size_t i = 0; i + 1 < sym.size(); ++i) {
        uint16_t rank, res;
        if (lookup((uint32_t(sym[i]) << 16) | sym[i + 1], &rank, &res) && rank < best) {
          best = rank; at = i; bres = res;
        }
      }
      if (best == 0xFFFFFFFFu) break;
      sym[at] = bres;
      sym.erase(sym.begin() + at + 1);
    }
    for (uint16_t s : sym) emit(x, s);
  }
};

}  // namespace ethoric
