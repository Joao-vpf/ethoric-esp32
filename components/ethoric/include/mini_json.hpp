// Minimal JSON helpers for the HTTP front end: a parser for one flat top-level object and a string escaper.
// Nested objects/arrays are skipped, not stored. No exceptions, no dependencies besides the standard library.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

namespace ethoric {

class JsonObject {
 public:
  // Parses `s` as a JSON object. Returns false on any syntax error.
  bool parse(const char* s, size_t n) {
    items_.clear();
    p_ = s;
    end_ = s + n;
    ws();
    if (!eat('{')) return false;
    ws();
    if (eat('}')) return true;
    for (;;) {
      ws();
      Item it;
      if (!str(&it.key)) return false;
      ws();
      if (!eat(':')) return false;
      ws();
      if (p_ < end_ && *p_ == '"') {
        it.is_string = true;
        if (!str(&it.text)) return false;
      } else if (p_ < end_ && (*p_ == '-' || (*p_ >= '0' && *p_ <= '9'))) {
        char* e = nullptr;
        std::string tmp(p_, size_t(end_ - p_ > 40 ? 40 : end_ - p_));
        it.number = strtod(tmp.c_str(), &e);
        if (e == tmp.c_str()) return false;
        it.is_number = true;
        p_ += (e - tmp.c_str());
      } else if (lit("true")) {
        it.is_number = true; it.number = 1;
      } else if (lit("false")) {
        it.is_number = true; it.number = 0;
      } else if (lit("null")) {
        // stored as neither string nor number
      } else if (!skip_value()) {
        return false;
      }
      items_.push_back(it);
      ws();
      if (eat(',')) continue;
      return eat('}');
    }
  }

  bool get_string(const char* key, std::string* out) const {
    for (const Item& it : items_) if (it.is_string && it.key == key) { *out = it.text; return true; }
    return false;
  }
  bool get_number(const char* key, double* out) const {
    for (const Item& it : items_) if (it.is_number && it.key == key) { *out = it.number; return true; }
    return false;
  }

 private:
  struct Item {
    std::string key, text;
    double number = 0;
    bool is_string = false, is_number = false;
  };
  std::vector<Item> items_;
  const char* p_ = nullptr;
  const char* end_ = nullptr;

  void ws() { while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_; }
  bool eat(char c) { if (p_ < end_ && *p_ == c) { ++p_; return true; } return false; }
  bool lit(const char* w) {
    const size_t n = strlen(w);
    if (size_t(end_ - p_) >= n && memcmp(p_, w, n) == 0) { p_ += n; return true; }
    return false;
  }
  static void put_utf8(std::string* o, uint32_t c) {
    if (c < 0x80) { *o += char(c); }
    else if (c < 0x800) { *o += char(0xC0 | (c >> 6)); *o += char(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { *o += char(0xE0 | (c >> 12)); *o += char(0x80 | ((c >> 6) & 0x3F)); *o += char(0x80 | (c & 0x3F)); }
    else { *o += char(0xF0 | (c >> 18)); *o += char(0x80 | ((c >> 12) & 0x3F)); *o += char(0x80 | ((c >> 6) & 0x3F)); *o += char(0x80 | (c & 0x3F)); }
  }
  bool hex4(uint32_t* v) {
    if (end_ - p_ < 4) return false;
    uint32_t r = 0;
    for (int i = 0; i < 4; i++) {
      const char c = *p_++;
      r <<= 4;
      if (c >= '0' && c <= '9') r |= c - '0';
      else if (c >= 'a' && c <= 'f') r |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') r |= c - 'A' + 10;
      else return false;
    }
    *v = r;
    return true;
  }
  bool str(std::string* o) {
    if (!eat('"')) return false;
    o->clear();
    while (p_ < end_) {
      const char c = *p_++;
      if (c == '"') return true;
      if (c != '\\') { *o += c; continue; }
      if (p_ >= end_) return false;
      const char e = *p_++;
      switch (e) {
        case '"': *o += '"'; break;
        case '\\': *o += '\\'; break;
        case '/': *o += '/'; break;
        case 'b': *o += '\b'; break;
        case 'f': *o += '\f'; break;
        case 'n': *o += '\n'; break;
        case 'r': *o += '\r'; break;
        case 't': *o += '\t'; break;
        case 'u': {
          uint32_t v;
          if (!hex4(&v)) return false;
          if (v >= 0xD800 && v < 0xDC00) {   // surrogate pair
            uint32_t lo;
            if (end_ - p_ >= 2 && p_[0] == '\\' && p_[1] == 'u') {
              p_ += 2;
              if (!hex4(&lo) || lo < 0xDC00 || lo >= 0xE000) return false;
              v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
            } else {
              v = 0xFFFD;
            }
          } else if (v >= 0xDC00 && v < 0xE000) {
            v = 0xFFFD;
          }
          put_utf8(o, v);
          break;
        }
        default: return false;
      }
    }
    return false;
  }
  // Skips a nested value (object, array, string, number, literal).
  bool skip_value() {
    if (p_ >= end_) return false;
    if (*p_ == '"') { std::string t; return str(&t); }
    if (*p_ == '{' || *p_ == '[') {
      int depth = 0;
      while (p_ < end_) {
        const char c = *p_;
        if (c == '"') { std::string t; if (!str(&t)) return false; continue; }
        ++p_;
        if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') { if (--depth == 0) return true; }
      }
      return false;
    }
    return false;
  }
};

// Escapes `s` as the contents of a JSON string (without the quotes). Invalid UTF-8 becomes U+FFFD so the output is
// always valid JSON, even when generation stopped in the middle of a multi-byte character.
inline std::string json_escape(const char* s, size_t n) {
  static const char* hex = "0123456789abcdef";
  std::string o;
  o.reserve(n + 8);
  const uint8_t* b = reinterpret_cast<const uint8_t*>(s);
  size_t i = 0;
  while (i < n) {
    const uint8_t c = b[i];
    if (c < 0x80) {
      if (c == '"') o += "\\\"";
      else if (c == '\\') o += "\\\\";
      else if (c == '\n') o += "\\n";
      else if (c == '\r') o += "\\r";
      else if (c == '\t') o += "\\t";
      else if (c < 0x20) { o += "\\u00"; o += hex[c >> 4]; o += hex[c & 15]; }
      else o += char(c);
      ++i;
      continue;
    }
    size_t l = 0;
    if (c >= 0xC2 && c < 0xE0) l = 2;
    else if (c >= 0xE0 && c < 0xF0) l = 3;
    else if (c >= 0xF0 && c < 0xF5) l = 4;
    bool ok = l && i + l <= n;
    for (size_t k = 1; ok && k < l; k++) ok = (b[i + k] & 0xC0) == 0x80;
    if (ok && l == 3) {
      const uint32_t v = (uint32_t(c & 0x0F) << 12) | (uint32_t(b[i + 1] & 0x3F) << 6) | (b[i + 2] & 0x3F);
      ok = v >= 0x800 && !(v >= 0xD800 && v < 0xE000);
    }
    if (ok && l == 4) {
      const uint32_t v = (uint32_t(c & 0x07) << 18) | (uint32_t(b[i + 1] & 0x3F) << 12) |
                         (uint32_t(b[i + 2] & 0x3F) << 6) | (b[i + 3] & 0x3F);
      ok = v >= 0x10000 && v <= 0x10FFFF;
    }
    if (ok) { o.append(s + i, l); i += l; }
    else { o += "\xEF\xBF\xBD"; ++i; }
  }
  return o;
}

}  // namespace ethoric
