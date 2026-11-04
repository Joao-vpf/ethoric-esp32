// Unit tests for components/ethoric/include/mini_json.hpp.
#include <stdio.h>

#include "mini_json.hpp"

static int g_fail = 0;
#define CHECK(cond, name)                                  \
  do {                                                     \
    const bool ok_ = (cond);                               \
    printf("[%s] %s\n", ok_ ? "OK" : "FAIL", name);        \
    if (!ok_) ++g_fail;                                    \
  } while (0)

int main() {
  using ethoric::JsonObject;
  using ethoric::json_escape;
  {
    const char* j = R"({"prompt": "Hi \"there\"\n\u00e9 \ud83d\ude00", "n_predict": 32, "temperature": 0.7, "stream": false,
                        "extra": {"a": [1, 2, {"b": "}"}], "c": "]"}, "seed": -5, "nothing": null, "top_k": 20})";
    JsonObject o;
    std::string s;
    double v = 0;
    CHECK(o.parse(j, strlen(j)), "parses a typical request");
    CHECK(o.get_string("prompt", &s) && s == "Hi \"there\"\n\xC3\xA9 \xF0\x9F\x98\x80", "string escapes and surrogate pairs");
    CHECK(o.get_number("n_predict", &v) && v == 32, "integer");
    CHECK(o.get_number("temperature", &v) && v == 0.7, "float");
    CHECK(o.get_number("seed", &v) && v == -5, "negative number");
    CHECK(o.get_number("top_k", &v) && v == 20, "value after a skipped nested object");
    CHECK(o.get_number("stream", &v) && v == 0, "boolean");
    CHECK(!o.get_number("missing", &v) && !o.get_string("n_predict", &s), "missing key / wrong type");
  }
  {
    JsonObject o;
    const char* bad[] = {"", "{", "{\"a\"}", "{\"a\":}", "[1]", "{\"a\":\"x}", "{\"a\":1,}", "{\"a\":\"\\q\"}"};
    bool all = true;
    for (const char* b : bad) all &= !o.parse(b, strlen(b));
    CHECK(all, "rejects malformed input");
    CHECK(o.parse("{}", 2), "empty object");
  }
  {
    const std::string e = json_escape("a\"b\\c\n\t\x01 \xC3\xA9", 11);
    CHECK(e == "a\\\"b\\\\c\\n\\t\\u0001 \xC3\xA9", "escape quotes, controls, keeps valid UTF-8");
    const std::string t = json_escape("ok \xE2\x82", 5);   // truncated 3-byte sequence
    CHECK(t == "ok \xEF\xBF\xBD\xEF\xBF\xBD", "invalid/truncated UTF-8 becomes U+FFFD");
  }
  printf("\n%s\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
  return g_fail ? 1 : 0;
}
