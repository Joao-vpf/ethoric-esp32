// Unit tests for components/ethoric/include/dns_captive.hpp.
#include <stdio.h>

#include <vector>

#include "dns_captive.hpp"

static int g_fail = 0;
#define CHECK(cond, name)                           \
  do {                                              \
    const bool ok_ = (cond);                        \
    printf("[%s] %s\n", ok_ ? "OK" : "FAIL", name); \
    if (!ok_) ++g_fail;                             \
  } while (0)

static std::vector<uint8_t> query(const char* const* labels, int n, uint16_t qtype, uint16_t id = 0x1234) {
  std::vector<uint8_t> q = {uint8_t(id >> 8), uint8_t(id), 0x01, 0x00, 0, 1, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < n; i++) {
    q.push_back(uint8_t(strlen(labels[i])));
    q.insert(q.end(), labels[i], labels[i] + strlen(labels[i]));
  }
  q.push_back(0);
  q.push_back(uint8_t(qtype >> 8)); q.push_back(uint8_t(qtype));
  q.push_back(0); q.push_back(1);
  return q;
}

int main() {
  const uint8_t ip[4] = {192, 168, 4, 1};
  uint8_t out[512];
  const char* name[] = {"connectivitycheck", "gstatic", "com"};

  auto qa = query(name, 3, 1);
  size_t n = ethoric::dns_build_reply(qa.data(), qa.size(), ip, out, sizeof(out));
  CHECK(n == qa.size() + 16, "A query: reply = question + one 16-byte answer");
  CHECK(out[0] == 0x12 && out[1] == 0x34, "transaction id is echoed");
  CHECK((out[2] & 0x80) && out[2] & 0x01 && out[3] == 0x80, "QR, RD and RA flags");
  CHECK(out[7] == 1 && out[5] == 1, "ANCOUNT = 1, QDCOUNT = 1");
  CHECK(memcmp(out + n - 4, ip, 4) == 0, "answer is 192.168.4.1");

  auto q6 = query(name, 3, 28);
  n = ethoric::dns_build_reply(q6.data(), q6.size(), ip, out, sizeof(out));
  CHECK(n == q6.size() && out[7] == 0 && out[3] == 0x80, "AAAA query: empty NOERROR answer");

  CHECK(ethoric::dns_build_reply(qa.data(), 5, ip, out, sizeof(out)) == 0, "too short is dropped");
  std::vector<uint8_t> resp = qa; resp[2] |= 0x80;
  CHECK(ethoric::dns_build_reply(resp.data(), resp.size(), ip, out, sizeof(out)) == 0, "responses are ignored");
  std::vector<uint8_t> trunc(qa.begin(), qa.end() - 3);
  CHECK(ethoric::dns_build_reply(trunc.data(), trunc.size(), ip, out, sizeof(out)) == 0, "truncated question is dropped");
  std::vector<uint8_t> loop = qa; loop[12] = 0xC0;
  CHECK(ethoric::dns_build_reply(loop.data(), loop.size(), ip, out, sizeof(out)) == 0, "compression pointer in a query is dropped");
  CHECK(ethoric::dns_build_reply(qa.data(), qa.size(), ip, out, 20) == 0, "output buffer too small");

  printf("\n%s\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
  return g_fail ? 1 : 0;
}
