// Minimal DNS responder for a captive portal: every name resolves to the access point's own address.
// Pure functions (no sockets), so they can be unit-tested on the host; the socket loop lives in main/net_server.cpp.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace ethoric {

// Builds the reply to the DNS query `q` (n bytes) into `out` (capacity `cap`).
//  - A queries are answered with `ip` (TTL 60 s);
//  - any other type gets an empty NOERROR answer, so clients fall back to IPv4 instead of waiting.
// Returns the reply length, or 0 when the packet is not a plain single-question query (the caller drops it).
inline size_t dns_build_reply(const uint8_t* q, size_t n, const uint8_t ip[4], uint8_t* out, size_t cap) {
  if (n < 12 || n > 512) return 0;
  const uint16_t flags = uint16_t((q[2] << 8) | q[3]);
  const uint16_t qd = uint16_t((q[4] << 8) | q[5]);
  if ((flags & 0x8000) != 0 || ((flags >> 11) & 0xF) != 0 || qd != 1) return 0;   // response, non-standard opcode, !=1 question

  size_t i = 12;                                      // walk the QNAME labels
  while (i < n && q[i] != 0) {
    if ((q[i] & 0xC0) != 0) return 0;                 // compression pointers do not occur in queries
    i += 1 + q[i];
  }
  if (i + 5 > n) return 0;                            // terminating zero + QTYPE + QCLASS
  const size_t qend = i + 5;
  const uint16_t qtype = uint16_t((q[i + 1] << 8) | q[i + 2]);
  const uint16_t qclass = uint16_t((q[i + 3] << 8) | q[i + 4]);
  const bool answer = qtype == 1 && qclass == 1;
  const size_t len = qend + (answer ? 16 : 0);
  if (len > cap) return 0;

  memcpy(out, q, qend);                               // header + question
  out[2] = uint8_t(0x80 | (q[2] & 0x01));             // QR=1, opcode 0, RD copied
  out[3] = 0x80;                                      // RA=1, RCODE=0
  out[6] = 0; out[7] = answer ? 1 : 0;                // ANCOUNT
  out[8] = out[9] = out[10] = out[11] = 0;            // NSCOUNT, ARCOUNT
  if (answer) {
    uint8_t* a = out + qend;
    a[0] = 0xC0; a[1] = 0x0C;                         // name: pointer to the question
    a[2] = 0; a[3] = 1;                               // TYPE A
    a[4] = 0; a[5] = 1;                               // CLASS IN
    a[6] = a[7] = 0; a[8] = 0; a[9] = 60;             // TTL
    a[10] = 0; a[11] = 4;                             // RDLENGTH
    memcpy(a + 12, ip, 4);
  }
  return len;
}

}  // namespace ethoric
