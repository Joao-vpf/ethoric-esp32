// Interface between the generation engine (main.cpp) and the network front end (net_server.cpp).
#pragma once
#include <stdint.h>

#include <string>

// Per-request generation settings. A negative value (seed: zero) means "use the firmware default".
struct GenParams {
  int max_new = -1;
  float temp = -1.f;
  int top_k = -1;
  float rep = -1.f;
  uint32_t seed = 0;
  bool echo = false;  // also emit the prompt text before the generated text
};

struct GenStats {
  int prompt_tokens = 0;
  double prompt_ms = 0;
  int tokens = 0;          // generated tokens
  double predicted_ms = 0;
  bool hit_eos = false;
};

// Tokenizes `prompt` on the device, generates a completion and appends its text to *out. Only one generation runs at
// a time. Returns 0 on success, otherwise an HTTP-style status (400 bad request, 503 busy, 500 internal error) and a
// message in *err.
int engine_complete(const std::string& prompt, const GenParams& gp, std::string* out, GenStats* st, std::string* err);

// Starts WiFi (station or access point, configured by .env) and the HTTP server. Implemented in net_server.cpp.
void net_start(int max_ctx);
