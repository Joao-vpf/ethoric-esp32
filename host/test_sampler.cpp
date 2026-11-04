// Sampler unit test (g++ -std=c++17 -O2 -o test_sampler test_sampler.cpp && ./test_sampler)
#include <cstdio>
#include <vector>

#include "sampler.hpp"

static int fails = 0;
#define CHECK(cond, msg) do { if (cond) printf("[OK] %s\n", msg); else { printf("[FAIL] %s\n", msg); fails++; } } while (0)

int main() {
  const int V = 100;
  std::vector<float> lg(V);
  for (int i = 0; i < V; i++) lg[i] = -float(i) * 0.1f;   // token 0 is the largest, then 1, 2, ...
  lg[0] = 5.0f; lg[1] = 4.9f; lg[2] = 4.8f; lg[3] = 4.7f;

  { ethoric::Sampler s(V);
    CHECK(s.pick(lg.data(), nullptr, 0) == 0, "greedy picks the argmax"); }

  { ethoric::Sampler s(V);
    int hist[] = {0};
    s.rep_penalty = 1.2f;
    CHECK(s.pick(lg.data(), hist, 1) == 1, "repetition penalty demotes token 0 (5.0/1.2 = 4.17 < 4.9)");
    int hist2[] = {0, 0, 0, 0};   // repeated inside the window: counts only once
    CHECK(s.pick(lg.data(), hist2, 4) == 1, "a token repeated in the window is penalized only once"); }

  { ethoric::Sampler s(V, 64, 7);
    s.temp = 1.0f; s.top_k = 3;
    int cnt[V] = {0};
    for (int i = 0; i < 6000; i++) cnt[s.pick(lg.data(), nullptr, 0)]++;
    int outside = 0;
    for (int i = 3; i < V; i++) outside += cnt[i];
    CHECK(outside == 0, "top-k=3 never samples outside the 3 largest (0,1,2)");
    // expected probabilities: softmax of (5.0, 4.9, 4.8) = 0.3672, 0.3322, 0.3006
    const double f0 = cnt[0] / 6000.0, f1 = cnt[1] / 6000.0, f2 = cnt[2] / 6000.0;
    CHECK(f0 > 0.33 && f0 < 0.41 && f1 > 0.30 && f1 < 0.37 && f2 > 0.27 && f2 < 0.34, "frequencies follow the softmax of the 3 largest");
    printf("     frequencies: %.3f %.3f %.3f\n", f0, f1, f2); }

  { ethoric::Sampler a(V, 64, 42), b(V, 64, 42);
    a.temp = b.temp = 0.8f; a.top_k = b.top_k = 10;
    bool same = true;
    for (int i = 0; i < 200; i++) same &= a.pick(lg.data(), nullptr, 0) == b.pick(lg.data(), nullptr, 0);
    CHECK(same, "same seed, same sequence"); }

  printf(fails ? "\nSOME TESTS FAILED\n" : "\nALL TESTS PASSED\n");
  return fails ? 1 : 0;
}
