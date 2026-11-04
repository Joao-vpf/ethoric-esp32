// Next-token selection: greedy, or top-k with temperature and a repetition penalty.
// Greedy decoding of Ethoric-10M falls into loops ("The lake was 1.5 miles long. The lake was ...");
// the repetition penalty and top-k break them. Header-only, no STL.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace ethoric {

class Sampler {
 public:
  float temp = 0.f;          // <= 0: greedy (argmax)
  int top_k = 0;             // 0 or 1: greedy
  float rep_penalty = 1.f;   // > 1 penalizes tokens already seen (CTRL style): logit > 0 -> / p, logit < 0 -> * p
  int rep_window = 64;       // only the last N tokens are considered

  Sampler(int vocab, int max_top_k = 64, uint32_t seed = 1) : vocab_(vocab), max_k_(max_top_k), rng_(seed ? seed : 1) {
    scratch_ = (float*)malloc(sizeof(float) * vocab);
    ti_ = (int*)malloc(sizeof(int) * max_top_k);
    tv_ = (float*)malloc(sizeof(float) * max_top_k);
  }
  ~Sampler() { free(scratch_); free(ti_); free(tv_); }
  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;

  // hist: ids already emitted/processed (oldest to newest); n_hist: how many.
  int pick(const float* logits, const int* hist, int n_hist) {
    for (int i = 0; i < vocab_; i++) scratch_[i] = logits[i];
    if (rep_penalty > 1.0f) {
      const int from = n_hist > rep_window ? n_hist - rep_window : 0;
      for (int i = from; i < n_hist; i++) {
        float& s = scratch_[hist[i]];       // repeated inside the window: applying it twice would be wrong, see the mark below
        if (s != s) continue;
        if (!seen_mark(hist, from, i)) {    // first occurrence inside the window
          s = s > 0 ? s / rep_penalty : s * rep_penalty;
        }
      }
    }
    int k = top_k > max_k_ ? max_k_ : top_k;
    if (temp <= 0.f || k <= 1) {
      int best = 0;
      for (int i = 1; i < vocab_; i++) if (scratch_[i] > scratch_[best]) best = i;
      return best;
    }
    int n = 0;                                // top-k by ordered insertion (descending)
    for (int i = 0; i < vocab_; i++) {
      const float v = scratch_[i];
      if (n == k && v <= tv_[n - 1]) continue;
      int j = n < k ? n++ : n - 1;
      while (j > 0 && tv_[j - 1] < v) { tv_[j] = tv_[j - 1]; ti_[j] = ti_[j - 1]; j--; }
      tv_[j] = v;
      ti_[j] = i;
    }
    const float vmax = tv_[0];                // keep it: tv_[0] is overwritten in the loop below
    float sum = 0.f;
    for (int i = 0; i < n; i++) { tv_[i] = expf((tv_[i] - vmax) / temp); sum += tv_[i]; }
    float r = next_float() * sum;
    for (int i = 0; i < n; i++) {
      r -= tv_[i];
      if (r <= 0.f) return ti_[i];
    }
    return ti_[n - 1];
  }

 private:
  static bool seen_mark(const int* h, int from, int i) {   // has h[i] already appeared in [from, i)?
    for (int j = from; j < i; j++) if (h[j] == h[i]) return true;
    return false;
  }
  float next_float() {                        // xorshift32 -> [0, 1)
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 17;
    rng_ ^= rng_ << 5;
    return float(rng_ >> 8) * (1.0f / 16777216.0f);
  }
  int vocab_, max_k_;
  uint32_t rng_;
  float* scratch_;
  int* ti_;
  float* tv_;
};

}  // namespace ethoric
