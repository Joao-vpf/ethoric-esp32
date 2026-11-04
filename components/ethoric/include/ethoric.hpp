// Ethoric-10M decoder in C++17: header-only, no STL, no ESP-IDF dependency.
// Reads the file produced by tools/export_model.py. Weights are NOT copied by default: Model keeps pointers
// into the blob (flash mapped with esp_partition_mmap, or a plain buffer on the host); `relocate` can copy part
// of them to faster memory (e.g. PSRAM on the ESP32-S3).
// Per-row int8 weights (optionally per-group int4 for the MLP); float or int8 activations; one token at a time (KV cache + Canon state).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "bpe.hpp"

namespace ethoric {

struct Config {
  int d, n_layer, n_head, n_kv, hd, ffn, vocab, max_seq, kernel, window, eos;
  bool doc_masking, canon_doc_masking;
  uint32_t window_mask;
  float theta, eps;
};

struct Mat {            // bits=8: int8 [rows][ld] (ld >= cols, extra columns are 0), one fp32 scale per row (s)
  const float* s;
  const int8_t* q;
  int ld;
  // bits=4 (MLP matrices only, format v3): q holds [rows][ld/2] bytes and gs holds bf16 scales per group of 32 columns,
  // [rows][ld/32] (s is unused). A 32-column group occupies 16 bytes: byte j stores column j in the LOW nibble
  // (value + 8, 0..15) and column j + 16 in the HIGH nibble (two's complement, -8..7).
  const uint16_t* gs;
  int bits;
};

struct Layer {
  const float *attn_in_norm, *attn_out_norm, *mlp_in_norm, *mlp_out_norm;
  const float *q_norm, *k_norm, *gate_w, *gate_b, *canon_a, *canon_c;
  Mat q, k, v, o, mlp_gate, mlp_up, mlp_down;
  bool windowed;
};

// ----------------------------------------------------------------------------- float-activation kernels: y = scale * (W int8) x

// y[r] = s[r] * sum_c q[r*cols + c] * x[c] for r in [0, rows). `q` points at the first row of the block.
typedef void (*MvKernel)(const int8_t* q, const float* s, const float* x, float* y, int rows, int cols);

// Reference: a single accumulator (floating-point dependency chain on every MAC).
static inline void mv_ref(const int8_t* q, const float* s, const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    const int8_t* w = q + size_t(r) * cols;
    float acc = 0.f;
    for (int c = 0; c < cols; c++) acc += float(w[c]) * x[c];
    y[r] = acc * s[r];
  }
}

// One row at a time, four independent accumulators (breaks the dependency chain).
static inline void mv_u4(const int8_t* q, const float* s, const float* x, float* y, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    const int8_t* w = q + size_t(r) * cols;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    int c = 0;
    for (; c + 4 <= cols; c += 4) {
      a0 += float(w[c]) * x[c];
      a1 += float(w[c + 1]) * x[c + 1];
      a2 += float(w[c + 2]) * x[c + 2];
      a3 += float(w[c + 3]) * x[c + 3];
    }
    for (; c < cols; c++) a0 += float(w[c]) * x[c];
    y[r] = ((a0 + a1) + (a2 + a3)) * s[r];
  }
}

// Four rows at a time: four independent accumulators and x is loaded once for all four rows.
static inline void mv_r4(const int8_t* q, const float* s, const float* x, float* y, int rows, int cols) {
  int r = 0;
  for (; r + 4 <= rows; r += 4) {
    const int8_t* w0 = q + size_t(r) * cols;
    const int8_t* w1 = w0 + cols;
    const int8_t* w2 = w1 + cols;
    const int8_t* w3 = w2 + cols;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int c = 0; c < cols; c++) {
      const float xv = x[c];
      a0 += float(w0[c]) * xv;
      a1 += float(w1[c]) * xv;
      a2 += float(w2[c]) * xv;
      a3 += float(w3[c]) * xv;
    }
    y[r] = a0 * s[r];
    y[r + 1] = a1 * s[r + 1];
    y[r + 2] = a2 * s[r + 2];
    y[r + 3] = a3 * s[r + 3];
  }
  if (r < rows) mv_ref(q + size_t(r) * cols, s + r, x, y + r, rows - r, cols);
}

// Eight rows at a time (more parallelism; may run out of floating-point registers on Xtensa).
static inline void mv_r8(const int8_t* q, const float* s, const float* x, float* y, int rows, int cols) {
  int r = 0;
  for (; r + 8 <= rows; r += 8) {
    const int8_t* w = q + size_t(r) * cols;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    for (int c = 0; c < cols; c++) {
      const float xv = x[c];
      a0 += float(w[c]) * xv;
      a1 += float(w[c + cols]) * xv;
      a2 += float(w[c + 2 * size_t(cols)]) * xv;
      a3 += float(w[c + 3 * size_t(cols)]) * xv;
      a4 += float(w[c + 4 * size_t(cols)]) * xv;
      a5 += float(w[c + 5 * size_t(cols)]) * xv;
      a6 += float(w[c + 6 * size_t(cols)]) * xv;
      a7 += float(w[c + 7 * size_t(cols)]) * xv;
    }
    y[r] = a0 * s[r];
    y[r + 1] = a1 * s[r + 1];
    y[r + 2] = a2 * s[r + 2];
    y[r + 3] = a3 * s[r + 3];
    y[r + 4] = a4 * s[r + 4];
    y[r + 5] = a5 * s[r + 5];
    y[r + 6] = a6 * s[r + 6];
    y[r + 7] = a7 * s[r + 7];
  }
  if (r < rows) mv_ref(q + size_t(r) * cols, s + r, x, y + r, rows - r, cols);
}

// ---- int8 activations (W8A8): x quantized in groups of G elements (one scale per group), integer int8*int8 products.
// y[r] = s[r] * sum_g xs[g] * (sum_{c in group g} q[r][c] * xq[c])   (inner sum in int32, exact)
static inline void mv_a8(const int8_t* q, const float* s, const int8_t* xq, const float* xs, int G, float* y, int rows,
                         int cols) {
  for (int r = 0; r < rows; r++) {
    const int8_t* w = q + size_t(r) * cols;
    float acc = 0.f;
    for (int c0 = 0, g = 0; c0 < cols; c0 += G, g++) {
      const int c1 = c0 + G < cols ? c0 + G : cols;
      int32_t a = 0;
      for (int c = c0; c < c1; c++) a += int32_t(w[c]) * int32_t(xq[c]);
      acc += float(a) * xs[g];
    }
    y[r] = acc * s[r];
  }
}

// Quantizes x[0..n) in groups of G: xs[g] = max|x| / 127, xq = rint(x / xs) in [-127, 127]. Returns the number of groups.
static inline int quant_act(const float* x, int n, int G, int8_t* xq, float* xs) {
  int g = 0;
  for (int c0 = 0; c0 < n; c0 += G, g++) {
    const int c1 = c0 + G < n ? c0 + G : n;
    float amax = 0.f;
    for (int c = c0; c < c1; c++) { const float a = fabsf(x[c]); if (a > amax) amax = a; }
    const float sc = amax > 0.f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / sc;
    xs[g] = sc;
    for (int c = c0; c < c1; c++) {
      float v = x[c] * inv;
      v = (v + 12582912.0f) - 12582912.0f;   // round half to even (same as rint) without calling libm
      v = v > 127.f ? 127.f : (v < -127.f ? -127.f : v);
      xq[c] = int8_t(v);
    }
  }
  return g;
}

typedef void (*MvA8)(const int8_t* q, const float* s, const int8_t* xq, const float* xs, int G, float* y, int rows,
                     int cols);
inline MvA8 g_a8 = mv_a8;    // integer kernel in use (mv_a8 in C; mv_a8_pie on the ESP32-S3 once it passes the boot self-check)

inline int g_aq_group = 0;   // 0 = float activations (default); G > 0 = int8 activations with one scale per group of G

// ---- int4 weights (MLP) x int8 activations (groups of 32): y[r] = sum_g float(a_g) * (xs[g] * ws[r][g]),
// a_g = sum_{j<16} (u_j - 8) * x[16g+j] + sum_{j<16} h_j * x[16g+16+j]   (u: low nibble 0..15, h: high nibble -8..7)
// xsum[g] = sum of the first 16 xq values of the group (lets the SIMD kernel remove the +8 offset of the low nibbles).
typedef void (*MvA4)(const uint8_t* q, const uint16_t* gs, const int8_t* xq, const float* xs, const int32_t* xsum,
                     float* y, int rows, int ld);

static inline float bf16_to_f32(uint16_t b) {
  const uint32_t u = uint32_t(b) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static inline void mv_a4(const uint8_t* q, const uint16_t* gs, const int8_t* xq, const float* xs, const int32_t* /*xsum*/,
                         float* y, int rows, int ld) {
  const int ng = ld >> 5;
  for (int r = 0; r < rows; r++) {
    const uint8_t* w = q + size_t(r) * (ld >> 1);
    float acc = 0.f;
    for (int g = 0; g < ng; g++) {
      int32_t a = 0;
      for (int j = 0; j < 16; j++) {
        const uint8_t b = w[g * 16 + j];
        a += (int32_t(b & 15u) - 8) * int32_t(xq[g * 32 + j]);
        a += int32_t(int8_t(b & 0xF0u) >> 4) * int32_t(xq[g * 32 + 16 + j]);
      }
      acc += float(a) * (xs[g] * bf16_to_f32(gs[size_t(r) * ng + g]));
    }
    y[r] = acc;
  }
}

inline MvA4 g_a4 = mv_a4;   // int4 kernel in use (mv_a4 in C; mv_a4_pie on the ESP32-S3 once it passes the boot self-check)

#if (defined(ETHORIC_PIE) && defined(__XTENSA__)) || defined(ETHORIC_PIE_EMUL)
// ESP32-S3 (PIE): 16 int8*int8 products per instruction, accumulated in ACCX (40 bits). Requirements: G == 32,
// cols a multiple of 16, q and xq aligned to 16 bytes (rows have ld a multiple of 16 in format v2).
// The result is EXACTLY that of mv_a8 (integer sums in int32, same float accumulation order).
// pie_dot32 / pie_dot16: sum of 32 / 16 products; w and x advance to the next block.
#if defined(ETHORIC_PIE_EMUL)      // C emulation (host test of the kernel's block and pointer logic)
static inline int32_t pie_dot32(const int8_t*& w, const int8_t*& x) {
  int32_t a = 0;
  for (int i = 0; i < 32; i++) a += int32_t(w[i]) * int32_t(x[i]);
  w += 32; x += 32;
  return a;
}
static inline int32_t pie_dot16(const int8_t*& w, const int8_t*& x) {
  int32_t a = 0;
  for (int i = 0; i < 16; i++) a += int32_t(w[i]) * int32_t(x[i]);
  w += 16; x += 16;
  return a;
}
#else
static inline int32_t pie_dot32(const int8_t*& w, const int8_t*& x) {
  int32_t a;
  asm volatile(
      "ee.zero.accx\n"
      "ee.vld.128.ip q0, %[w], 16\n"
      "ee.vld.128.ip q1, %[x], 16\n"
      "ee.vmulas.s8.accx q0, q1\n"
      "ee.vld.128.ip q2, %[w], 16\n"
      "ee.vld.128.ip q3, %[x], 16\n"
      "ee.vmulas.s8.accx q2, q3\n"
      "rur.accx_0 %[a]\n"
      : [a] "=r"(a), [w] "+r"(w), [x] "+r"(x)
      :
      : "memory");
  return a;
}
static inline int32_t pie_dot16(const int8_t*& w, const int8_t*& x) {
  int32_t a;
  asm volatile(
      "ee.zero.accx\n"
      "ee.vld.128.ip q0, %[w], 16\n"
      "ee.vld.128.ip q1, %[x], 16\n"
      "ee.vmulas.s8.accx q0, q1\n"
      "rur.accx_0 %[a]\n"
      : [a] "=r"(a), [w] "+r"(w), [x] "+r"(x)
      :
      : "memory");
  return a;
}
#endif
static inline void mv_a8_pie(const int8_t* q, const float* s, const int8_t* xq, const float* xs, int /*G*/, float* y,
                             int rows, int cols) {
  const int nb = cols >> 4;   // 16-element blocks
  const int8_t* w = q;
  for (int r = 0; r < rows; r++) {
    const int8_t* xp = xq;
    float acc = 0.f;
    int b = 0, g = 0;
    for (; b + 2 <= nb; b += 2, g++) acc += float(pie_dot32(w, xp)) * xs[g];
    if (b < nb) acc += float(pie_dot16(w, xp)) * xs[g];   // one 16-element block left (ld is a multiple of 16 but not of 32)
    y[r] = acc * s[r];
  }
}
#endif

#if (defined(ETHORIC_PIE) && defined(__XTENSA__)) || defined(ETHORIC_PIE_EMUL)
// int4 SIMD kernel: only AND and int8 multiply-accumulate (no shifts):
//   w & 0x0F  = u (0..15)        -> sum u*x_low  ;  a_low  = sum u*x - 8*xsum
//   w & 0xF0  = 16*h (int8)      -> sum (16h)*x_high ;  a_high = (sum (16h)*x) >> 4   (exact division)
#if defined(ETHORIC_PIE_EMUL)
static inline void pie4_group(const uint8_t*& w, const int8_t*& x, const uint8_t* /*mask*/, int32_t& lo, int32_t& hi) {
  lo = 0; hi = 0;
  for (int j = 0; j < 16; j++) {
    lo += int32_t(w[j] & 0x0Fu) * int32_t(x[j]);
    hi += int32_t(int8_t(w[j] & 0xF0u)) * int32_t(x[16 + j]);
  }
  w += 16; x += 32;
}
static inline void pie4_load_masks(const uint8_t*) {}
#else
static inline void pie4_load_masks(const uint8_t* mask) {   // mask: 32 bytes aligned to 16 (16 x 0x0F, 16 x 0xF0)
  asm volatile(
      "ee.vld.128.ip q6, %[m], 16\n"
      "ee.vld.128.ip q7, %[m], 16\n"
      : [m] "+r"(mask)
      :
      : "memory");
}
static inline void pie4_group(const uint8_t*& w, const int8_t*& x, const uint8_t* /*mask*/, int32_t& lo, int32_t& hi) {
  asm volatile(
      "ee.vld.128.ip q0, %[w], 16\n"
      "ee.vld.128.ip q1, %[x], 16\n"
      "ee.vld.128.ip q2, %[x], 16\n"
      "ee.andq q3, q0, q6\n"
      "ee.andq q4, q0, q7\n"
      "ee.zero.accx\n"
      "ee.vmulas.s8.accx q3, q1\n"
      "rur.accx_0 %[lo]\n"
      "ee.zero.accx\n"
      "ee.vmulas.s8.accx q4, q2\n"
      "rur.accx_0 %[hi]\n"
      : [lo] "=r"(lo), [hi] "=r"(hi), [w] "+r"(w), [x] "+r"(x)
      :
      : "memory");
}
#endif
static inline void mv_a4_pie(const uint8_t* q, const uint16_t* gs, const int8_t* xq, const float* xs, const int32_t* xsum,
                             float* y, int rows, int ld) {
  alignas(16) uint8_t mask[32];
  for (int i = 0; i < 16; i++) { mask[i] = 0x0F; mask[16 + i] = 0xF0; }
  pie4_load_masks(mask);
  const int ng = ld >> 5;
  const uint8_t* w = q;
  for (int r = 0; r < rows; r++) {
    const int8_t* xp = xq;
    const uint16_t* gr = gs + size_t(r) * ng;
    float acc = 0.f;
    for (int g = 0; g < ng; g++) {
      int32_t lo, hi;
      pie4_group(w, xp, mask, lo, hi);
      const int32_t a = lo - 8 * xsum[g] + (hi >> 4);
      acc += float(a) * (xs[g] * bf16_to_f32(gr[g]));
    }
    y[r] = acc;
  }
}
#endif

static inline bool a4_selfcheck(MvA4 cand, const Mat& m, int rows) {
  const int ld = m.ld, ng = ld >> 5;
  int8_t* xq = static_cast<int8_t*>(aligned_alloc(16, size_t(ld + 31) & ~size_t(15)));
  float* xs = static_cast<float*>(malloc(sizeof(float) * (ng + 2)));
  int32_t* xsum = static_cast<int32_t*>(malloc(sizeof(int32_t) * (ng + 2)));
  float* ya = static_cast<float*>(malloc(sizeof(float) * rows));
  float* yb = static_cast<float*>(malloc(sizeof(float) * rows));
  bool good = xq && xs && xsum && ya && yb && m.bits == 4;
  if (good) {
    uint32_t st = 777u;
    for (int c = 0; c < ld; c++) {
      st = st * 1664525u + 1013904223u;
      xq[c] = int8_t(int((st >> 24) & 255u) - 128);
    }
    for (int g = 0; g < ng; g++) {
      xs[g] = 0.01f * float(g + 1);
      xsum[g] = 0;
      for (int j = 0; j < 16; j++) xsum[g] += xq[g * 32 + j];
    }
    mv_a4(reinterpret_cast<const uint8_t*>(m.q), m.gs, xq, xs, xsum, ya, rows, ld);
    cand(reinterpret_cast<const uint8_t*>(m.q), m.gs, xq, xs, xsum, yb, rows, ld);
    good = memcmp(ya, yb, sizeof(float) * rows) == 0;
  }
  free(xq); free(xs); free(xsum); free(ya); free(yb);
  return good;
}

// Checks `cand` against mv_a8 (C) on a real matrix with a pseudo-random vector: returns true if the `rows` results
// are bit-for-bit IDENTICAL. Used at ESP32 boot to enable the SIMD kernel only if it is correct.
static inline bool a8_selfcheck(MvA8 cand, const Mat& m, int rows) {
  const int ld = m.ld;
  int8_t* xq = static_cast<int8_t*>(aligned_alloc(16, size_t(ld + 31) & ~size_t(15)));
  float* xs = static_cast<float*>(malloc(sizeof(float) * (ld / 16 + 2)));
  float* ya = static_cast<float*>(malloc(sizeof(float) * rows));
  float* yb = static_cast<float*>(malloc(sizeof(float) * rows));
  bool good = xq && xs && ya && yb;
  if (good) {
    uint32_t st = 12345u;
    for (int c = 0; c < ld; c++) {
      st = st * 1664525u + 1013904223u;
      xq[c] = int8_t(int((st >> 24) & 255u) - 128);
    }
    for (int g = 0; g < ld / 16 + 2; g++) xs[g] = 0.01f * float(g + 1);
    mv_a8(m.q, m.s, xq, xs, 32, ya, rows, ld);
    cand(m.q, m.s, xq, xs, 32, yb, rows, ld);
    good = memcmp(ya, yb, sizeof(float) * rows) == 0;
  }
  free(xq); free(xs); free(ya); free(yb);
  return good;
}

// SIMD kernel when compiled in (ETHORIC_PIE on the ESP32-S3), otherwise the C one.
static inline MvA8 mv_a8_pie_sel() {
#if (defined(ETHORIC_PIE) && defined(__XTENSA__)) || defined(ETHORIC_PIE_EMUL)
  return mv_a8_pie;
#else
  return mv_a8;
#endif
}
static inline MvA4 mv_a4_pie_sel() {
#if (defined(ETHORIC_PIE) && defined(__XTENSA__)) || defined(ETHORIC_PIE_EMUL)
  return mv_a4_pie;
#else
  return mv_a4;
#endif
}

struct KernelInfo {
  const char* name;
  MvKernel fn;
};
static const KernelInfo kKernels[] = {{"ref", mv_ref}, {"u4", mv_u4}, {"r4", mv_r4}, {"r8", mv_r8}};
static const int kNumKernels = int(sizeof(kKernels) / sizeof(kKernels[0]));

inline MvKernel g_mv = mv_ref;   // kernel in use (change with set_kernel / autotune)

static inline bool set_kernel(const char* name) {
  for (int i = 0; i < kNumKernels; i++)
    if (!strcmp(kKernels[i].name, name)) { g_mv = kKernels[i].fn; return true; }
  return false;
}

// Mean time (us) of one call of kernel k. `now_us` returns a clock in microseconds.
static inline double bench_kernel(MvKernel k, const int8_t* q, const float* s, const float* x, float* y, int rows,
                                  int cols, int reps, int64_t (*now_us)()) {
  k(q, s, x, y, rows, cols);   // warm-up (cache, MMU)
  const int64_t t0 = now_us();
  for (int i = 0; i < reps; i++) k(q, s, x, y, rows, cols);
  const int64_t t1 = now_us();
  return double(t1 - t0) / reps;
}

// ----------------------------------------------------------------------------- parallelism (2 cores)

// The Runner splits the rows of each matrix into `nparts` = 2 parts: part 0 runs on the calling core and part 1 on the other one.
struct Parallel {
  void* ctx;
  void (*begin)(void* ctx);   // start of a token (wakes the worker)
  void (*end)(void* ctx);     // end of a token (the worker may sleep)
  // runs job(jctx, 0, 2) here and job(jctx, 1, 2) on the other core; returns when both are done
  void (*run)(void* ctx, void (*job)(void* jctx, int part, int nparts), void* jctx);
};

// ----------------------------------------------------------------------------- model

class Model {
 public:
  Config cfg{};
  Mat embed{};
  const float* final_norm = nullptr;
  Layer* layers = nullptr;

  ~Model() {
    for (int i = 0; i < n_owned_; i++) free(owned_[i]);
    free(layers);
  }

  // blob: start of the file (64-byte aligned). Returns false if the format does not match.
  bool load(const uint8_t* blob, size_t size) {
    if (size < 128 || memcmp(blob, "ETHW", 4) != 0) return false;
    uint32_t ver;
    memcpy(&ver, blob + 4, 4);
    if (ver < 1 || ver > 3) return false;   // v2: rows padded to a multiple of 16; v3: MLP in int4 (flags bit 2)
    ver_ = ver;
    int32_t v[11];
    memcpy(v, blob + 8, sizeof(v));
    uint32_t flags, wmask;
    memcpy(&flags, blob + 52, 4);
    memcpy(&wmask, blob + 56, 4);
    float th, ep;
    memcpy(&th, blob + 60, 4);
    memcpy(&ep, blob + 64, 4);
    memcpy(&body_crc_, blob + 68, 4);
    memcpy(&total_, blob + 72, 8);
    uint32_t voff, vbytes;               // vocabulary table (optional; zeros = absent, as in the old format)
    memcpy(&voff, blob + 80, 4);
    memcpy(&vbytes, blob + 84, 4);
    if (total_ < 128 || total_ > size) return false;   // truncated file / incomplete flash write
    blob_ = blob;
    cfg = Config{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10],
                 (flags & 1) != 0, (flags & 2) != 0, wmask, th, ep};
    const int d = cfg.d, nh = cfg.n_head, nkv = cfg.n_kv, hd = cfg.hd, F = cfg.ffn, k = cfg.kernel;
    size_t off = 128;
    bool ok = true;
    auto take = [&](size_t bytes) -> const void* {
      off = (off + 63) & ~size_t(63);
      const void* p = blob + off;
      off += bytes;
      if (off > total_) ok = false;
      return p;
    };
    auto f32 = [&](size_t n) { return static_cast<const float*>(take(n * 4)); };
    uint32_t m4mask = 0;
    memcpy(&m4mask, blob + 88, 4);   // v3: bit i = the MLP of layer i is stored in int4
    int4_ = ver_ >= 3 && (flags & 4u) != 0 && m4mask != 0;
    auto mat = [&](int rows, int cols, int bits) {
      Mat m;
      m.bits = bits;
      m.gs = nullptr;
      if (bits == 4) {
        m.s = nullptr;
        m.ld = (cols + 31) & ~31;
        m.gs = static_cast<const uint16_t*>(take(size_t(rows) * (m.ld / 32) * 2));
        m.q = static_cast<const int8_t*>(take(size_t(rows) * (m.ld / 2)));
      } else {
        m.s = f32(rows);
        m.ld = ver_ >= 2 ? ((cols + 15) & ~15) : cols;
        m.q = static_cast<const int8_t*>(take(size_t(rows) * m.ld));
      }
      return m;
    };
    embed = mat(cfg.vocab, d, 8);
    final_norm = f32(d);
    layers = static_cast<Layer*>(calloc(cfg.n_layer, sizeof(Layer)));
    if (!layers) return false;
    for (int i = 0; i < cfg.n_layer; i++) {
      Layer& L = layers[i];
      L.attn_in_norm = f32(d);
      L.attn_out_norm = f32(d);
      L.mlp_in_norm = f32(d);
      L.mlp_out_norm = f32(d);
      L.q_norm = f32(hd);
      L.k_norm = f32(hd);
      L.gate_w = f32(size_t(nh) * d);
      L.gate_b = f32(nh);
      L.canon_a = f32(size_t(d) * k);
      L.canon_c = f32(size_t(d) * k);
      L.q = mat(nh * hd, d, 8);
      L.k = mat(nkv * hd, d, 8);
      L.v = mat(nkv * hd, d, 8);
      L.o = mat(d, nh * hd, 8);
      const int mb = int4_ && ((m4mask >> i) & 1u) ? 4 : 8;
      L.mlp_gate = mat(F, d, mb);
      L.mlp_up = mat(F, d, mb);
      L.mlp_down = mat(d, F, mb);
      L.windowed = cfg.window > 0 && ((wmask >> i) & 1u);
    }
    if (ok && voff != 0) {               // table: u32 offs[vocab+1] followed by the token bytes
      const uint64_t table = 4ull * (uint64_t(cfg.vocab) + 1);
      if (uint64_t(voff) + vbytes <= total_ && vbytes >= table && (voff & 3u) == 0) {
        vocab_offs_ = reinterpret_cast<const uint32_t*>(blob + voff);
        vocab_data_ = blob + voff + table;
        vocab_data_len_ = uint32_t(vbytes - table);
      }
    }
    uint32_t boff = 0, bbytes = 0;       // optional BPE tokenizer tables (text -> ids on the device)
    memcpy(&boff, blob + 92, 4);
    memcpy(&bbytes, blob + 96, 4);
    if (ok && boff != 0 && uint64_t(boff) + bbytes <= total_ && (boff & 3u) == 0) {
      bpe_.init(blob + boff, bbytes);
    }
    return ok;   // `take` already rejects any tensor that extends past the total size recorded in the header
  }

  // Checks the CRC32 of the body against the header (reads the whole file; ~0.5 s on the ESP32). Call it once at boot,
  // BEFORE `relocate` (afterwards the pointers no longer point into the blob, but the CRC still reads the blob).
  bool verify_crc() const {
    if (!blob_) return false;
    static const uint32_t t[16] = {0x00000000, 0x1DB71064, 0x3B6E20C8, 0x26D930AC, 0x76DC4190, 0x6B6B51F4,
                                   0x4DB26158, 0x5005713C, 0xEDB88320, 0xF00F9344, 0xD6D6A3E8, 0xCB61B38C,
                                   0x9B64C2B0, 0x86D3D2D4, 0xA00AE278, 0xBDBDF21C};
    uint32_t crc = 0xFFFFFFFFu;
    for (uint64_t i = 128; i < total_; i++) {
      crc ^= blob_[i];
      crc = (crc >> 4) ^ t[crc & 15];
      crc = (crc >> 4) ^ t[crc & 15];
    }
    return ~crc == body_crc_;
  }

  uint64_t total_bytes() const { return total_; }
  bool has_int4() const { return int4_; }   // MLP in int4: requires int8 activations (the Runner enables them automatically)

  // Copies matrices (scales + weights) into memory obtained from `alloc` (e.g. PSRAM) as long as they fit in `budget`
  // bytes: small float tables first, then the embedding, then layer 0, 1, ... Every byte is read once per token, so any
  // subset pays off equally per byte. Returns the number of bytes copied. The memory is released in the destructor.
  size_t relocate(void* (*alloc)(size_t), size_t budget) {
    size_t used = 0;
    auto one = [&](Mat& m, int rows, int /*cols*/) {
      if (!m.q) return;
      const size_t sbytes = m.bits == 4 ? size_t(rows) * (m.ld / 32) * 2 : size_t(rows) * 4;
      const size_t qbytes = m.bits == 4 ? size_t(rows) * (m.ld / 2) : size_t(rows) * m.ld;
      const size_t sb = (sbytes + 15) & ~size_t(15), total = sb + qbytes;   // q aligned to 16
      if (used + total > budget || n_owned_ >= kMaxOwned) return;
      uint8_t* p = static_cast<uint8_t*>(alloc(total));
      if (!p) return;
      memcpy(p, m.bits == 4 ? static_cast<const void*>(m.gs) : static_cast<const void*>(m.s), sbytes);
      memcpy(p + sb, m.q, qbytes);
      if (m.bits == 4) m.gs = reinterpret_cast<const uint16_t*>(p);
      else m.s = reinterpret_cast<const float*>(p);
      m.q = reinterpret_cast<const int8_t*>(p + sb);
      owned_[n_owned_++] = p;
      used += total;
    };
    const int d = cfg.d, nh = cfg.n_head, nkv = cfg.n_kv, hd = cfg.hd, F = cfg.ffn;
    // small float tables (norms, gate, Canon): ~21 KB per layer, read in full on every token
    for (int i = 0; i < cfg.n_layer; i++) {
      Layer& L = layers[i];
      const size_t nf = size_t(4) * d + 2 * hd + size_t(nh) * d + nh + size_t(2) * d * cfg.kernel;
      const size_t bytes = (nf * 4 + 63) & ~size_t(63);
      if (used + bytes > budget || n_owned_ >= kMaxOwned) break;
      float* p = static_cast<float*>(alloc(bytes));
      if (!p) break;
      float* w = p;
      auto mv = [&](const float*& src, size_t n) { memcpy(w, src, n * 4); src = w; w += n; };
      mv(L.attn_in_norm, d); mv(L.attn_out_norm, d); mv(L.mlp_in_norm, d); mv(L.mlp_out_norm, d);
      mv(L.q_norm, hd); mv(L.k_norm, hd); mv(L.gate_w, size_t(nh) * d); mv(L.gate_b, nh);
      mv(L.canon_a, size_t(d) * cfg.kernel); mv(L.canon_c, size_t(d) * cfg.kernel);
      owned_[n_owned_++] = p;
      used += bytes;
    }
    one(embed, cfg.vocab, d);
    for (int i = 0; i < cfg.n_layer; i++) {
      Layer& L = layers[i];
      one(L.q, nh * hd, d);
      one(L.k, nkv * hd, d);
      one(L.v, nkv * hd, d);
      one(L.o, d, nh * hd);
      one(L.mlp_gate, F, d);
      one(L.mlp_up, F, d);
      one(L.mlp_down, d, F);
    }
    return used;
  }

  // UTF-8 bytes of token `id` (vocabulary embedded in the file). nullptr if the file has no vocabulary;
  // *len = 0 for special tokens (nothing to print).
  const uint8_t* token_bytes(int id, int* len) const {
    *len = 0;
    if (!vocab_offs_ || id < 0 || id >= cfg.vocab) return nullptr;
    const uint32_t a = vocab_offs_[id], b = vocab_offs_[id + 1];
    if (b < a || b > vocab_data_len_) return nullptr;
    *len = int(b - a);
    return vocab_data_ + a;
  }
  bool has_vocab() const { return vocab_offs_ != nullptr; }

  // Text -> token ids using the tables embedded in the file. Returns the id count, -1 if `cap` is too small, or -2 if
  // the file has no tokenizer tables.
  bool has_tokenizer() const { return bpe_.ready(); }
  int encode(const char* text, size_t len, int* out, int cap) const {
    return bpe_.ready() ? bpe_.encode(text, len, out, cap) : -2;
  }

 private:
  static const int kMaxOwned = 256;
  const uint8_t* blob_ = nullptr;
  uint32_t body_crc_ = 0;
  uint32_t ver_ = 1;
  bool int4_ = false;
  uint64_t total_ = 0;
  const uint32_t* vocab_offs_ = nullptr;
  const uint8_t* vocab_data_ = nullptr;
  uint32_t vocab_data_len_ = 0;
  BpeTokenizer bpe_;
  void* owned_[kMaxOwned] = {};
  int n_owned_ = 0;
};

// ----------------------------------------------------------------------------- operators

static inline void rmsnorm(float* y, const float* x, const float* w, int n, float eps) {
  float ss = 0.f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  const float inv = 1.0f / sqrtf(ss / float(n) + eps);
  for (int i = 0; i < n; i++) y[i] = x[i] * inv * w[i];
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// ----------------------------------------------------------------------------- execution

class Runner {
 public:
  // max_ctx <= cfg.max_seq. KV in fp32; windowed layers use a ring of `window` entries.
  Runner(const Model& m, int max_ctx) : m_(m), c_(m.cfg) {
    max_ctx_ = max_ctx < c_.max_seq ? max_ctx : c_.max_seq;
    const int half = c_.hd / 2, kv = c_.n_kv * c_.hd;
    const int L = c_.n_layer, k1 = c_.kernel - 1;
    cap_ = (int*)calloc(L, sizeof(int));
    if (!cap_) return;    // ok_ stays false
    K_ = (float**)calloc(L, sizeof(float*));
    V_ = (float**)calloc(L, sizeof(float*));
    kdoc_ = (int16_t**)calloc(L, sizeof(int16_t*));
    ca_ = (float**)calloc(L, sizeof(float*));
    cc_ = (float**)calloc(L, sizeof(float*));
    cadoc_ = (int*)calloc(size_t(L) * k1 + 1, sizeof(int));
    ccdoc_ = (int*)calloc(size_t(L) * k1 + 1, sizeof(int));
    if (!K_ || !V_ || !kdoc_ || !ca_ || !cc_ || !cadoc_ || !ccdoc_) return;
    for (int l = 0; l < L; l++) {
      cap_[l] = m.layers[l].windowed && c_.window < max_ctx_ ? c_.window : max_ctx_;
      K_[l] = (float*)calloc(size_t(cap_[l]) * kv, sizeof(float));
      V_[l] = (float*)calloc(size_t(cap_[l]) * kv, sizeof(float));
      kdoc_[l] = (int16_t*)calloc(cap_[l], sizeof(int16_t));
      ca_[l] = (float*)calloc(size_t(k1) * c_.d, sizeof(float));
      cc_[l] = (float*)calloc(size_t(k1) * c_.d, sizeof(float));
    }
    cos_ = (float*)malloc(sizeof(float) * size_t(max_ctx_) * half);
    sin_ = (float*)malloc(sizeof(float) * size_t(max_ctx_) * half);
    const int bigw = c_.n_head * c_.hd > c_.ffn ? c_.n_head * c_.hd : c_.ffn;
    x_ = (float*)malloc(sizeof(float) * c_.d);
    h_ = (float*)calloc(size_t(c_.d) + 32, sizeof(float));   // matrix input buffers: zeroed tail (padding up to ld)
    t_ = (float*)malloc(sizeof(float) * c_.d);
    q_ = (float*)malloc(sizeof(float) * c_.n_head * c_.hd);
    k_ = (float*)malloc(sizeof(float) * kv);
    v_ = (float*)malloc(sizeof(float) * kv);
    o_ = (float*)calloc(size_t(c_.n_head) * c_.hd + 32, sizeof(float));
    g_ = (float*)calloc(size_t(bigw) + 32, sizeof(float));
    u_ = (float*)malloc(sizeof(float) * bigw);
    sc_ = (float*)malloc(sizeof(float) * 2 * size_t(max_ctx_));   // one score buffer per core
    xq_ = (int8_t*)aligned_alloc(16, (size_t(bigw) + 80) & ~size_t(15));
    xs_ = (float*)malloc(sizeof(float) * (size_t(bigw) / 16 + 4));
    xsum_ = (int32_t*)malloc(sizeof(int32_t) * (size_t(bigw) / 16 + 4));
    logits_ = (float*)malloc(sizeof(float) * c_.vocab);
    ok_ = cap_ && K_ && V_ && kdoc_ && ca_ && cc_ && cadoc_ && ccdoc_ && cos_ && sin_ && x_ && h_ && t_ && q_ &&
          k_ && v_ && o_ && g_ && u_ && sc_ && xq_ && xs_ && xsum_ && logits_;
    for (int l = 0; ok_ && l < L; l++) ok_ = K_[l] && V_[l] && kdoc_[l] && ca_[l] && cc_[l];
    if (!ok_) return;     // out of memory: the caller must check ok() before use
    for (int p = 0; p < max_ctx_; p++)
      for (int i = 0; i < half; i++) {
        const float f = 1.0f / powf(c_.theta, float(i) / float(half));
        const float a = float(p) * f;
        cos_[p * half + i] = cosf(a);
        sin_[p * half + i] = sinf(a);
      }
    reset();
  }

  bool ok() const { return ok_; }   // false if there was not enough memory for the requested context

  ~Runner() {
    if (K_ && V_ && kdoc_ && ca_ && cc_)
      for (int l = 0; l < c_.n_layer; l++) {
        free(K_[l]); free(V_[l]); free(kdoc_[l]); free(ca_[l]); free(cc_[l]);
      }
    free(cap_); free(K_); free(V_); free(kdoc_); free(ca_); free(cc_); free(cadoc_); free(ccdoc_);
    free(cos_); free(sin_); free(x_); free(h_); free(t_); free(q_); free(k_); free(v_); free(o_);
    free(g_); free(u_); free(sc_); free(xq_); free(xs_); free(xsum_); free(logits_);
  }
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  void reset() {
    pos_ = 0;
    doc_ = 0;
    const int k1 = c_.kernel - 1;
    for (int l = 0; l < c_.n_layer; l++) {
      memset(ca_[l], 0, sizeof(float) * k1 * c_.d);
      memset(cc_[l], 0, sizeof(float) * k1 * c_.d);
      for (int i = 0; i < k1; i++) { cadoc_[l * k1 + i] = -1; ccdoc_[l * k1 + i] = -1; }
    }
  }

  // Splits every matrix-vector product between two cores (nullptr = off). The object must outlive the Runner.
  void set_parallel(const Parallel* p) { par_ = p; }

  int pos() const { return pos_; }
  int max_ctx() const { return max_ctx_; }

  // RAM bytes used (KV + states + tables + buffers), for planning PSRAM usage.
  size_t ram_bytes() const {
    size_t b = 0;
    const int kv = c_.n_kv * c_.hd, half = c_.hd / 2, k1 = c_.kernel - 1;
    for (int l = 0; l < c_.n_layer; l++)
      b += size_t(cap_[l]) * (2 * kv * 4 + 2) + 2 * size_t(k1) * c_.d * 4;
    b += size_t(max_ctx_) * half * 8;
    b += sizeof(float) * (size_t(c_.d) * 3 + c_.n_head * c_.hd * 2 + kv * 2 + 2 * (c_.ffn > c_.n_head * c_.hd ? c_.ffn : c_.n_head * c_.hd) + max_ctx_ + c_.vocab);
    return b;
  }

  // Processes ONE token and returns the logits [vocab] (internal buffer, valid until the next step), or nullptr when the context is full.
  const float* step(int tok) {
    if (pos_ >= max_ctx_) return nullptr;
    if (m_.has_int4()) g_aq_group = 32;   // int4 weights only work with int8 activations (groups of 32)
    if (par_) par_->begin(par_->ctx);
    const int d = c_.d, H = c_.n_head, Hkv = c_.n_kv, hd = c_.hd, half = hd / 2;
    const int kv = Hkv * hd, k = c_.kernel, k1 = k - 1;
    const int adoc = c_.doc_masking ? doc_ : 0;
    const int cdoc = c_.canon_doc_masking ? doc_ : 0;
    const float scale = 1.0f / sqrtf(float(hd));
    const float* cs = cos_ + size_t(pos_) * half;
    const float* sn = sin_ + size_t(pos_) * half;

    const int8_t* er = m_.embed.q + size_t(tok) * m_.embed.ld;
    for (int i = 0; i < d; i++) x_[i] = float(er[i]) * m_.embed.s[tok];

    for (int l = 0; l < c_.n_layer; l++) {
      const Layer& L = m_.layers[l];
      rmsnorm(h_, x_, L.attn_in_norm, d, c_.eps);
      canon(L.canon_a, h_, ca_[l], cadoc_ + l * k1, cdoc);
      prep(h_, L.q.ld);
      mvp(L.q, h_, q_, H * hd);
      mvp(L.k, h_, k_, kv);
      mvp(L.v, h_, v_, kv);
      for (int j = 0; j < H; j++) { rmsnorm(q_ + j * hd, q_ + j * hd, L.q_norm, hd, c_.eps); rope(q_ + j * hd, cs, sn, half); }
      for (int j = 0; j < Hkv; j++) { rmsnorm(k_ + j * hd, k_ + j * hd, L.k_norm, hd, c_.eps); rope(k_ + j * hd, cs, sn, half); }

      const int cap = cap_[l];
      const int slot = pos_ % cap;
      memcpy(K_[l] + size_t(slot) * kv, k_, sizeof(float) * kv);
      memcpy(V_[l] + size_t(slot) * kv, v_, sizeof(float) * kv);
      kdoc_[l][slot] = int16_t(adoc);
      const int lo = L.windowed ? (pos_ - c_.window + 1 > 0 ? pos_ - c_.window + 1 : 0) : 0;
      const int n = pos_ - lo + 1;

      AttnJob aj{this, &L, l, lo, n, adoc, cap, scale};
      if (par_ && H >= 2) par_->run(par_->ctx, attn_job, &aj);   // heads split between the two cores
      else attn_job(&aj, 0, 1);
      prep(o_, L.o.ld);
      mvp(L.o, o_, t_, d);
      rmsnorm(t_, t_, L.attn_out_norm, d, c_.eps);
      for (int i = 0; i < d; i++) x_[i] += t_[i];

      rmsnorm(h_, x_, L.mlp_in_norm, d, c_.eps);
      canon(L.canon_c, h_, cc_[l], ccdoc_ + l * k1, cdoc);
      prep(h_, L.mlp_gate.ld);
      mvp(L.mlp_gate, h_, g_, c_.ffn);
      mvp(L.mlp_up, h_, u_, c_.ffn);
      for (int i = 0; i < c_.ffn; i++) g_[i] = g_[i] * sigmoidf(g_[i]) * u_[i];
      prep(g_, L.mlp_down.ld);
      mvp(L.mlp_down, g_, t_, d);
      rmsnorm(t_, t_, L.mlp_out_norm, d, c_.eps);
      for (int i = 0; i < d; i++) x_[i] += t_[i];
    }
    rmsnorm(h_, x_, m_.final_norm, d, c_.eps);
    prep(h_, m_.embed.ld);
    mvp(m_.embed, h_, logits_, c_.vocab);   // tied embedding
    pos_++;
    if (tok == c_.eos) doc_++;                 // EOS closes the document
    if (par_) par_->end(par_->ctx);
    return logits_;
  }

 private:
  struct MvJob {
    MvKernel k;
    const int8_t* q;
    const float* s;
    const float* x;
    float* y;
    int rows, cols;
    const int8_t* xq;   // != nullptr: int8 path (W8A8)
    const float* xs;
    int G;
    MvA8 a8;
    const uint16_t* gs;      // != nullptr: int4 matrix (weights in q as bytes)
    const int32_t* xsum;
    MvA4 a4;
  };
  static void mv_job(void* a, int part, int nparts) {
    const MvJob* j = static_cast<const MvJob*>(a);
    const int r0 = int((int64_t(j->rows) * part) / nparts);
    const int r1 = int((int64_t(j->rows) * (part + 1)) / nparts);
    if (r1 <= r0) return;
    if (j->gs) {
      const int ng = j->cols >> 5;
      j->a4(reinterpret_cast<const uint8_t*>(j->q) + size_t(r0) * (j->cols >> 1), j->gs + size_t(r0) * ng, j->xq, j->xs,
            j->xsum, j->y + r0, r1 - r0, j->cols);
    } else if (j->xq) j->a8(j->q + size_t(r0) * j->cols, j->s + r0, j->xq, j->xs, j->G, j->y + r0, r1 - r0, j->cols);
    else j->k(j->q + size_t(r0) * j->cols, j->s + r0, j->x, j->y + r0, r1 - r0, j->cols);
  }
  struct AttnJob {
    Runner* r;
    const Layer* L;
    int l, lo, n, adoc, cap;
    float scale;
  };
  static void attn_job(void* a, int part, int nparts) {
    const AttnJob* j = static_cast<const AttnJob*>(a);
    const int H = j->r->c_.n_head;
    const int j0 = (H * part) / nparts, j1 = (H * (part + 1)) / nparts;
    j->r->heads(*j, j0, j1, j->r->sc_ + size_t(part) * j->r->max_ctx_);
  }
  // Attention for heads [j0, j1) of layer j.l (each head is independent: the result does not depend on how the heads are split).
  void heads(const AttnJob& J, int j0, int j1, float* sc) {
    const int H = c_.n_head, hd = c_.hd, d = c_.d, kv = c_.n_kv * hd, rep = H / c_.n_kv;
    const int l = J.l, lo = J.lo, n = J.n, adoc = J.adoc, cap = J.cap;
    const Layer& L = *J.L;
    const int16_t* kd = kdoc_[l];
    for (int j = j0; j < j1; j++) {
      const int g = j / rep;
      const float* qj = q_ + j * hd;
      float mx = -INFINITY;
      int sl = lo % cap;
      for (int i = 0; i < n; i++, sl = (sl + 1 == cap ? 0 : sl + 1)) {
        if (kd[sl] != adoc) { sc[i] = -INFINITY; continue; }
        const float* kk = K_[l] + size_t(sl) * kv + g * hd;
        float s = 0.f;
        for (int e = 0; e < hd; e++) s += qj[e] * kk[e];
        s *= J.scale;
        sc[i] = s;
        if (s > mx) mx = s;
      }
      float sum = 0.f;
      for (int i = 0; i < n; i++) {
        sc[i] = sc[i] == -INFINITY ? 0.f : expf(sc[i] - mx);
        sum += sc[i];
      }
      const float inv = 1.0f / sum;
      float* oj = o_ + j * hd;
      for (int e = 0; e < hd; e++) oj[e] = 0.f;
      sl = lo % cap;
      for (int i = 0; i < n; i++, sl = (sl + 1 == cap ? 0 : sl + 1)) {
        if (sc[i] == 0.f) continue;
        const float p = sc[i] * inv;
        const float* vv = V_[l] + size_t(sl) * kv + g * hd;
        for (int e = 0; e < hd; e++) oj[e] += p * vv[e];
      }
      float gl = L.gate_b[j];
      const float* gw = L.gate_w + size_t(j) * d;
      for (int i = 0; i < d; i++) gl += gw[i] * h_[i];
      const float gate = sigmoidf(gl);
      for (int e = 0; e < hd; e++) oj[e] *= gate;
    }
  }
  // Prepares the input x (n floats, n = ld of the matrix) for the following mvp calls: quantizes it once to int8 when W8A8 is on.
  void prep(const float* x, int n) {
    if (g_aq_group > 0) {
      const int ng = quant_act(x, n, g_aq_group, xq_, xs_);
      if (g_aq_group == 32)   // sum of the first 16 values of each group (int4 kernels)
        for (int g = 0; g < ng; g++) {
          int32_t t = 0;
          for (int j = 0; j < 16; j++) t += xq_[g * 32 + j];
          xsum_[g] = t;
        }
    }
  }
  // y = M x (M: rows x ld; x has ld floats with a zeroed tail). With `par_`, half of the rows run on each core.
  void mvp(const Mat& m, const float* x, float* y, int rows) {
    MvJob job{g_mv, m.q, m.s, x, y, rows, m.ld, nullptr, nullptr, 0, g_a8, nullptr, nullptr, g_a4};
    if (g_aq_group > 0) { job.xq = xq_; job.xs = xs_; job.G = g_aq_group; }
    if (m.bits == 4) { job.gs = m.gs; job.xsum = xsum_; }
    if (par_ && rows >= 32) par_->run(par_->ctx, mv_job, &job);
    else mv_job(&job, 0, 1);
  }

  static inline void rope(float* x, const float* cs, const float* sn, int half) {
    for (int i = 0; i < half; i++) {
      const float x1 = x[i], x2 = x[i + half];
      x[i] = x1 * cs[i] - x2 * sn[i];
      x[i + half] = x2 * cs[i] + x1 * sn[i];
    }
  }

  // Canon: h <- h + causal depthwise convolution of kernel size k (window [state..., h]), with document masking.
  void canon(const float* w, float* h, float* state, int* sdoc, int doc) {
    const int d = c_.d, k = c_.kernel, k1 = k - 1;
    for (int c = 0; c < d; c++) {
      float acc = w[c * k + k1] * h[c];
      for (int i = 0; i < k1; i++)
        if (sdoc[i] == doc) acc += w[c * k + i] * state[i * d + c];
      t_[c] = acc;
    }
    if (k1 > 0) {
      if (k1 > 1) {
        memmove(state, state + d, sizeof(float) * (k1 - 1) * d);
        memmove(sdoc, sdoc + 1, sizeof(int) * (k1 - 1));
      }
      memcpy(state + (k1 - 1) * d, h, sizeof(float) * d);   // store the ORIGINAL input
      sdoc[k1 - 1] = doc;
    }
    for (int c = 0; c < d; c++) h[c] += t_[c];
  }

  const Model& m_;
  const Config& c_;
  const Parallel* par_ = nullptr;
  int max_ctx_ = 0, pos_ = 0, doc_ = 0;
  bool ok_ = false;
  int* cap_ = nullptr;
  float **K_ = nullptr, **V_ = nullptr, **ca_ = nullptr, **cc_ = nullptr;
  int16_t** kdoc_ = nullptr;
  int *cadoc_ = nullptr, *ccdoc_ = nullptr;
  float *cos_ = nullptr, *sin_ = nullptr;
  float *x_ = nullptr, *h_ = nullptr, *t_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *o_ = nullptr,
        *g_ = nullptr, *u_ = nullptr, *sc_ = nullptr, *xs_ = nullptr, *logits_ = nullptr;
  int8_t* xq_ = nullptr;
  int32_t* xsum_ = nullptr;
};

}  // namespace ethoric
