// Host (PC) driver for the Ethoric-10M C++ decoder. It builds against the very same ethoric.hpp that runs on
// the ESP32-S3, so everything verified here (numerics, kernels, threading, relocation) carries over to the device.
//
//   g++ -O2 -std=c++17 -pthread -I../components/ethoric/include -o ethoric_host host_main.cpp
//   ./ethoric_host model.bin --ids 461,307,412 --new 40 --text
//   ./ethoric_host model.bin --ids 461,307,...  --ppl                  # perplexity of the given ids
//   ./ethoric_host model.bin --ids ...          --logits-out l.bin     # float32 logits for every input token
//   ./ethoric_host model.bin --verify                                  # CRC32 check only
//   ./ethoric_host model.bin --bench                                   # time every float kernel
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define ETHORIC_PIE_EMUL 1  // emulate the SIMD kernels in C so their block/pointer logic can be tested on the host
#include "ethoric.hpp"
#include "sampler.hpp"

namespace {

// ---------------------------------------------------------------------------------------------------------------
// Two-thread executor, used to test the parallel path (on the ESP32 the second thread is core 1).
// The worker spins only while a job is pending.
// ---------------------------------------------------------------------------------------------------------------
struct Pool {
  std::atomic<int> state{0};  // 0 = idle, 1 = job ready, 2 = job done
  std::atomic<bool> quit{false};
  void (*job)(void*, int, int) = nullptr;
  void* jctx = nullptr;
  std::thread th;

  Pool() : th([this] {
    while (!quit.load()) {
      if (state.load(std::memory_order_acquire) == 1) {
        job(jctx, 1, 2);
        state.store(2, std::memory_order_release);
      } else {
        std::this_thread::yield();
      }
    }
  }) {}
  ~Pool() {
    quit = true;
    th.join();
  }

  static void begin(void*) {}
  static void end(void*) {}
  static void run(void* c, void (*j)(void*, int, int), void* jc) {
    Pool* p = static_cast<Pool*>(c);
    p->job = j;
    p->jctx = jc;
    p->state.store(1, std::memory_order_release);
    j(jc, 0, 2);
    while (p->state.load(std::memory_order_acquire) != 2) {}
    p->state.store(0, std::memory_order_relaxed);
  }
};

int64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<int> parse_ids(const char* s) {
  std::vector<int> v;
  const char* p = s;
  while (*p) {
    v.push_back(atoi(p));
    while (*p && *p != ',') p++;
    if (*p == ',') p++;
  }
  return v;
}

struct Options {
  const char* model_path = nullptr;
  std::vector<int> ids;
  const char* prompt = nullptr;  // text prompt, encoded with the tables embedded in the model file
  bool encode_only = false;
  int n_new = 0;
  int max_ctx = 512;
  bool ppl = false, verify = false, bench = false, text = false, pie_emul = false;
  const char* logits_out = nullptr;
  const char* kernel = nullptr;
  int threads = 1;
  int reloc_kb = 0;
  int aq = 0;  // 0 = float activations, G = int8 activations with groups of G
  float temp = 0.f, rep = 1.f;
  int top_k = 0;
  unsigned seed = 1;
};

void usage(const char* prog) {
  fprintf(stderr,
          "usage: %s model.bin [options]\n"
          "  --ids a,b,c        input token ids\n"
          "  --prompt TEXT      input text, tokenized by the embedded BPE tables (like the device does)\n"
          "  --encode           with --prompt: print the token ids and exit\n"
          "  --new N            generate N tokens after the input\n"
          "  --text             print generated tokens as text (needs an embedded vocabulary)\n"
          "  --ppl              print the perplexity of the input ids\n"
          "  --logits-out FILE  write the float32 logits of every input token\n"
          "  --verify           check the CRC32 of the model file\n"
          "  --bench            time every float kernel\n"
          "  --max-ctx N        context length (default 512)\n"
          "  --temp T --top-k K --rep R --seed S   sampling (default: greedy)\n"
          "  --kernel NAME      float kernel: ref, u4, r4, r8\n"
          "  --aq G             int8 activations with groups of G (32 enables the SIMD-style path)\n"
          "  --threads N        1 or 2 (second thread splits rows and attention heads)\n"
          "  --reloc-kb KB      copy up to KB kilobytes of weights into private buffers\n"
          "  --pie-emul         use the C emulation of the SIMD kernels and self-check them\n",
          prog);
}

bool parse_args(int argc, char** argv, Options* o) {
  if (argc < 2) return false;
  o->model_path = argv[1];
  for (int i = 2; i < argc; i++) {
    const char* a = argv[i];
    const bool has_val = i + 1 < argc;
    if (!strcmp(a, "--ids") && has_val) o->ids = parse_ids(argv[++i]);
    else if (!strcmp(a, "--prompt") && has_val) o->prompt = argv[++i];
    else if (!strcmp(a, "--encode")) o->encode_only = true;
    else if (!strcmp(a, "--new") && has_val) o->n_new = atoi(argv[++i]);
    else if (!strcmp(a, "--max-ctx") && has_val) o->max_ctx = atoi(argv[++i]);
    else if (!strcmp(a, "--ppl")) o->ppl = true;
    else if (!strcmp(a, "--verify")) o->verify = true;
    else if (!strcmp(a, "--logits-out") && has_val) o->logits_out = argv[++i];
    else if (!strcmp(a, "--temp") && has_val) o->temp = (float)atof(argv[++i]);
    else if (!strcmp(a, "--top-k") && has_val) o->top_k = atoi(argv[++i]);
    else if (!strcmp(a, "--rep") && has_val) o->rep = (float)atof(argv[++i]);
    else if (!strcmp(a, "--kernel") && has_val) o->kernel = argv[++i];
    else if (!strcmp(a, "--aq") && has_val) o->aq = atoi(argv[++i]);
    else if (!strcmp(a, "--pie-emul")) o->pie_emul = true;
    else if (!strcmp(a, "--bench")) o->bench = true;
    else if (!strcmp(a, "--text")) o->text = true;
    else if (!strcmp(a, "--threads") && has_val) o->threads = atoi(argv[++i]);
    else if (!strcmp(a, "--reloc-kb") && has_val) o->reloc_kb = atoi(argv[++i]);
    else if (!strcmp(a, "--seed") && has_val) o->seed = (unsigned)atoi(argv[++i]);
    else {
      fprintf(stderr, "unknown or incomplete option: %s\n", a);
      return false;
    }
  }
  return true;
}

// Reads the whole file into a 64-byte-aligned buffer (the model format requires it).
uint8_t* read_file(const char* path, size_t* size) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return nullptr;
  }
  fseek(f, 0, SEEK_END);
  *size = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t* blob = (uint8_t*)aligned_alloc(64, (*size + 63) & ~size_t(63));
  if (!blob || fread(blob, 1, *size, f) != *size) {
    fprintf(stderr, "could not read %s\n", path);
    free(blob);
    fclose(f);
    return nullptr;
  }
  fclose(f);
  return blob;
}

void run_bench(const ethoric::Model& model) {
  const auto& L = model.layers[0];
  const int rows = model.cfg.n_head * model.cfg.hd;
  std::vector<float> x(model.cfg.d, 0.5f), y(rows);
  for (int k = 0; k < ethoric::kNumKernels; k++) {
    const double us = ethoric::bench_kernel(ethoric::kKernels[k].fn, L.q.q, L.q.s, x.data(), y.data(), rows,
                                            model.cfg.d, 200, now_us);
    printf("kernel %-3s %8.1f us\n", ethoric::kKernels[k].name, us);
  }
}

void run_pie_selfcheck(const ethoric::Model& model) {
  const auto& L = model.layers[0];
  if (model.has_int4()) {
    printf("selfcheck pie-emul int4: gate %d up %d down %d\n", ethoric::a4_selfcheck(ethoric::mv_a4_pie, L.mlp_gate, 64),
           ethoric::a4_selfcheck(ethoric::mv_a4_pie, L.mlp_up, 64), ethoric::a4_selfcheck(ethoric::mv_a4_pie, L.mlp_down, 64));
  } else {
    printf("selfcheck pie-emul: gate %d down %d embed %d\n", ethoric::a8_selfcheck(ethoric::mv_a8_pie, L.mlp_gate, 64),
           ethoric::a8_selfcheck(ethoric::mv_a8_pie, L.mlp_down, 64),
           ethoric::a8_selfcheck(ethoric::mv_a8_pie, model.embed, 256));
  }
}

// Adds -log p(next token) to *nll, given the logits that predicted it.
void accumulate_nll(const float* logits, int vocab, int target, double* nll) {
  float mx = logits[0];
  for (int v = 1; v < vocab; v++) if (logits[v] > mx) mx = logits[v];
  double s = 0;
  for (int v = 0; v < vocab; v++) s += exp((double)logits[v] - mx);
  *nll -= (double)logits[target] - mx - log(s);
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  if (!parse_args(argc, argv, &o)) {
    usage(argv[0]);
    return 2;
  }
  if (o.ids.empty() && !o.prompt && !o.verify && !o.bench) {
    fprintf(stderr, "missing --ids or --prompt\n");
    return 2;
  }

  size_t size = 0;
  uint8_t* blob = read_file(o.model_path, &size);
  if (!blob) return 1;

  ethoric::Model model;
  if (!model.load(blob, size)) {
    fprintf(stderr, "invalid or truncated model file\n");
    return 1;
  }
  if (o.prompt) {
    if (!model.has_tokenizer()) {
      fprintf(stderr, "this model file has no embedded tokenizer tables: re-export it\n");
      return 1;
    }
    o.ids.resize(strlen(o.prompt) + 8);
    const int n = model.encode(o.prompt, strlen(o.prompt), o.ids.data(), (int)o.ids.size());
    if (n < 0) return 1;
    o.ids.resize(n);
    if (o.encode_only) {
      for (size_t i = 0; i < o.ids.size(); i++) printf(i ? ",%d" : "%d", o.ids[i]);
      printf("\n");
      return 0;
    }
  }
  if (o.verify) {
    const bool good = model.verify_crc();
    printf("crc32: %s\n", good ? "ok" : "BAD (corrupted file or incomplete write)");
    if (o.ids.empty()) return good ? 0 : 3;
  }
  if (o.kernel && !ethoric::set_kernel(o.kernel)) {
    fprintf(stderr, "unknown kernel: %s (ref, u4, r4, r8)\n", o.kernel);
    return 2;
  }
  if (o.bench) {
    run_bench(model);
    if (o.ids.empty()) return 0;
  }
  if (o.reloc_kb > 0) {
    const size_t n = model.relocate([](size_t b) -> void* { return aligned_alloc(64, (b + 63) & ~size_t(63)); },
                                    size_t(o.reloc_kb) * 1024);
    fprintf(stderr, "relocated %.2f MiB into private copies\n", n / 1048576.0);
  }
  ethoric::g_aq_group = o.aq;
  if (o.pie_emul) {
    ethoric::g_a8 = ethoric::mv_a8_pie;
    ethoric::g_a4 = ethoric::mv_a4_pie;
    run_pie_selfcheck(model);
  }

  ethoric::Runner run(model, o.max_ctx);
  if (!run.ok()) {
    fprintf(stderr, "not enough memory for the requested context\n");
    return 1;
  }
  Pool* pool = nullptr;
  ethoric::Parallel par{};
  if (o.threads >= 2) {
    pool = new Pool();
    par = {pool, Pool::begin, Pool::end, Pool::run};
    run.set_parallel(&par);
  }
  fprintf(stderr, "model: d=%d layers=%d vocab=%d | runtime RAM ~%.2f MiB (context %d)\n", model.cfg.d,
          model.cfg.n_layer, model.cfg.vocab, run.ram_bytes() / 1048576.0, run.max_ctx());

  // Feed the input ids, optionally dumping logits and scoring the perplexity.
  FILE* lo = o.logits_out ? fopen(o.logits_out, "wb") : nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  const float* logits = nullptr;
  double nll = 0.0;
  int n_scored = 0;
  for (size_t i = 0; i < o.ids.size(); i++) {
    if (o.ppl && i > 0) {  // `logits` still holds the previous token's output: it predicts ids[i]
      accumulate_nll(logits, model.cfg.vocab, o.ids[i], &nll);
      n_scored++;
    }
    logits = run.step(o.ids[i]);
    if (!logits) {
      fprintf(stderr, "context exhausted\n");
      return 1;
    }
    if (lo) fwrite(logits, sizeof(float), model.cfg.vocab, lo);
  }
  if (lo) fclose(lo);
  if (o.ppl) printf("perplexity: %.4f (%d predicted tokens)\n", exp(nll / n_scored), n_scored);

  // Generate.
  std::vector<int> out = o.ids;
  ethoric::Sampler sampler(model.cfg.vocab, 64, o.seed);
  sampler.temp = o.temp;
  sampler.top_k = o.top_k;
  sampler.rep_penalty = o.rep;
  for (int n = 0; n < o.n_new; n++) {
    const int next = sampler.pick(logits, out.data(), (int)out.size());
    out.push_back(next);
    if (next == model.cfg.eos) break;
    logits = run.step(next);
    if (!logits) break;
  }
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (o.n_new > 0) {
    printf("ids:");
    for (int t : out) printf(" %d", t);
    printf("\n");
    if (o.text) {
      if (!model.has_vocab()) {
        fprintf(stderr, "this model file has no embedded vocabulary: re-export it\n");
      } else {
        printf("text: ");
        for (size_t i = o.ids.size(); i < out.size(); i++) {
          int len;
          const uint8_t* b = model.token_bytes(out[i], &len);
          if (b) fwrite(b, 1, len, stdout);
        }
        printf("\n");
      }
    }
  }
  delete pool;
  fprintf(stderr, "%d tokens in %.3f s (%.2f ms/token on the host; not representative of the ESP32)\n", run.pos(), dt,
          dt / run.pos() * 1e3);
  free(blob);
  return 0;
}
