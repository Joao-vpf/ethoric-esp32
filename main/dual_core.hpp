// Two-core executor for ethoric::Parallel: core 0 (the main task) and core 1 (task "ethoric_w")
// split the rows of every matrix. Between tokens the worker SLEEPS (blocked on a notification) so the IDLE1 task
// runs and feeds the watchdog; during a token it spins waiting for work (microsecond latency).
// The flags live in static variables (internal RAM), never in PSRAM.
#pragma once
#include <atomic>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "ethoric.hpp"

class DualCore {
 public:
  // Creates the task on core 1. Returns false on failure (the caller then continues with one core).
  bool start() {
    return xTaskCreatePinnedToCore(&DualCore::task_entry, "ethoric_w", 4096, this, configMAX_PRIORITIES - 2, &task_, 1) ==
           pdPASS;
  }
  ethoric::Parallel* parallel() { return &par_; }

 private:
  static void task_entry(void* a) { static_cast<DualCore*>(a)->loop(); }
  void loop() {
    for (;;) {
      ulTaskNotifyTake(pdTRUE, portMAX_DELAY);      // sleep until the next token
      while (active_.load(std::memory_order_acquire)) {
        if (ready_.load(std::memory_order_acquire)) {
          job_(jctx_, 1, 2);
          ready_.store(0, std::memory_order_relaxed);
          done_.store(1, std::memory_order_release);
        }
      }
    }
  }
  static void s_begin(void* c) {
    DualCore* s = static_cast<DualCore*>(c);
    s->active_.store(1, std::memory_order_release);
    xTaskNotifyGive(s->task_);
  }
  static void s_end(void* c) { static_cast<DualCore*>(c)->active_.store(0, std::memory_order_release); }
  static void s_run(void* c, void (*job)(void*, int, int), void* jctx) {
    DualCore* s = static_cast<DualCore*>(c);
    s->job_ = job;
    s->jctx_ = jctx;
    s->done_.store(0, std::memory_order_relaxed);
    s->ready_.store(1, std::memory_order_release);
    job(jctx, 0, 2);
    while (!s->done_.load(std::memory_order_acquire)) {}
  }

  TaskHandle_t task_ = nullptr;
  std::atomic<int> active_{0}, ready_{0}, done_{0};
  void (*job_)(void*, int, int) = nullptr;
  void* jctx_ = nullptr;
  ethoric::Parallel par_{this, &DualCore::s_begin, &DualCore::s_end, &DualCore::s_run};
};
