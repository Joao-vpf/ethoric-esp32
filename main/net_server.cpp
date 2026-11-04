// WiFi and HTTP front end: a small llama.cpp-style server.
//
//   POST /completion   {"prompt": "...", "n_predict": 64, "temperature": 0.8, "top_k": 20,
//                       "repeat_penalty": 1.2, "seed": 0}
//     -> {"content": "...", "stop": true, "tokens_evaluated": N, "tokens_predicted": M,
//         "timings": {"prompt_n", "prompt_ms", "predicted_n", "predicted_ms",
//                     "predicted_per_token_ms", "predicted_per_second"}}
//   GET  /health       {"status": "ok"}
//   GET  /             chat web page (works on a phone browser)
//
// The reply is sent when generation finishes. Requests are processed one at a time; a second request gets 503.
#include <stdio.h>
#include <string.h>

#include <string>

#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "mini_json.hpp"
#include "nvs_flash.h"
#include "dns_captive.hpp"
#include "engine.hpp"
#include "wifi_config.h"   // generated from .env

static const char* TAG = "ethoric-net";
static const int kMaxBody = 8192;
static bool g_ap_mode = false;   // true when the board runs its own access point (captive portal active)

static esp_err_t send_json(httpd_req_t* req, const char* status, const std::string& body) {
  httpd_resp_set_status(req, status);
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, body.c_str(), body.size());
}

static esp_err_t send_error(httpd_req_t* req, int code, const std::string& msg) {
  const char* status = code == 400 ? "400 Bad Request" : code == 413 ? "413 Payload Too Large"
                       : code == 503 ? "503 Service Unavailable" : "500 Internal Server Error";
  return send_json(req, status,
                   "{\"error\":{\"code\":" + std::to_string(code) + ",\"message\":\"" +
                       ethoric::json_escape(msg.data(), msg.size()) + "\"}}");
}

static esp_err_t completion_handler(httpd_req_t* req) {
  if (req->content_len == 0) return send_error(req, 400, "empty body: expected a JSON object with \"prompt\"");
  if (req->content_len > (size_t)kMaxBody) return send_error(req, 413, "body too large");
  std::string body(req->content_len, '\0');
  size_t got = 0;
  while (got < body.size()) {
    const int r = httpd_req_recv(req, &body[got], body.size() - got);
    if (r == HTTPD_SOCK_ERR_TIMEOUT) continue;
    if (r <= 0) return ESP_FAIL;
    got += r;
  }

  ethoric::JsonObject j;
  std::string prompt;
  if (!j.parse(body.data(), body.size()) || !j.get_string("prompt", &prompt)) {
    return send_error(req, 400, "invalid JSON or missing string field \"prompt\"");
  }
  GenParams gp;
  double v = 0;
  if (j.get_number("n_predict", &v) && v >= 0) gp.max_new = (int)v;
  if (j.get_number("temperature", &v) && v >= 0) gp.temp = (float)v;
  if (j.get_number("top_k", &v) && v >= 0) gp.top_k = (int)v;
  if (j.get_number("repeat_penalty", &v) && v > 0) gp.rep = (float)v;
  if (j.get_number("seed", &v) && v > 0) gp.seed = (uint32_t)v;
  if (j.get_number("echo", &v)) gp.echo = v != 0;

  std::string text, err;
  GenStats st;
  const int rc = engine_complete(prompt, gp, &text, &st, &err);
  if (rc != 0) return send_error(req, rc, err);

  const double per_tok = st.tokens > 0 ? st.predicted_ms / st.tokens : 0.0;
  char t[320];
  snprintf(t, sizeof(t),
           "\"stop\":true,\"stopped_eos\":%s,\"tokens_evaluated\":%d,\"tokens_predicted\":%d,"
           "\"timings\":{\"prompt_n\":%d,\"prompt_ms\":%.1f,\"predicted_n\":%d,\"predicted_ms\":%.1f,"
           "\"predicted_per_token_ms\":%.1f,\"predicted_per_second\":%.2f}}",
           st.hit_eos ? "true" : "false", st.prompt_tokens, st.tokens, st.prompt_tokens, st.prompt_ms, st.tokens,
           st.predicted_ms, per_tok, per_tok > 0 ? 1000.0 / per_tok : 0.0);
  return send_json(req, "200 OK", "{\"content\":\"" + ethoric::json_escape(text.data(), text.size()) + "\"," + t);
}

static esp_err_t health_handler(httpd_req_t* req) { return send_json(req, "200 OK", "{\"status\":\"ok\"}"); }

// Chat page served at "/" (main/web/index.html, embedded into the firmware by main/CMakeLists.txt).
extern const char index_html_start[] asm("_binary_index_html_start");

static esp_err_t redirect_handler(httpd_req_t* req, httpd_err_code_t);

// True when the request was addressed to the board itself (Host: 192.168.4.1), false for any other name that the
// captive DNS pointed here (e.g. "Host: google.com").
static bool addressed_to_board(httpd_req_t* req) {
  const size_t n = httpd_req_get_hdr_value_len(req, "Host");
  if (n == 0 || n > 63) return true;   // no Host header (or an odd one): serve the page
  char host[64];
  if (httpd_req_get_hdr_value_str(req, "Host", host, sizeof(host)) != ESP_OK) return true;
  return strncmp(host, "192.168.4.1", 11) == 0 && (host[11] == 0 || host[11] == ':');
}

static esp_err_t index_handler(httpd_req_t* req) {
  if (g_ap_mode && !addressed_to_board(req)) return redirect_handler(req, HTTPD_404_NOT_FOUND);   // one canonical address
  httpd_resp_set_type(req, "text/html; charset=utf-8");
  return httpd_resp_send(req, index_html_start, strlen(index_html_start));
}

// ---- WiFi ------------------------------------------------------------------------------------------------------------
static void wifi_event(void*, esp_event_base_t base, int32_t id, void* data) {
  if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
  } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
    ESP_LOGW(TAG, "WiFi disconnected, retrying");
    esp_wifi_connect();
  } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
    const ip_event_got_ip_t* e = (const ip_event_got_ip_t*)data;
    ESP_LOGW(TAG, "connected: http://" IPSTR ":%d/completion", IP2STR(&e->ip_info.ip), ETHORIC_HTTP_PORT);
  }
}

static void wifi_start() {
  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    nvs_flash_erase();
    nvs_flash_init();
  }
  esp_netif_init();
  esp_event_loop_create_default();
  const bool station = ETHORIC_WIFI_SSID[0] != 0;
  g_ap_mode = !station;
  esp_netif_t* netif = station ? esp_netif_create_default_wifi_sta() : esp_netif_create_default_wifi_ap();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  esp_wifi_init(&cfg);
  esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event, nullptr, nullptr);
  esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event, nullptr, nullptr);

  wifi_config_t wc = {};
  if (station) {
    strlcpy((char*)wc.sta.ssid, ETHORIC_WIFI_SSID, sizeof(wc.sta.ssid));
    strlcpy((char*)wc.sta.password, ETHORIC_WIFI_PASSWORD, sizeof(wc.sta.password));
    wc.sta.threshold.authmode = ETHORIC_WIFI_PASSWORD[0] ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wc);
  } else {
    strlcpy((char*)wc.ap.ssid, ETHORIC_AP_SSID, sizeof(wc.ap.ssid));
    wc.ap.ssid_len = strlen(ETHORIC_AP_SSID);
    wc.ap.channel = 1;
    wc.ap.max_connection = 4;
    if (strlen(ETHORIC_AP_PASSWORD) >= 8) {
      strlcpy((char*)wc.ap.password, ETHORIC_AP_PASSWORD, sizeof(wc.ap.password));
      wc.ap.authmode = WIFI_AUTH_WPA2_PSK;
    } else {
      wc.ap.authmode = WIFI_AUTH_OPEN;
    }
    esp_wifi_set_mode(WIFI_MODE_AP);
    esp_wifi_set_config(WIFI_IF_AP, &wc);
    ESP_LOGW(TAG, "access point \"%s\": open any web address, or http://192.168.4.1/", ETHORIC_AP_SSID);
  }
  esp_wifi_start();
  if (g_ap_mode) {
    // Make the DHCP server hand out the board itself as the DNS server, so every name lookup reaches dns_task().
    esp_netif_dns_info_t dns = {};
    dns.ip.u_addr.ip4.addr = ESP_IP4TOADDR(192, 168, 4, 1);
    dns.ip.type = ESP_IPADDR_TYPE_V4;
    uint8_t offer_dns = 0x02;   // OFFER_DNS in lwIP's dhcpserver.h
    esp_netif_dhcps_stop(netif);
    esp_netif_dhcps_option(netif, ESP_NETIF_OP_SET, ESP_NETIF_DOMAIN_NAME_SERVER, &offer_dns, sizeof(offer_dns));
    esp_netif_set_dns_info(netif, ESP_NETIF_DNS_MAIN, &dns);
    // RFC 8910 "captive portal URI" (DHCP option 114): iOS 14+ and Android 11+ use it to open the page reliably.
    static const char kPortalUri[] = "http://192.168.4.1/";
    esp_netif_dhcps_option(netif, ESP_NETIF_OP_SET, ESP_NETIF_CAPTIVEPORTAL_URI, (void*)kPortalUri, sizeof(kPortalUri) - 1);
    esp_netif_dhcps_start(netif);
  }
  esp_wifi_set_ps(WIFI_PS_NONE);  // lowest latency; the radio is not the bottleneck
}

// ---- Captive portal (access point mode) -----------------------------------------------------------------------------
// Every DNS name resolves to the board and every unknown URL redirects to the chat page, so phones open it by themselves
// ("sign in to network") and any plain-HTTP address typed in a browser lands there. HTTPS sites cannot be redirected.
static void dns_task(void*) {
  const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(53);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  if (sock < 0 || bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
    ESP_LOGE(TAG, "DNS socket failed");
    vTaskDelete(nullptr);
    return;
  }
  const uint8_t ip[4] = {192, 168, 4, 1};
  uint8_t in[512], out[512 + 16];
  for (;;) {
    sockaddr_in from = {};
    socklen_t flen = sizeof(from);
    const int n = recvfrom(sock, in, sizeof(in), 0, (sockaddr*)&from, &flen);
    if (n <= 0) continue;
    const size_t m = ethoric::dns_build_reply(in, (size_t)n, ip, out, sizeof(out));
    if (m) sendto(sock, out, m, 0, (sockaddr*)&from, flen);
  }
}

static esp_err_t redirect_handler(httpd_req_t* req, httpd_err_code_t) {
  httpd_resp_set_status(req, "302 Found");
  httpd_resp_set_hdr(req, "Location", "http://192.168.4.1/");
  httpd_resp_set_type(req, "text/plain");
  return httpd_resp_send(req, "Redirecting to http://192.168.4.1/", HTTPD_RESP_USE_STRLEN);
}

void net_start(int max_ctx) {
  (void)max_ctx;
  wifi_start();

  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = g_ap_mode ? 80 : ETHORIC_HTTP_PORT;   // captive-portal checks use port 80
  cfg.stack_size = 12288;       // generation runs inside the handler
  cfg.core_id = 0;              // core 1 belongs to the compute worker
  cfg.max_open_sockets = 5;      // iOS opens several parallel connections; must stay <= CONFIG_LWIP_MAX_SOCKETS - 4
  cfg.lru_purge_enable = true;   // drop the oldest idle connection instead of refusing new ones
  cfg.recv_wait_timeout = 10;
  cfg.send_wait_timeout = 10;
  httpd_handle_t srv = nullptr;
  if (httpd_start(&srv, &cfg) != ESP_OK) {
    ESP_LOGE(TAG, "could not start the HTTP server");
    return;
  }
  const httpd_uri_t routes[] = {
      {.uri = "/completion", .method = HTTP_POST, .handler = completion_handler, .user_ctx = nullptr},
      {.uri = "/health", .method = HTTP_GET, .handler = health_handler, .user_ctx = nullptr},
      {.uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = nullptr},
  };
  for (const httpd_uri_t& r : routes) httpd_register_uri_handler(srv, &r);
  if (g_ap_mode) {
    httpd_register_err_handler(srv, HTTPD_404_NOT_FOUND, redirect_handler);
    xTaskCreatePinnedToCore(dns_task, "dns", 4096, nullptr, 6, nullptr, 0);
  }
}
