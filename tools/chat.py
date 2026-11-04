#!/usr/bin/env python3
"""llama.cpp-style client for the ESP32: send a prompt as text, get the completion back when it is finished.

The board tokenizes the prompt itself, so no tokenizer files are needed here.

  # over WiFi (HTTP server on the board; the IP address is printed at boot)
  python tools/chat.py --url http://192.168.4.1 -p "The lighthouse stood at" -n 80

  # over the USB serial port (needs: pip install pyserial)
  python tools/chat.py --port /dev/ttyACM0 -p "The lighthouse stood at" -n 80

  python tools/chat.py --url http://192.168.4.1 -i       # interactive, one prompt per line

Options mirror llama.cpp (-p, -n, --temp, --top-k, --repeat-penalty, -s). The reply is printed only when the device
finishes; timing statistics go to stderr. The firmware must be flashed with a model file that embeds the tokenizer
tables (tools/export_model.py does this by default).
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request


def complete_http(url, prompt, args):
    body = json.dumps({"prompt": prompt, "n_predict": args.n_predict, "temperature": args.temp, "top_k": args.top_k,
                       "repeat_penalty": args.repeat_penalty, "seed": args.seed}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/completion", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {url}: {e.reason}")
    t = d.get("timings", {})
    return d["content"], {"prompt_tokens": t.get("prompt_n", 0), "prompt_ms": t.get("prompt_ms", 0.0),
                          "tokens": t.get("predicted_n", 0), "ms_per_token": t.get("predicted_per_token_ms", 0.0)}


class Serial:
    def __init__(self, port, baud, boot_timeout):
        import serial  # imported lazily: only needed for --port
        self.ser = serial.Serial(port, baud, timeout=0.1)
        deadline, buf = time.time() + boot_timeout, b""
        while time.time() < deadline:
            buf += self.ser.read(self.ser.in_waiting or 1)
            if b"@@READY" in buf:
                print(buf[buf.index(b"@@READY"):].split(b"\n")[0].decode(errors="replace"), file=sys.stderr)
                return
        sys.exit("timeout waiting for the device: is it flashed with CHAT_MODE 1? press RESET and retry")

    def complete(self, prompt, args):
        esc = prompt.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        self.ser.reset_input_buffer()
        line = f"C n={args.n_predict} t={args.temp} k={args.top_k} r={args.repeat_penalty} s={args.seed} |{esc}\n"
        self.ser.write(line.encode("utf-8"))
        buf, deadline = b"", time.time() + args.timeout
        while time.time() < deadline:
            buf += self.ser.read(self.ser.in_waiting or 1)
            for marker in (b"@@END", b"@@ERR"):
                if marker in buf:
                    head, _, tail = buf.partition(marker)
                    while b"\n" not in tail and time.time() < deadline:  # wait for the end of the status line
                        tail += self.ser.read(self.ser.in_waiting or 1)
                    status = (marker + tail.split(b"\n")[0]).decode(errors="replace")
                    if marker == b"@@ERR":
                        sys.exit(status)
                    return head.decode("utf-8", errors="replace"), {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", status)}
        sys.exit("timeout waiting for the reply (raise --timeout)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--url", help="base URL of the board's HTTP server, e.g. http://192.168.4.1")
    where.add_argument("--port", help="serial port, e.g. /dev/ttyACM0 (default when --url is not given)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("-p", "--prompt", help="prompt text; omit with -i for interactive mode")
    ap.add_argument("-i", "--interactive", action="store_true", help="read prompts from the terminal")
    ap.add_argument("-n", "--n-predict", type=int, default=64, help="maximum new tokens (default: %(default)s)")
    ap.add_argument("--temp", type=float, default=0.8, help="temperature, 0 = greedy (default: %(default)s)")
    ap.add_argument("--top-k", type=int, default=20, help="top-k cut-off (default: %(default)s)")
    ap.add_argument("--repeat-penalty", type=float, default=1.2, help="repetition penalty (default: %(default)s)")
    ap.add_argument("-s", "--seed", type=int, default=0, help="random seed, 0 = random (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds to wait for a reply (default: %(default)s)")
    ap.add_argument("--boot-timeout", type=float, default=30.0)
    args = ap.parse_args()
    if not args.prompt and not args.interactive:
        ap.error("provide -p PROMPT or -i")

    if args.url:
        complete = lambda p: complete_http(args.url, p, args)
    else:
        complete = lambda p, dev=Serial(args.port or "/dev/ttyACM0", args.baud, args.boot_timeout): dev.complete(p, args)

    def run_one(prompt):
        t0 = time.time()
        text, st = complete(prompt)
        print(text)
        mpt = st.get("ms_per_token", 0.0)
        print(f"\nprompt eval: {st.get('prompt_tokens', 0):.0f} tokens in {st.get('prompt_ms', 0):.0f} ms | "
              f"generation: {st.get('tokens', 0):.0f} tokens, {mpt:.1f} ms/token ({1000 / mpt if mpt else 0:.2f} tokens/s) | "
              f"total {time.time() - t0:.1f} s", file=sys.stderr)

    if args.prompt:
        run_one(args.prompt)
    if args.interactive:
        try:
            while True:
                p = input("\n> ")
                if p.strip():
                    run_one(p)
        except (EOFError, KeyboardInterrupt):
            print()


if __name__ == "__main__":
    main()
