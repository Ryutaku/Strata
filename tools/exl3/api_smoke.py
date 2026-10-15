"""Start a localhost-only Strata API child and check two concurrent arithmetic requests.

Requires a server-format config (exe, args, cwd, tokenizer, model_name, parallel=2).
Does not alter services. Retains the complete responses and logs, and stops only its child group.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18185)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", a.port))
    cfg = json.loads(a.config.read_text())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{a.port}"

    def request(path, data=None):
        headers = {"Content-Type": "application/json"}
        # The optional API key stays in memory; no request headers are written to the report.
        if cfg.get("api_key"): headers["Authorization"] = "Bearer " + cfg["api_key"]
        req = urllib.request.Request(base + path, data=None if data is None else json.dumps(data).encode(), headers=headers)
        with opener.open(req, timeout=180) as response: return json.load(response)

    report = {"passed": False}
    root = Path(__file__).resolve().parents[2]
    with (a.output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "strata", "--config",
            str(a.config.resolve()), "--host", "127.0.0.1", "--port", str(a.port)],
            cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name != "nt"))
        try:
            deadline = time.monotonic() + 240
            while True:
                if process.poll() is not None: raise RuntimeError("server exited before readiness; see server.log")
                try:
                    health = request("/health")
                    if health.get("loaded"): break
                except (OSError, ValueError): pass
                if time.monotonic() > deadline: raise TimeoutError("server did not become ready")
                time.sleep(0.25)
            barrier = threading.Barrier(2)

            def arithmetic(value):
                barrier.wait(timeout=10)
                return request("/v1/chat/completions", {"model": cfg.get("model_name", "strata-exl3"),
                    "messages": [{"role": "user", "content": f"What is 17 + {value}? Reply with the number only."}],
                    "max_tokens": 64, "temperature": 0, "reasoning_effort": "none"})

            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(arithmetic, (26, 27)))
            answers = [r["choices"][0]["message"].get("content", "") for r in responses]
            report.update(health=health, responses=responses, answers=answers, wall_s=time.monotonic()-started,
                passed=all(re.search(rf"(?<!\d){expected}(?!\d)", answer) for expected, answer in zip((43,44),answers)))
            print(json.dumps({k:v for k,v in report.items() if k != "responses"}, indent=2), flush=True)
            return 0 if report["passed"] else 1
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            if process.poll() is None:
                if os.name == "nt": process.terminate()
                else: os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name == "nt": process.kill()
                    else: os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
