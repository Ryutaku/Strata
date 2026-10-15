"""Reproducible EXL3 GEN/BGEN benchmark; keeps raw engine logs and exact prompt/output IDs.

No service management. Config uses the same args/tokenizer/cwd fields as serve/server.py.
--batch sends two independent prompts together. --needle checks three distant synthetic keys;
otherwise requests a long answer for a meaningful decode measurement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from strata_tokenizer import Tokenizer


def load_tokenizer(path):
    root = Path(path)
    vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    if isinstance(vocab, dict):
        tokens = [None] * len(vocab)
        for token, index in vocab.items(): tokens[index] = token
    else:
        tokens = vocab
    return Tokenizer(tokens, (root / "merges.txt").read_text(encoding="utf-8").splitlines(),
                     json.loads((root / "token_type.json").read_text()))


def prompt(tk, count, slot, needle):
    keys = [f"STRATA{slot}{label}7319" for label in ("ALPHA", "MIDDLE", "OMEGA")]
    head = tk.encode(f"<|im_start|>user\nRun {slot}: remember FIRST={keys[0]}.\n", parse_special=True)
    mid = tk.encode(f"\nMIDDLE={keys[1]}.\n")
    question = ("Return FIRST, MIDDLE and LAST exactly, and nothing else." if needle else
                "Write a detailed technical essay of at least 1500 words about database indexing and its tradeoffs. "
                "Cover B-trees, LSM trees, concurrency, recovery and storage costs with concrete examples. "
                "Do not summarise the records above.")
    tail = tk.encode(f"\nLAST={keys[2]}.\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n",
                     parse_special=True)
    remaining = count - len(head) - len(mid) - len(tail)
    if remaining < 0: raise ValueError("prompt is too short for the test instructions")
    filler = []
    index = 0
    while len(filler) < remaining:
        text = "".join(f"Record {i}: station {(i*31)%997} measured {(i*71)%1009}; the committee checked its budget "
                       "and requested a written report for the next meeting.\n" for i in range(index, index+1024))
        filler.extend(tk.encode(text))
        index += 1024
    half = remaining // 2
    return head + filler[:half] + mid + filler[half:remaining] + tail, keys


class Engine:
    def __init__(self, command, cfg, output):
        self.lines = queue.Queue()
        self.stdout = (output / "engine.stdout.log").open("w", encoding="utf-8")
        self.stderr = (output / "engine.stderr.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in cfg.get("env", {}).items()})
        if cfg.get("lib_dirs"):
            env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
        self.process = subprocess.Popen(command, cwd=cfg.get("cwd"), env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr, text=True, encoding="utf-8", bufsize=1)
        def read():
            for line in self.process.stdout:
                self.stdout.write(line); self.stdout.flush()
                self.lines.put(line.rstrip())
            self.lines.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def receive(self, deadline):
        remaining = deadline-time.monotonic()
        if remaining <= 0: raise TimeoutError("engine deadline exceeded")
        try: line = self.lines.get(timeout=remaining)
        except queue.Empty: raise TimeoutError("engine stopped producing output")
        if line is None: raise RuntimeError(f"engine ended with status {self.process.poll()}; see engine.stderr.log")
        if line.startswith(("ERR", "FATAL")): raise RuntimeError(line)
        return line

    def send(self, line):
        self.process.stdin.write(line+"\n"); self.process.stdin.flush()

    def close(self):
        if self.process.poll() is None:
            try: self.send("QUIT")
            except (OSError, BrokenPipeError): pass
            try: self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try: self.process.wait(timeout=10)
                except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
        self.reader.join(timeout=5)
        self.stdout.close(); self.stderr.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--prompt-tokens", type=int, default=32768)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--needle", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    tk = load_tokenizer(cfg["tokenizer"])
    prompts, keys = zip(*(prompt(tk, a.prompt_tokens, slot, a.needle) for slot in range(2 if a.batch else 1)))
    (a.output / "prompts.json").write_text(json.dumps(prompts), encoding="utf-8")
    command = [str(a.engine.resolve()), "--serve", *cfg["args"]]
    report = {"command": command, "environment": cfg.get("env", {}), "prompt_tokens_each": a.prompt_tokens,
              "max_new": a.max_new, "batch": a.batch, "needle": a.needle,
              "engine_sha256": hashlib.sha256(a.engine.read_bytes()).hexdigest(), "outputs": []}
    eng = Engine(command, cfg, a.output)
    deadline = time.monotonic()+a.timeout
    try:
        while True:
            line = eng.receive(deadline)
            if line.startswith("INFO "): report["info"] = line
            if line.startswith("READY "): break
        start = time.monotonic()
        for slot, ids in enumerate(prompts):
            prefix = f"BGEN {slot}" if a.batch else "GEN"
            eng.send(f"{prefix} {a.max_new} temperature=0 seed=7319 " + ",".join(map(str, ids)))
        outputs = [[] for _ in prompts]
        finished = set()
        current = 0
        done_lines = []
        while len(finished) < len(prompts):
            line = eng.receive(deadline)
            if line.startswith("T "): outputs[current].append(int(line.split()[1]))
            elif line.startswith("BT "):
                _, slot, token = line.split(); outputs[int(slot)].append(int(token))
            elif line.startswith("BADM "):
                fields = line.split(); current = min(int(fields[1])+1, len(prompts)-1)
                if fields[2] == "0": finished.add(int(fields[1]))
            elif line.startswith("BDONE "):
                finished.add(int(line.split()[1])); done_lines.append(line)
            elif line.startswith("DONE "):
                done_lines.append(line)
                if not a.batch: finished.add(0)
            if line.startswith(("PP ", "DONE ", "BDONE ")): print(line, flush=True)
        report["wall_s"] = time.monotonic()-start
        report["done"] = done_lines
        if not a.batch and done_lines:
            fields = done_lines[-1].split()
            generated, read = int(fields[1]), int(fields[2])
            prefill_ms, decode_ms = float(fields[3]), float(fields[4])
            report["prefill_tokens_s"] = read * 1000 / prefill_ms if prefill_ms else None
            report["decode_tokens_s"] = generated * 1000 / decode_ms if decode_ms else None
        report["total_tokens"] = sum(map(len, outputs))
        report["requested_tokens_completed"] = all(len(ids) >= a.max_new for ids in outputs)
        report["overall_tokens_s"] = report["total_tokens"]/report["wall_s"]
        passed = True
        for slot, ids in enumerate(outputs):
            text = tk.decode(ids)
            found = [key in text for key in keys[slot]]
            report["outputs"].append({"ids": ids, "text": text, "keys": keys[slot], "keys_found": found})
            if a.needle: passed = passed and all(found)
        report["passed"] = passed
        (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("outputs", "command")}, indent=2), flush=True)
        return 0 if passed else 1
    finally:
        eng.close()


if __name__ == "__main__":
    raise SystemExit(main())
