"""Compare two EXL3 slots with solo execution across cancellation, continuation and slot reuse.

Uses the engine protocol directly. Saves raw logs and token IDs, and terminates only its own child.
Run with a config enabling --batch 2, --batch-mtp and explicit --layer-split boundaries.
"""
import argparse
import json
from pathlib import Path
import time

from bench import Engine, load_tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text())
    tk = load_tokenizer(cfg["tokenizer"])
    cfg.setdefault("env", {})["STRATA_IQ_MT_MIN"] = "1"
    eng = Engine([str(a.engine.resolve()), "--serve", *cfg["args"]], cfg, a.output)
    deadline = time.monotonic() + 900
    report = {"checks": {}, "tokens": {}}

    def chat(question):
        return tk.encode(f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n",
                         parse_special=True)

    def send(prefix, limit, ids):
        eng.send(f"{prefix} {limit} temperature=0 seed=7319 " + ",".join(map(str, ids)))

    def solo(ids, count):
        send("GEN", count, ids)
        tokens = []
        while True:
            line = eng.receive(deadline)
            if line.startswith("T "): tokens.append(int(line.split()[1]))
            if line.startswith("DONE "): return tokens

    def check(name, condition):
        report["checks"][name] = bool(condition)
        print(f"{name}: {'PASS' if condition else 'FAIL'}", flush=True)

    try:
        while not eng.receive(deadline).startswith("READY "): pass
        A = chat("Write a Python function that merges overlapping intervals, then explain it in detail.")
        B = chat("Compare TCP and QUIC: handshake, congestion control, multiplexing and recovery in detail.")
        C = chat("Explain why a database index can speed up reads but slow down writes. Give examples.")
        refs = [solo(A, 64), solo(B, 64)]
        B2 = B + refs[1] + chat("Continue your explanation with one concrete example.")
        ref_B2, ref_C = solo(B2, 32), solo(C, 32)
        check("references_reached_length", all(len(r) == 64 for r in refs))
        send("BGEN 0", 128, A)
        send("BGEN 1", 64, B)
        outputs = [[], []]
        done, reasons = set(), {}
        admitting, stopped = 0, False
        while len(done) < 2:
            line = eng.receive(deadline)
            f = line.split()
            if line.startswith("T "): outputs[admitting].append(int(f[1]))
            elif line.startswith("BADM "):
                admitting = min(int(f[1]) + 1, 1)
                if f[2] == "0": done.add(int(f[1]))
            elif line.startswith("BT "): outputs[int(f[1])].append(int(f[2]))
            elif line.startswith("BDONE "):
                done.add(int(f[1])); reasons[int(f[1])] = f[3]
            if len(outputs[0]) >= 16 and not stopped:
                eng.send("BSTOP 0"); stopped = True
        check("cancel_acknowledged", reasons.get(0) == "cancel" and 16 <= len(outputs[0]) < 64)
        check("other_slot_unchanged", outputs[1] == refs[1])
        resumed = solo(A + outputs[0], max(1, 64 - len(outputs[0])))
        check("cancelled_slot_continues_exactly", outputs[0] + resumed == refs[0])
        next_turn = solo(B2, 32)
        check("other_slot_next_turn_exact", next_turn == ref_B2)
        send("BGEN 0", 32, C)
        reused = []
        while True:
            line = eng.receive(deadline)
            if line.startswith("T "): reused.append(int(line.split()[1]))
            elif line.startswith("BT 0 "): reused.append(int(line.split()[2]))
            elif line.startswith("BADM 0 0") or line.startswith("BDONE 0 "): break
        check("slot_reused_without_leaking_state", reused == ref_C)
        report["tokens"] = {"references": refs, "batch": outputs, "resumed": resumed,
                            "next_turn": next_turn, "next_turn_reference": ref_B2,
                            "reused": reused, "reuse_reference": ref_C}
        report["passed"] = all(report["checks"].values())
        return 0 if report["passed"] else 1
    except Exception as error:
        report["error"] = str(error)
        report["passed"] = False
        raise
    finally:
        eng.close()
        (a.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
