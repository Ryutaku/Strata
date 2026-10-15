"""Run a short real-model EXL3 regression on a GPU selected by the caller.

Requires a complete Qwen3.8-Flash-Next EXL3 model, its PLE file and a free GPU.
Does not start/stop services or change model files.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
from strata_tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--gpu", required=True, help="physical GPU index")
    parser.add_argument("--cache", type=int, default=64, help="EXL3 slots per layer; 0 disables caching")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spec", type=int, default=0)
    parser.add_argument("--mtp", type=Path)
    parser.add_argument("--prefill", type=int, default=0)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    if args.cache < 0:
        parser.error("--cache must be nonnegative")
    model = args.model.resolve()
    with tempfile.TemporaryDirectory(prefix="strata-exl3-smoke-") as tmp:
        root = Path(tmp)
        subprocess.run([sys.executable, str(TOOLS / "exl3/tokenizer_to_pack.py"), "--hf", str(model),
                        "--out", tmp, "--check"], check=True, stdout=sys.stderr)
        pack = root / "tokenizer"
        vocab = json.loads((pack / "vocab.json").read_text())
        if isinstance(vocab, dict):
            tokens_by_id = [None] * len(vocab)
            for token, index in vocab.items(): tokens_by_id[index] = token
        else:
            tokens_by_id = vocab
        tk = Tokenizer(tokens_by_id,
                       (pack / "merges.txt").read_text().splitlines(),
                       json.loads((pack / "token_type.json").read_text()))
        prompt = ("<|im_start|>user\nWhat is 17 + 25? Reply with the number only.<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n</think>\n\n")
        tokens = root / "prompt.tokens"
        tokens.write_text(",".join(map(str, tk.encode(prompt, parse_special=True))))
        command = [str(args.engine.resolve()), "--gpu", args.gpu, "--exl3", str(model),
                   "--ple-exl3", str(model / "ngram_embedding.safetensors"),
                   "--tokens-file", str(tokens), "--max-new", "64", "--max-context", "512",
                   "--expert-cache", str(args.cache), "--pcie-frac", "0", "--stop-eos",
                   "--ple-io", "mmap", "--pool-workers", "8"]
        if args.spec: command.extend(["--spec", str(args.spec)])
        if args.mtp: command.extend(["--mtp", str(args.mtp.resolve())])
        if args.prefill: command.extend(["--prefill", str(args.prefill)])
        result = subprocess.run(command, capture_output=True, text=True)
        if args.log_dir:
            args.log_dir.mkdir(parents=True, exist_ok=True)
            (args.log_dir / "stdout.log").write_text(result.stdout)
            (args.log_dir / "stderr.log").write_text(result.stderr)
        if result.returncode:
            sys.stderr.write(result.stderr)
            return result.returncode if result.returncode > 0 else 1
        output = next((line for line in result.stdout.splitlines() if line.startswith("output  :")), None)
        if output is None:
            raise RuntimeError("engine did not return output token IDs")
        ids = list(map(int, output.split(":", 1)[1].split()))
        answer = tk.decode(ids)
        if not re.search(r"(?<!\d)42(?!\d)", answer):
            raise RuntimeError("wrong arithmetic answer: " + repr(answer))
        if "R4 hit path ON" in result.stderr or "expert cache " in result.stderr:
            raise RuntimeError("EXL3 incorrectly enabled the GGUF expert cache")
        report = {"answer": answer, "tokens": ids, "cache_slots_per_layer": args.cache, "spec": args.spec,
                  "mtp": str(args.mtp) if args.mtp else None, "prefill_chunk": args.prefill,
                  "timings": [line for line in result.stdout.splitlines()
                              if line.startswith(("decode ", "prefill "))]}
        encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.write_text(encoded)
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
