"""EXL3 tokenizer export must use the same vocabulary contract as serve/server.py."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_tokenizer import BYTE_TO_UNICODE, Tokenizer


class ExportTokenizer(unittest.TestCase):
    def test_server_vocabulary_contract_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf = root / "hf"
            hf.mkdir()
            vocab = {token: byte for byte, token in BYTE_TO_UNICODE.items()}
            (hf / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 260}}))
            (hf / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}, "added_tokens": [
                {"id": 256, "content": "<|im_end|>", "special": True}]}))
            (hf / "merges.txt").write_text("")
            script = Path(__file__).resolve().parent / "exl3/tokenizer_to_pack.py"
            p = subprocess.run([sys.executable, str(script), "--hf", str(hf), "--out", str(root / "pack"),
                                "--check"], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            exported = root / "pack/tokenizer"
            raw = json.loads((exported / "vocab.json").read_text(encoding="utf-8"))
            self.assertIsInstance(raw, dict)
            self.assertEqual(len(raw), 260)
            tokens = [None] * len(raw)
            for token, index in raw.items(): tokens[index] = token
            types = json.loads((exported / "token_type.json").read_text())
            tk = Tokenizer(tokens, [], types)
            for text in ["Hello!", "你好🌍", "def f(x):\n  return x + 1", "<|im_end|>"]:
                self.assertEqual(tk.decode(tk.encode(text, parse_special=True)), text)
            self.assertEqual(tk.encode("<|im_end|>", parse_special=True), [256])


if __name__ == "__main__":
    unittest.main()
