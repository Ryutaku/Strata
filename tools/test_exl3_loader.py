"""Exercise the C++ model loader with small valid and malformed safetensors files.

STRATA_EXL3_MODEL_CHECK=build-cuda12/strata-exl3-model python tools/test_exl3_loader.py
"""
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("STRATA_EXL3_MODEL_CHECK"), "needs the C++ model checker")
class LoaderTests(unittest.TestCase):
    def run_model(self, edit=None, bits=3, pad=False, header_length=None):
        tensors = {
            "linear.trellis": {"dtype": "I16", "shape": [8, 8, bits * 16],
                               "data_offsets": [0, 8 * 8 * bits * 32]},
        }
        offset = tensors["linear.trellis"]["data_offsets"][1]
        for name in ("suh", "svh"):
            tensors["linear." + name] = {"dtype": "F16", "shape": [128],
                                         "data_offsets": [offset, offset + 256]}
            offset += 256
        payload = bytes(offset)
        if edit:
            edit(tensors)
        if pad:
            payload = payload.ljust(max(t["data_offsets"][1] for t in tensors.values()), b"\0")
        header = json.dumps(tensors).encode()
        with tempfile.TemporaryDirectory(prefix="strata-exl3-loader-") as tmp:
            root = Path(tmp)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {key: "model.safetensors" for key in tensors}}))
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header) if header_length is None else header_length) + header + payload)
            return subprocess.run([os.environ["STRATA_EXL3_MODEL_CHECK"], tmp, "linear"],
                                  capture_output=True, text=True)

    def test_integer_bitrates(self):
        for bits in range(1, 9):
            with self.subTest(bits=bits):
                result = self.run_model(bits=bits)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_unsupported_geometry(self):
        for shape in ([8, 8, 24], [8, 8, 0], [8, 8, 144], [1, 8, 48], [0, 8, 48], [8, 8]):
            with self.subTest(shape=shape):
                def edit(tensors):
                    t = tensors["linear.trellis"]
                    t["shape"] = shape
                    # Give the malformed geometry its correct byte count; fail in linear(), not file parsing.
                    count = 2
                    for dim in shape:
                        count *= dim
                    t["data_offsets"][1] = count
                result = self.run_model(edit, pad=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("integer K=1..8", result.stderr)

    def test_bad_scales(self):
        for key, value in (("dtype", "I16"), ("shape", [64, 2])):
            with self.subTest(key=key):
                result = self.run_model(lambda t: t["linear.suh"].update({key: value}))
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("F16 suh[in]", result.stderr)

    def test_file_bounds_and_sizes(self):
        for end in (-1, 100000, 1):
            with self.subTest(end=end):
                result = self.run_model(lambda t: t["linear.trellis"].update(data_offsets=[0, end]))
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("safetensors:", result.stderr)
        for shape in ([-1, -1], [9223372036854775807, 3], [9223372036854775808]):
            with self.subTest(shape=shape):
                result = self.run_model(lambda t: t["linear.trellis"].update(shape=shape))
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("safetensors", result.stderr)

    def test_header_length_overflow(self):
        for length in (0, (1 << 64) - 1):
            with self.subTest(length=length):
                result = self.run_model(header_length=length)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("bad header length", result.stderr)

    def test_bias_is_rejected(self):
        def edit(tensors):
            tensors["linear.bias"] = copy.deepcopy(tensors["linear.svh"])
        result = self.run_model(edit)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("biased EXL3", result.stderr)


@unittest.skipUnless(os.environ.get("STRATA_EXL3_ENGINE"), "needs the engine")
class CommandLineTests(unittest.TestCase):
    def engine(self, *args):
        return subprocess.run([os.environ["STRATA_EXL3_ENGINE"], "--tokens", "1", *args],
                              capture_output=True, text=True)

    def test_ple_sources_are_exclusive(self):
        for args in (("--ple-exl3", "fake", "--no-ple"),
                     ("--ple-exl3", "fake", "--ple-gguf", "fake")):
            with self.subTest(args=args):
                result = self.engine(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("choose one", result.stderr)

    def test_unsupported_modes_fail_before_loading(self):
        for args in (("--layer-split", "auto"),
                     ("--native", "fake.gguf")):
            with self.subTest(args=args):
                result = self.engine("--exl3", "fake", "--no-ple", *args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--exl3 does not yet accept", result.stderr)

    def test_help_lists_exl3(self):
        result = self.engine("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--exl3 DIR", result.stderr + result.stdout)
        self.assertIn("--ple-exl3 FILE", result.stderr + result.stdout)

    def test_diagnostic_dense_modes_are_rejected_for_windows(self):
        for dense in ("exl3", "only:output"):
            with self.subTest(dense=dense):
                result = subprocess.run([os.environ["STRATA_EXL3_ENGINE"], "--exl3", "fake",
                                         "--no-ple", "--tokens", "1", "--spec", "4"],
                    env={**os.environ, "STRATA_EXL3_DENSE": dense}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("unset STRATA_EXL3_DENSE", result.stderr)


if __name__ == "__main__":
    unittest.main()
