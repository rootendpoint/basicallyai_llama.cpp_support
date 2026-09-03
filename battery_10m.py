#!/usr/bin/env python3
"""Cross-check pebble-10m GGUFs against the raw HF repos via the numpy oracle."""
from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA = Path("/mnt/ssd/projects/llama.cpp/build/bin/llama-completion")
NL = chr(10)

PROMPTS = [
    "Hello",
    "The capital of France is",
    "Once upon a time",
    "the quick brown fox",
    "1 2 3 4 5",
    "What is 2+2?",
    "To be or not to be",
    "user: hi" + NL + "assistant:",
]

sys.path.insert(0, str(HERE))
from numpy_reference import run as oracle_run  # noqa: E402


def oracle_text(model_dir: Path, prompt: str, n: int) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        oracle_run(model_dir, prompt, n, eos_id=0)
    line = [l for l in buf.getvalue().splitlines() if l.startswith("generated text:")][-1]
    return eval(line[len("generated text: "):]).rstrip(NL)


def llama_text(gguf: Path, prompt: str, n: int) -> str:
    out = subprocess.run(
        [str(LLAMA), "-m", str(gguf), "-p", prompt, "-n", str(n),
         "--temp", "0", "--top-k", "0", "--top-p", "1.0", "--seed", "0"],
        capture_output=True, text=True, timeout=300,
    ).stdout
    return out.rstrip(NL)


def main() -> None:
    ok = True
    pairs = [
        (HERE / "Pebble-10M-orig", Path("/tmp/fresh10m/pebble-10m-f16.gguf")),
        (HERE / "Pebble-10M-Chat-orig", Path("/tmp/fresh10m/pebble-10m-chat-f16-notpl.gguf")),
    ]
    for model_dir, gguf in pairs:
        for p in PROMPTS:
            o = oracle_text(model_dir, p, 32)
            l = llama_text(gguf, p, 32)
            match = o == l
            ok = ok and match
            print(("PASS" if match else "FAIL"), model_dir.name, repr(p), "->", repr(o[:50]))
            if not match:
                print("   llama:", repr(l[:50]))
    print("RESULT:", "ALL MATCH" if ok else "DIVERGENCE FOUND")


if __name__ == "__main__":
    main()
