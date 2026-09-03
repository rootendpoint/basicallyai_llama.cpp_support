#!/usr/bin/env python3
"""Cross-check pebble-25m GGUFs against the raw HF repos via the numpy oracle.

Runs the oracle (pure-numpy HF math) and llama-completion (patched llama.cpp,
CPU, greedy) on the same prompt battery and asserts token-identical output.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA = Path("/mnt/ssd/projects/llama.cpp/build/bin/llama-completion")

PROMPTS = [
    "Hello",
    "The capital of France is",
    "Once upon a time",
    "the quick brown fox",
    "1 2 3 4 5",
    "What is 2+2?",
    "To be or not to be",
    "user: hi\nassistant:",
]

sys.path.insert(0, str(HERE))
from numpy_reference import run as oracle_run, Tokenizer  # noqa: E402
import io
from contextlib import redirect_stdout


def oracle_text(model_dir: Path, prompt: str, n: int) -> str:
    tok = Tokenizer(model_dir)
    ids = tok.encode(prompt)
    buf = io.StringIO()
    with redirect_stdout(buf):
        oracle_run(model_dir, prompt, n, eos_id=0)
    # last printed line is "generated text: '...'"
    line = [l for l in buf.getvalue().splitlines() if l.startswith("generated text:")][-1]
    return eval(line[len("generated text: "):]).rstrip("\n")


def llama_text(gguf: Path, prompt: str, n: int) -> str:
    out = subprocess.run(
        [str(LLAMA), "-m", str(gguf), "-p", prompt, "-n", str(n),
         "--temp", "0", "--top-k", "0", "--top-p", "1.0", "--seed", "0"],
        capture_output=True, text=True, timeout=300,
    ).stdout
    return out.rstrip("\n")  # full output: prompt echo + generation, like the oracle


def main() -> None:
    ok = True
    for model in ("Pebble-25M", "Pebble-25M-Chat"):
        # math check uses the template-free build (this llama.cpp build applies
        # embedded chat templates to -p prompts in completion mode)
        gguf = HERE / (("pebble-25m-chat-f16-notpl.gguf") if "Chat" in model else "pebble-25m-f16.gguf")
        for p in PROMPTS:
            o = oracle_text(HERE / model, p, 32)
            l = llama_text(gguf, p, 32)
            match = o == l
            ok = ok and match
            print(("PASS" if match else "FAIL"), model, repr(p), "->", repr(o[:60]))
            if not match:
                print("   llama:", repr(l[:60]))
    print("RESULT:", "ALL MATCH" if ok else "DIVERGENCE FOUND")


if __name__ == "__main__":
    main()
