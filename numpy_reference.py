#!/usr/bin/env python3
"""Independent numpy reference for Pebble-10M (basically-ai).

Reimplements the Pebble forward pass (hybrid Mamba2 + attention) in pure numpy
from the HF safetensors, following modeling_pebble.py and the mamba_ssm Mamba2
math that llama.cpp's graph reproduces. Used to cross-check llama.cpp's pebble
arch output token-by-token under greedy decoding.

Deps: numpy, safetensors only. No torch/transformers.

Usage:
  .venv/bin/python numpy_reference.py --model-dir Pebble-10M --prompt "Hello" --n-tokens 24
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file as load_safetensors

# ---------------------------------------------------------------- tokenizer

def bytes_to_unicode() -> dict[int, str]:
    # canonical GPT-2 bytes -> unicode map
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

GPT2_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(self, model_dir: Path):
        with open(model_dir / "tokenizer.json", encoding="utf-8") as f:
            tj = json.load(f)
        self.vocab = tj["model"]["vocab"]
        self.id_to_token = {int(i): t for t, i in self.vocab.items()}
        merges = tj["model"]["merges"]
        self.rank = {tuple(m): i for i, m in enumerate(merges)}
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

    def bpe(self, token: str) -> tuple[str, ...]:
        word = tuple(token)
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            best = min(pairs, key=lambda p: self.rank.get(p, 1 << 30))
            if best not in self.rank:
                break
            new = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    new.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
        return word

    def encode(self, text: str) -> list[int]:
        ids = []
        for m in GPT2_PAT.finditer(text):
            chunk = "".join(self.b2u[b] for b in m.group(0).encode("utf-8"))
            for tok in self.bpe(chunk):
                ids.append(self.vocab[tok])
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.id_to_token.get(i, "") for i in ids)
        out = bytearray()
        for c in text:
            out.append(self.u2b.get(c, ord(c)))
        return out.decode("utf-8", errors="replace")

# ---------------------------------------------------------------- math helpers

def softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20.0, x, np.log1p(np.exp(x)))

def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))

def gelu(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))

def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x * np.reciprocal(np.sqrt(np.mean(x * x) + eps)) * w

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# ---------------------------------------------------------------- model

class PebbleRef:
    def __init__(self, model_dir: Path):
        with open(model_dir / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        self.t = load_safetensors(str(model_dir / "model.safetensors"))
        # mirror the GGUF dtype policy: matmul weights are F16, everything else F32
        f16_suffixes = (".in_proj.weight", ".out_proj.weight", ".wqkv.weight",
                        ".wo.weight", ".fc1.weight", ".fc2.weight", "wte.weight")
        for k in list(self.t):
            if any(k.endswith(s) for s in f16_suffixes):
                self.t[k] = self.t[k].astype(np.float16)
        self.hidden = cfg["hidden_size"]
        self.n_head = cfg["num_attention_heads"]
        self.head_dim = self.hidden // self.n_head
        self.n_layer = cfg["num_hidden_layers"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = float(cfg["attention"]["rope_theta"])
        mamba = cfg.get("mamba2") or {}
        self.d_state = mamba.get("d_state", 128)
        self.d_conv = mamba.get("d_conv", 4)
        self.d_inner = (mamba.get("expand", 2)) * self.hidden
        self.ssm_head_dim = mamba.get("headdim", 96)
        self.ssm_heads = self.d_inner // self.ssm_head_dim
        self.is_attn = [c == "a" for c in cfg["block_pattern"] if c in "ma"]
        self.wte = self.t["wte.weight"].astype(np.float32)
        self.lnf = self.t["lnf.weight"].astype(np.float32)
        self.conv_hist: list[list[np.ndarray] | None] = [None] * self.n_layer
        self.ssm_state: list[np.ndarray | None] = [None] * self.n_layer
        self.k_cache: list[list[np.ndarray]] = [[] for _ in range(self.n_layer)]
        self.v_cache: list[list[np.ndarray]] = [[] for _ in range(self.n_layer)]

    def forward_token(self, tok_id: int, pos: int) -> np.ndarray:
        x = self.wte[tok_id].copy()
        for il in range(self.n_layer):
            if self.is_attn[il]:
                x = self._attn_layer(il, x, pos)
            else:
                x = self._mamba_layer(il, x)
        h = rmsnorm(x, self.lnf, self.eps)
        return self.wte @ h  # tied lm_head

    def _mamba_layer(self, il: int, x: np.ndarray) -> np.ndarray:
        p = f"blocks.{il}"
        ln = self.t[f"{p}.ln.weight"].astype(np.float32)
        w_in = self.t[f"{p}.mixer.in_proj.weight"].astype(np.float32)      # [1800, 384]
        w_conv = self.t[f"{p}.mixer.conv1d.weight"].astype(np.float32).reshape(
            self.d_inner + 2 * self.d_state, self.d_conv)
        b_conv = self.t[f"{p}.mixer.conv1d.bias"].astype(np.float32)
        dt_bias = self.t[f"{p}.mixer.dt_bias"].astype(np.float32)
        A = -np.exp(self.t[f"{p}.mixer.A_log"].astype(np.float32))
        D = self.t[f"{p}.mixer.D"].astype(np.float32)
        w_norm = self.t[f"{p}.mixer.norm.weight"].astype(np.float32)
        w_out = self.t[f"{p}.mixer.out_proj.weight"].astype(np.float32)    # [384, 768]

        h = rmsnorm(x, ln, self.eps)
        zxBCdt = w_in @ h  # [1800] = z | x | B | C | dt
        z = zxBCdt[: self.d_inner]
        xin = zxBCdt[self.d_inner : 2 * self.d_inner]
        B = zxBCdt[2 * self.d_inner : 2 * self.d_inner + self.d_state]
        C = zxBCdt[2 * self.d_inner + self.d_state : 2 * self.d_inner + 2 * self.d_state]
        dt = zxBCdt[-self.ssm_heads:]

        xBC = np.concatenate([xin, B, C])
        hist = self.conv_hist[il]
        if hist is None:
            hist = [np.zeros_like(xBC)] * (self.d_conv - 1)
        # causal conv, F.conv1d semantics: y[t] = w0*x[t-3] + w1*x[t-2] + w2*x[t-1] + w3*x[t]
        out = b_conv + w_conv[:, -1] * xBC
        for k in range(self.d_conv - 1):
            out = out + w_conv[:, self.d_conv - 2 - k] * hist[k]
        self.conv_hist[il] = [xBC] + hist[:-1]
        xBC = silu(out)

        xin = xBC[: self.d_inner]
        B = xBC[self.d_inner : self.d_inner + self.d_state]
        C = xBC[self.d_inner + self.d_state : self.d_inner + 2 * self.d_state]
        dt = softplus(dt + dt_bias)

        if self.ssm_state[il] is None:
            self.ssm_state[il] = np.zeros((self.ssm_heads, self.ssm_head_dim, self.d_state), dtype=np.float32)
        st = self.ssm_state[il]
        y = np.empty(self.d_inner, dtype=np.float32)
        for g in range(self.ssm_heads):
            xg = xin[g * self.ssm_head_dim : (g + 1) * self.ssm_head_dim]
            # per (head, channel): h = dA*h + dt*B*x ; y = C.h + D*x
            st[g] = st[g] * math.exp(dt[g] * A[g]) + np.outer(xg * dt[g], B)
            y[g * self.ssm_head_dim : (g + 1) * self.ssm_head_dim] = st[g] @ C + D[g] * xg
        y = y * silu(z)
        y = rmsnorm(y, w_norm, self.eps)
        return x + (w_out @ y)

    def _attn_layer(self, il: int, x: np.ndarray, pos: int) -> np.ndarray:
        p = f"blocks.{il}"
        ln1 = self.t[f"{p}.ln1.weight"].astype(np.float32)
        ln2 = self.t[f"{p}.ln2.weight"].astype(np.float32)
        wqkv = self.t[f"{p}.wqkv.weight"].astype(np.float32)
        wo = self.t[f"{p}.wo.weight"].astype(np.float32)
        fc1 = self.t[f"{p}.fc1.weight"].astype(np.float32)
        fc2 = self.t[f"{p}.fc2.weight"].astype(np.float32)

        h = rmsnorm(x, ln1, self.eps)
        qkv = wqkv @ h
        q = qkv[: self.hidden].reshape(self.n_head, self.head_dim).astype(np.float32)
        k = qkv[self.hidden : 2 * self.hidden].reshape(self.n_head, self.head_dim).astype(np.float32)
        v = qkv[2 * self.hidden :].reshape(self.n_head, self.head_dim).astype(np.float32)
        half = self.head_dim // 2
        for i in range(half):
            ang = pos * (self.theta ** (-2.0 * i / self.head_dim))
            c, s = math.cos(ang), math.sin(ang)
            q1, q2 = q[:, i].copy(), q[:, i + half].copy()
            q[:, i] = q1 * c - q2 * s
            q[:, i + half] = q1 * s + q2 * c
            k1, k2 = k[:, i].copy(), k[:, i + half].copy()
            k[:, i] = k1 * c - k2 * s
            k[:, i + half] = k1 * s + k2 * c

        self.k_cache[il].append(k)
        self.v_cache[il].append(v)
        K = np.stack(self.k_cache[il])
        V = np.stack(self.v_cache[il])
        scores = np.einsum("hd,thd->ht", q, K) / math.sqrt(self.head_dim)  # [n_head, T]
        att = np.stack([softmax(scores[hh, :]) for hh in range(self.n_head)])
        y = np.einsum("ht,thd->hd", att, V)
        x = x + (wo @ y.ravel())

        h2 = rmsnorm(x, ln2, self.eps)
        x = x + (fc2 @ gelu(fc1 @ h2))
        return x

# ---------------------------------------------------------------- main

def load_gguf_tensors(gguf_path: Path, llama_cpp: Path) -> dict:
    """Map GGUF tensor names back to HF-style names, bit-exact with llama.cpp."""
    import sys
    sys.path.insert(1, str(llama_cpp / "gguf-py"))
    import gguf
    r = gguf.GGUFReader(str(gguf_path))
    out = {}
    for t in r.tensors:
        arr = np.array(t.data)  # already F16/F32 per GGUF policy
        name = t.name
        if name == "token_embd.weight":
            out["wte.weight"] = arr
        elif name == "output_norm.weight":
            out["lnf.weight"] = arr
        else:
            parts = name.split(".")
            blk = f"blocks.{parts[1]}"
            rest = ".".join(parts[2:])
            if rest == "attn_norm.weight":
                out[f"{blk}.ln.weight"] = arr
                out[f"{blk}.ln1.weight"] = arr
            elif rest == "ssm_in.weight":
                out[f"{blk}.mixer.in_proj.weight"] = arr
            elif rest == "ssm_conv1d.weight":
                out[f"{blk}.mixer.conv1d.weight"] = arr.reshape(-1, 1, arr.shape[-1])
            elif rest == "ssm_conv1d.bias":
                out[f"{blk}.mixer.conv1d.bias"] = arr
            elif rest == "ssm_dt.bias":
                out[f"{blk}.mixer.dt_bias"] = arr
            elif rest == "ssm_a":
                out[f"{blk}.mixer.A"] = arr.reshape(-1)  # already -exp(A_log)
            elif rest == "ssm_d":
                out[f"{blk}.mixer.D"] = arr.reshape(-1)
            elif rest == "ssm_norm.weight":
                out[f"{blk}.mixer.norm.weight"] = arr.reshape(-1)
            elif rest == "ssm_out.weight":
                out[f"{blk}.mixer.out_proj.weight"] = arr
            elif rest == "attn_qkv.weight":
                out[f"{blk}.wqkv.weight"] = arr
            elif rest == "attn_output.weight":
                out[f"{blk}.wo.weight"] = arr
            elif rest == "ffn_norm.weight":
                out[f"{blk}.ln2.weight"] = arr
            elif rest == "ffn_up.weight":
                out[f"{blk}.fc1.weight"] = arr
            elif rest == "ffn_down.weight":
                out[f"{blk}.fc2.weight"] = arr
            else:
                raise KeyError(f"unmapped gguf tensor {name}")
    # derive the two norms the converter folded into one
    return out

class PebbleRefGGUF(PebbleRef):
    def __init__(self, gguf_path: Path, llama_cpp: Path):
        from pathlib import Path as _P
        cfg_dir = gguf_path.parent / "Pebble-10M"
        with open(cfg_dir / "config.json", encoding="utf-8") as f:
            import json as _json
            cfg = _json.load(f)
        self.__dict__["cfg"] = cfg
        self.t = load_gguf_tensors(gguf_path, llama_cpp)
        self.hidden = cfg["hidden_size"]
        self.n_head = cfg["num_attention_heads"]
        self.head_dim = self.hidden // self.n_head
        self.n_layer = cfg["num_hidden_layers"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = float(cfg["attention"]["rope_theta"])
        mamba = cfg.get("mamba2") or {}
        self.d_state = mamba.get("d_state", 128)
        self.d_conv = mamba.get("d_conv", 4)
        self.d_inner = (mamba.get("expand", 2)) * self.hidden
        self.ssm_head_dim = mamba.get("headdim", 96)
        self.ssm_heads = self.d_inner // self.ssm_head_dim
        self.is_attn = [c == "a" for c in cfg["block_pattern"] if c in "ma"]
        self.wte = self.t["wte.weight"].astype(np.float32)
        self.lnf = self.t["lnf.weight"].astype(np.float32)
        self.conv_hist = [None] * self.n_layer
        self.ssm_state = [None] * self.n_layer
        self.k_cache = [[] for _ in range(self.n_layer)]
        self.v_cache = [[] for _ in range(self.n_layer)]
        # gguf stores A pre-transformed
        for il in range(self.n_layer):
            if not self.is_attn[il]:
                p = f"blocks.{il}"
                self.t[f"{p}.mixer.A_log"] = np.log(-self.t[f"{p}.mixer.A"].astype(np.float32))


def run(model_dir: Path, prompt: str, n_tokens: int, eos_id: int) -> None:
    tok = Tokenizer(model_dir)
    prompt_ids = tok.encode(prompt)
    print("prompt ids:", prompt_ids)

    model = PebbleRef(model_dir)
    logits = None
    for pos, tid in enumerate(prompt_ids):
        logits = model.forward_token(tid, pos)

    gen: list[int] = []
    nxt = int(np.argmax(logits))
    for _ in range(n_tokens):
        gen.append(nxt)
        if nxt == eos_id:
            break
        logits = model.forward_token(nxt, len(prompt_ids) + len(gen))
        nxt = int(np.argmax(logits))

    print("generated ids:", gen)
    print("generated text:", repr(tok.decode(prompt_ids + gen)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--prompt", type=str, default="Hello")
    ap.add_argument("--n-tokens", type=int, default=24)
    args = ap.parse_args()
    run(args.model_dir, args.prompt, args.n_tokens, eos_id=0)
