#!/usr/bin/env python3
"""Convert basically-ai/Pebble-10M and Pebble-10M-Chat HF checkpoints to GGUF.

Pebble is a custom hybrid Mamba2 + Attention architecture (model_type
"pebble_10m") that llama.cpp's stock converter does not register, hence this
standalone converter. Output GGUF uses general.architecture = "pebble";
loading it needs the matching llama.cpp runtime patch (new LLM_ARCH_PEBBLE).

Deps: numpy, safetensors (REQUIREMENTS.txt). gguf-py is imported from the
llama.cpp clone (--gguf-path). No torch/transformers needed.

Examples:
  python basicallyai_to_gguf.py --model-dir Pebble-10M --outfile pebble-10m-f16.gguf
  python basicallyai_to_gguf.py --model-dir Pebble-10M-Chat --outfile pebble-10m-chat-f16.gguf
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file as load_safetensors

DEFAULT_LLAMA_CPP = Path(__file__).resolve().parent.parent / "llama.cpp"

ARCH = "pebble"
FTYPE = 1  # MOSTLY_F16

DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}user: "
    "{% elif message['role'] == 'assistant' %}assistant: "
    "{% else %}{{ message['role'] }}: {% endif %}"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for part in message['content'] %}{{ part['text'] }}{% endfor %}{% endif %}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("basicallyai_to_gguf")


def parse_block_pattern(pattern: str, n_layers: int) -> list[bool]:
    letters = [c for c in pattern if c in "ma"]
    if len(letters) != n_layers:
        raise ValueError(f"block_pattern {pattern!r} gives {len(letters)} layers, expected {n_layers}")
    return [c == "a" for c in letters]


def build_vocab(model_dir: Path) -> tuple[list[str], list[int]]:
    with open(model_dir / "tokenizer.json", encoding="utf-8") as f:
        tj = json.load(f)
    id_to_token = {int(i): t for t, i in tj["model"]["vocab"].items()}
    vocab_size = max(id_to_token) + 1
    added_ids = {int(a["id"]) for a in tj.get("added_tokens", [])}

    special_ids: set[int] = set()
    tc_path = model_dir / "tokenizer_config.json"
    if tc_path.is_file():
        with open(tc_path, encoding="utf-8") as f:
            tc = json.load(f)
        for key in ("bos_token_id", "eos_token_id", "unk_token_id", "pad_token_id"):
            v = tc.get(key)
            if isinstance(v, int):
                special_ids.add(v)

    tokens, toktypes = [], []
    for i in range(vocab_size):
        tokens.append(id_to_token.get(i, f"[PAD{i}]"))
        if i in special_ids:
            toktypes.append(int(gguf.TokenType.CONTROL))
        elif i in added_ids:
            toktypes.append(int(gguf.TokenType.USER_DEFINED))
        elif i not in id_to_token:
            toktypes.append(int(gguf.TokenType.UNUSED))
        else:
            toktypes.append(int(gguf.TokenType.NORMAL))
    return tokens, toktypes


def resolve_chat_template(value: str | None, model_dir: Path) -> str | None:
    if value is not None:
        p = Path(value)
        return p.read_text(encoding="utf-8") if p.is_file() else value
    if "-chat" in model_dir.name.lower():
        return DEFAULT_CHAT_TEMPLATE
    return None


def convert(model_dir: Path, outfile: Path, chat_template: str | None) -> None:
    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    tensors = load_safetensors(str(model_dir / "model.safetensors"))

    n_layer = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    n_head = cfg["num_attention_heads"]
    head_dim = hidden // n_head
    ffn = cfg["intermediate_size"]
    vocab_size = cfg["vocab_size"]
    mamba = cfg.get("mamba2") or {}
    d_state = mamba.get("d_state", 128)
    d_conv = mamba.get("d_conv", 4)
    d_inner = (mamba.get("expand", 2)) * hidden
    headdim = mamba.get("headdim", 96)
    n_group = 1  # single group; matches conv1d channels d_inner + 2*d_state
    dt_rank = d_inner // headdim  # nheads; matches in_proj width
    is_attn = parse_block_pattern(cfg["block_pattern"], n_layer)

    gw = gguf.GGUFWriter(str(outfile), ARCH)
    gw.add_name(outfile.stem)
    gw.add_context_length(cfg["max_position_embeddings"])
    gw.add_embedding_length(hidden)
    gw.add_block_count(n_layer)
    gw.add_feed_forward_length(ffn)
    gw.add_head_count(n_head)
    gw.add_head_count_kv([n_head if a else 0 for a in is_attn])
    gw.add_key_length(head_dim)
    gw.add_value_length(head_dim)
    gw.add_rope_dimension_count(head_dim)
    gw.add_rope_freq_base(float(cfg["attention"]["rope_theta"]))
    gw.add_rope_scaling_type(gguf.RopeScalingType.NONE)
    gw.add_ssm_conv_kernel(d_conv)
    gw.add_ssm_inner_size(d_inner)
    gw.add_ssm_state_size(d_state)
    gw.add_ssm_time_step_rank(dt_rank)
    gw.add_ssm_group_count(n_group)
    gw.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    gw.add_file_type(FTYPE)
    gw.add_quantization_version(gguf.GGML_QUANT_VERSION)

    # vocab: gpt2-style BPE, pre-tokenizer "gpt-2"
    tokens, toktypes = build_vocab(model_dir)
    gw.add_tokenizer_model("gpt2")
    gw.add_tokenizer_pre("gpt-2")
    gw.add_token_list(tokens)
    gw.add_token_types(toktypes)
    gguf.SpecialVocab(model_dir, load_merges=True).add_to_gguf(gw)
    chat_template = resolve_chat_template(chat_template, model_dir)
    if chat_template:
        gw.add_chat_template(chat_template)

    def qtype(name: str, arr: np.ndarray) -> np.dtype:
        # mirror conversion/base.py: 1D tensors, norms, SSM_CONV1D and
        # non-".weight" names stay F32; only the matmul weights go F16
        if arr.ndim <= 1 or name.endswith("_norm.weight") or name.endswith("ssm_conv1d.weight") or not name.endswith(".weight"):
            return np.float32
        return np.float16

    def add(name: str, arr: np.ndarray) -> None:
        gw.add_tensor(name, np.ascontiguousarray(arr.astype(qtype(name, arr))))

    add("token_embd.weight", tensors["wte.weight"])
    add("output_norm.weight", tensors["lnf.weight"])
    # lm_head.weight is tied to wte -> omitted; runtime duplicates token_embd

    for i in range(n_layer):
        p = f"blocks.{i}"
        if is_attn[i]:
            add(f"blk.{i}.attn_norm.weight", tensors[f"{p}.ln1.weight"])
            add(f"blk.{i}.attn_qkv.weight", tensors[f"{p}.wqkv.weight"])
            add(f"blk.{i}.attn_output.weight", tensors[f"{p}.wo.weight"])
            add(f"blk.{i}.ffn_norm.weight", tensors[f"{p}.ln2.weight"])
            add(f"blk.{i}.ffn_up.weight", tensors[f"{p}.fc1.weight"])
            add(f"blk.{i}.ffn_down.weight", tensors[f"{p}.fc2.weight"])
        else:
            add(f"blk.{i}.attn_norm.weight", tensors[f"{p}.ln.weight"])
            add(f"blk.{i}.ssm_in.weight", tensors[f"{p}.mixer.in_proj.weight"])
            add(f"blk.{i}.ssm_conv1d.weight", tensors[f"{p}.mixer.conv1d.weight"].squeeze(axis=1))
            add(f"blk.{i}.ssm_conv1d.bias", tensors[f"{p}.mixer.conv1d.bias"])
            add(f"blk.{i}.ssm_dt.bias", tensors[f"{p}.mixer.dt_bias"])
            add(f"blk.{i}.ssm_a", (-np.exp(tensors[f"{p}.mixer.A_log"])).reshape(-1, 1))
            add(f"blk.{i}.ssm_d", tensors[f"{p}.mixer.D"].reshape(-1, 1))
            add(f"blk.{i}.ssm_norm.weight", tensors[f"{p}.mixer.norm.weight"].reshape(n_group, d_inner // n_group))
            add(f"blk.{i}.ssm_out.weight", tensors[f"{p}.mixer.out_proj.weight"])

    n_params = sum(int(np.prod(v.shape)) for v in gw.tensors[0].values())
    logger.info("params written: %d", n_params)

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_ti_data_to_file()
    while gw.tensors[0]:
        info = gw.tensors[0][next(iter(gw.tensors[0]))]
        gw.write_tensor_data(info.tensor)
    gw.close()
    logger.info("wrote %s", outfile)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True, help="HF model dir (Pebble-10M or Pebble-10M-Chat)")
    ap.add_argument("--outfile", type=Path, required=True, help="output .gguf path")
    ap.add_argument("--chat-template", type=str, default=None,
                    help="jinja chat template (literal or path to file); auto-default for *-Chat dirs")
    ap.add_argument("--gguf-path", type=Path, default=DEFAULT_LLAMA_CPP,
                    help=f"llama.cpp clone to import gguf-py from (default: {DEFAULT_LLAMA_CPP})")
    args = ap.parse_args()

    sys.path.insert(1, str(args.gguf_path / "gguf-py"))
    globals()["gguf"] = importlib.import_module("gguf")

    convert(args.model_dir, args.outfile, args.chat_template)


if __name__ == "__main__":
    main()
