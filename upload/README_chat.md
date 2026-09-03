---
license: apache-2.0
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- pebble
- mamba2
- hybrid
- small-language-model
- chat
base_model: basically-ai/Pebble-25M-Chat
---

# Pebble-25M-Chat-GGUF

GGUF conversions of [basically-ai/Pebble-25M-Chat](https://huggingface.co/basically-ai/Pebble-25M-Chat) (Apache 2.0).

## IMPORTANT: patched llama.cpp required

Pebble uses a custom hybrid Mamba2 + attention architecture. These GGUFs carry
`general.architecture = "pebble"`, which **upstream llama.cpp refuses to load**.

Everything needed to run them lives in the support repo:

**[rootendpoint/basicallyai_llama.cpp_support](https://github.com/rootendpoint/basicallyai_llama.cpp_support)**

- `llama.cpp-pebble.patch` - adds the `pebble` architecture to llama.cpp
  (applies cleanly against upstream commit `0eadefe`)
- `basicallyai_to_gguf.py` - standalone converter (numpy + safetensors only)
- `numpy_reference.py` - independent reference implementation used to verify correctness

The chat template is embedded in the GGUF metadata; chat mode works out of the box:

```bash
llama-cli -m pebble-25m-chat-f16.gguf
```

Chat format: lowercase `user: ...` / `assistant: ...` turns.

## Files

| Quant | Size | Type |
|-------|------|------|
| f16 | 49.1 MB | F16 |
| q8_0 | 26.3 MB | mostly Q8_0 |
| q4_k_m | 18.5 MB | mostly Q4_K_M |

## Verification

Outputs were cross-checked token-by-token against an independent pure-numpy
reference implementation over an 8-prompt battery (CPU and CUDA backends).
Identical greedy sequences on all prompts. Quantized builds verified
greedy-identical to f16 (8/8 prompts for both q8_0 and q4_k_m).

## Model

- 25M parameters, hidden 608, 8 layers (6 Mamba2 + 2 attention), ctx 2048, vocab 2048
- SFT chat variant; a research-scale model: expect toy-level output quality.
