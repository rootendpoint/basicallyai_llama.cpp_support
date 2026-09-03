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
base_model: basically-ai/Pebble-25M
---

# Pebble-25M-GGUF

GGUF conversions of [basically-ai/Pebble-25M](https://huggingface.co/basically-ai/Pebble-25M) (Apache 2.0).

## IMPORTANT: patched llama.cpp required

Pebble uses a custom hybrid Mamba2 + attention architecture. These GGUFs carry
`general.architecture = "pebble"`, which **upstream llama.cpp refuses to load**.

Everything needed to run them lives in the support repo:

**[rootendpoint/basicallyai_llama.cpp_support](https://github.com/rootendpoint/basicallyai_llama.cpp_support)**

- `llama.cpp-pebble.patch` - adds the `pebble` architecture to llama.cpp
  (applies cleanly against upstream commit `0eadefe`)
- `basicallyai_to_gguf.py` - standalone converter (numpy + safetensors only)
- `numpy_reference.py` - independent reference implementation used to verify correctness

Apply the patch, rebuild llama.cpp, then:

```bash
llama-cli -m pebble-25m-f16.gguf -p "The capital of France" -n 64
```

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
greedy-identical to f16 (q8_0: 7/8 prompts, one genuine argmax tie-flip;
q4_k_m: 2/8, expected at this scale).

## Model

- 25M parameters, hidden 608, 8 layers (6 Mamba2 + 2 attention), ctx 2048, vocab 2048
- A research-scale model: expect toy-level output quality.
