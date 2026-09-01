# basicallyai llama.cpp support

Tooling to run [basically-ai/Pebble-10M](https://huggingface.co/basically-ai/Pebble-10M)
and [basically-ai/Pebble-10M-Chat](https://huggingface.co/basically-ai/Pebble-10M-Chat)
- a custom hybrid Mamba2 + attention architecture - in [llama.cpp](https://github.com/ggml-org/llama.cpp).

Ready-to-use GGUFs: [ContextReq/Pebble-10M-GGUF](https://huggingface.co/ContextReq/Pebble-10M-GGUF)
and [ContextReq/Pebble-10M-Chat-GGUF](https://huggingface.co/ContextReq/Pebble-10M-Chat-GGUF).

Upstream llama.cpp refuses GGUFs with `general.architecture = "pebble"`, so a
patched build is required. This repo provides everything to reproduce it.

## Contents

| File | Purpose |
|------|---------|
| `llama.cpp-pebble.patch` | Adds the `pebble` architecture (LLM_ARCH_PEBBLE) to llama.cpp. Applies cleanly against upstream commit `0eadefe`. |
| `basicallyai_to_gguf.py` | Standalone HF -> GGUF converter (numpy + safetensors only, no torch). |
| `numpy_reference.py` | Independent pure-numpy reimplementation of the Pebble forward pass, used to verify the runtime token-by-token. |
| `REQUIREMENTS.txt` | Python deps for the converter and reference. |

## Build patched llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 0eadefe
git apply ../basicallyai_llama.cpp_support/llama.cpp-pebble.patch

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=ON
cmake --build build --config Release -j
```

Adjust the CMake flags for your hardware (the patch touches only the model
layer; every backend works, CPU included).

## Convert from the HF checkpoints

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r REQUIREMENTS.txt

python basicallyai_to_gguf.py --model-dir Pebble-10M --outfile pebble-10m-f16.gguf
python basicallyai_to_gguf.py --model-dir Pebble-10M-Chat --outfile pebble-10m-chat-f16.gguf

# optional quantization
llama-quantize pebble-10m-f16.gguf pebble-10m-f16-q8_0.gguf q8_0
```

The converter embeds the chat template for the Chat model (lowercase
`user:` / `assistant:` labels - the best match found for the undisclosed SFT
format, determined by A/B testing).

## Verify

```bash
python numpy_reference.py --model-dir Pebble-10M --prompt "The capital of France" --n-tokens 24
```

## Notes

- Validated on NVIDIA GTX 1660 SUPER (sm_75, CUDA) and CPU: identical greedy
  token sequences against the numpy reference across multiple prompts (the only
  divergences are genuine argmax ties), quantized KV cache included.
- Pebble-10M is a 10M-parameter research model: expect toy-level output quality.
- Known approximation: the HF implementation's Mamba2 mixer RMS norm uses eps 1e-5
  while the GGUF carries one eps (1e-6) shared by all norms - a ~1e-5 relative effect.

## License

Apache 2.0 (matching the original models).
