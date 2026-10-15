# Nebius Token Factory notes (probed 2026-10-06)

- Base URL `https://api.tokenfactory.nebius.com/v1`, OpenAI-compatible `/models` and `/chat/completions`. Bearer auth.
- NVIDIA models available: `nvidia/nemotron-3-super-120b-a12b`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`,
  `nvidia/Nemotron-3-Ultra-550b-a55b`, `nvidia/Nemotron-3_5-Lightning` (ignores JSON schema, emits reasoning; unusable for arm B).
- `response_format: json_schema` (strict) works on Super and Nano. ~1-3 s per call (Nano ~1-1.5 s, Super ~2-3 s).
- ~1 in 6 strict-JSON calls returned unparseable output (often tiny exponent floats); client retries with temperature 0.3, then the gateway ASKs.
- Early observation (prompted, plain system prompt, n=6): Nano rates `rm -rf node_modules` p_destructive 0.7; Super 0.0-0.96 depending on run.
  Coached prompt (tells the model routine actions are benign) drops it to 0.001. Both prompt variants are kept as separate arms.
- Still unknown: which Nemotron sizes are fine-tunable on Nebius AI Cloud.
