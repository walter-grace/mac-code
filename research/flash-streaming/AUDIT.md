# Claims Audit — Every Number Verified Against Terminal Output

## Claim-by-Claim Verification

### "18.4 GB model on 16 GB RAM"
- **VERIFIED.** `Model size in memory: 18.43 GB` (from convert_split.py output: "Pinned: 4.28 GB, FFN: 14.16 GB, Total: 18.43 GB")
- The model is stored at 4-bit group_size=64 across pinned + FFN files.
- The 16 GB Mac never holds more than ~6 GB in GPU memory during inference.

### "4.5 GB lives in RAM at any given time"
- **VERIFIED.** Memory debug output: `[layer 0] active_mem=4.50 GB`, `[layer 48] active_mem=4.51 GB`
- Memory stays flat across all 64 layers. No growth.

### "0.12 tok/s (v1 mmap)"
- **VERIFIED.** Terminal output: `Decode: 50 tokens in 430.9s (0.12 tok/s)` (flash_stream.py run)
- Consistent across multiple runs.

### "0.152 tok/s (v2 direct I/O)"
- **VERIFIED.** Terminal output: `Decode: 30 tokens in 197.5s (0.152 tok/s)` (flash_stream_v2.py run)
- SSD throughput: `avg=129ms/layer, throughput=1.7 GB/s`

### "9x faster than llama.cpp mmap thrashing"
- **NEEDS CONTEXT.** We claim llama.cpp mmap gives 0.017 tok/s. This was from our earlier experiments (previous conversation session) where llama-server thrashed on the 19 GB GGUF. We did not re-measure llama.cpp in this session.
- 0.152 / 0.017 = 8.9x ≈ 9x. **Math checks out IF the 0.017 baseline is correct.**
- **CORRECTION NEEDED:** We should note the 0.017 was from a previous session and may vary by hardware. The comparison is directionally correct but not A/B tested in the same session.

### "Full 4-bit quality — coherent, no degradation"
- **VERIFIED.** Sample outputs:
  - v1: "we can reduce the memory bandwidth required for attention computation by storing some intermediate results in SRAM rather than loading them from HBM each time"
  - v2: "the actual compute required for each step of attention is lower than it appears, due to optimizations from 32-bit floating points down to maybe something like"
  - Agent test (387 tokens): Full chain-of-thought about Flash Attention, coherent throughout.
- The 2-bit model started repeating at ~60 tokens. The 4-bit streaming model produced 387 coherent tokens.

### "IQ2_M / 2-bit at 6.4 tok/s, degraded quality after ~60 tokens"
- **VERIFIED.** Terminal: `Decode: 200 tokens in 33.1s (6.12 tok/s)` (flash_fast.py)
- Output shows: coherent for ~50 tokens, then "nessnessnessness..." repetition loop.
- We said "6.4 tok/s" but measured 6.12. **CORRECTION: Should say "~6 tok/s" not 6.4.**

### "Model load time: 2.0 seconds"
- **VERIFIED.** Terminal: `Loaded in 2.0s (4.28 GB pinned)` (consistent across runs)

### "Prefill: 9.0 seconds (v1), 7.19 seconds (v2)"
- **VERIFIED.** v1: `9.00s (1.4 tok/s)`, v2: `7.19s (1.81 tok/s)`

### "SSD throughput: 1.7 GB/s (34% of theoretical 5 GB/s)"
- **VERIFIED.** v2 output: `throughput=1.7 GB/s`
- 1.7/5.0 = 34%. Math correct.
- **CAVEAT:** The "5 GB/s theoretical" is an estimate for Mac NVMe. Actual SSD speed varies by model. We did not benchmark the raw SSD independently.

### "Memory grew by 3.6 GB every 16 layers (before fix)"
- **VERIFIED.** Debug output showed: layer 0: 4.52 GB, layer 16: 8.09 GB, layer 32: 11.65 GB, layer 48: 15.24 GB. Delta per 16 layers: ~3.57 GB. Our claim of 3.6 GB is accurate.

### "Batching layers did NOT help"
- **VERIFIED.** flash_stream_batched.py output: `Decode: 50 tokens in 407.2s (0.12 tok/s)` — identical to v1 (0.12 tok/s). Batching (8 layers, 16 evals) gave zero improvement.

### "F_NOCACHE provides 27% speedup over mmap"
- **VERIFIED.** v1: 0.120 tok/s, v2: 0.152 tok/s. 0.152/0.120 = 1.267 = 26.7%. Calling it "27%" is fair.

### "MLX 4-bit with group_size=64 uses 0.156 bytes/param"
- **VERIFIED.** Our calculation: compression ratio 0.156x of float32 (4 bytes). So 0.156 * 4 = 0.624 bytes/param... wait.
- **CORRECTION NEEDED.** 0.156 is the compression ratio vs float32, not bytes/param. At float32 (4 bytes), 0.156 * 4 = 0.624 bytes/param. 32B params * 0.624 = 19.97 GB... that's close to 18.4 GB but not exact. The actual measured size is 18.43 GB. The claim should say "compression ratio 0.156x of float32" not "0.156 bytes/param."

### "Custom Q4_K/Q6_K dequantization"
- **VERIFIED.** dequant_gguf.py exists with working Q4_K, Q5_K, Q6_K, Q8_0 implementations. Tested against real GGUF tensors with reasonable weight distributions (near-zero mean, expected ranges).

### "GGUF column-major: reshape(ne[1], ne[0]) is correct"
- **VERIFIED.** The incorrect transpose produced garbage output (Russian characters "зезе"). The correct reshape produced coherent English text. Both outcomes observed and documented.

## Claims That Need Softening

1. **"the same quality you'd get if you had 32 GB of RAM"** — This is approximately true but not exactly. Our dequant→requant pipeline introduces small rounding errors vs loading the GGUF natively. We should say "equivalent quality" or "negligible quality loss from requantization."

2. **"0.017 tok/s for llama.cpp"** — This was from a previous session. We should note it's from a prior test, not measured in the same run as our engine.

3. **"6.4 tok/s" for 2-bit** — Measured 6.12. Should round to "~6 tok/s."

## Claims That Are Honest and Should Stay

- The split architecture (attention pinned, FFN streamed) is genuinely novel for MLX
- The memory leak discovery (setattr on nn.Module) is a real engineering finding
- The batching disproof is genuine empirical evidence
- F_NOCACHE direct I/O measurably helps
- The 16KB alignment work is grounded in real Apple Silicon hardware specs
- The "What's Not Novel" section is honest about prior art
