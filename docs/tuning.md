# Measurement and tuning

## The controlled comparison

```sh
make bench                         # baseline only, two process starts, cold + warm each
make bench-mtp                     # baseline/MTP paired for draft lengths 1, 2, 4
make bench-mtp ARGS="--drafts 2 --repeats 2 --tokens 128 --prompt-file local/workload.txt"
```

These controls deliberately remain **legacy single-model Qwen3.8 benchmarks**, not a multi-model
router comparison. A benchmark owns exactly one server at a time and uses the same launch
function as `serve-mtp` / `serve-baseline`, not the new `serve` router. To test router integration
use `make smoke`; do not interpret legacy benchmark data as router-switching performance.
For each draft length it runs baseline then MTP in the first repetition, MTP then baseline in
the next. Each process gets the identical prompt, seed 42, temperature 0, top-k 1, top-p 1,
min-p 0 and output budget. Smoke/bench default to native reasoning budget `0`; select another
budget with `--reasoning-budget` (`-1` unrestricted, positive values bounded). The same chosen
budget is passed to both modes and recorded in `benchmark.json` and each `run.json`. Native
completion does not add a thinking/chat template, so the workload still determines whether
reasoning is exercised. No prompt changes between modes. Draft length 4 is the serving
default, not a proven optimum. MTP can be slower; only matched measurements justify using it.

The default workload is intentionally tiny and deterministic. Supply a representative,
non-sensitive UTF-8 prompt with `--prompt-file` for meaningful work. The prompt SHA256 and byte
count are recorded (the prompt is not copied into summary JSON; server logs can still reveal
content). Native `/completion` is used for token-level timing; it does not wrap the prompt in a
chat template, so use the same prepared workload throughout a comparison. The separate `smoke-single`
check exercises legacy chat/tool behavior.

## What is reported

`results/<run>/benchmark.json` contains each raw measurement, not a misleading single average.
Per-process subdirectories contain full `server.log`, `run.json` identities/settings/memory and
`cold.json` / `warm.json` streamed responses. `run.json` includes the selected model's filename,
byte size and actual GGUF metadata under `model_check`; keep quantization fixed for a matched
comparison. Partial completed measurements survive failures.

- **Startup**: process creation through healthy readiness, separate from request timing.
- **Cold request**: first request after fresh process start with `cache_prompt=false`.
  This is *not* cold SSD/page-cache or cold Metal compilation; the OS cache is uncontrolled and
  llama.cpp's default startup warmup remains enabled.
- **Warm request**: exact repeated prompt with `cache_prompt=true`. Hybrid checkpoints may reuse
  only part of the prompt; use `cache_n` and `prompt_n`, not the label alone, to see reuse.
- **TTFT**: client request start to first streamed text/tool delta (not a role-only chunk).
- **Prefill**: server `prompt_ms`, `prompt_n` and `prompt_per_second`, excluding reused `cache_n`.
- **Decode**: `accepted_output_tokens_per_second` is the server's `predicted_per_second`, based on
  emitted tokens, **not draft proposals**. At the pin, the first token is charged to prefill,
  so the decode numerator is `predicted_n - 1`; dividing all output tokens by `predicted_ms`
  would overstate speed, especially on short runs. Total output count is also preserved.
- **End-to-end throughput**: all emitted tokens / full client request seconds, including TTFT.
- **Draft statistics**: `draft_n` and `draft_n_accepted` when exposed; otherwise `unknown`.
  Missing counters are not evidence of zero draft attempts. The real length-2 smoke benchmark
  exposed both counters; the newer IQ3 length-4 endpoint check also exposed both, including
  during image input. Draft length 1 has not been hardware-validated here.
- **Memory**: roughly 1-second sampled process RSS and system-wide swap start/end/delta/peak.
  RSS is not total unified GPU allocation; swap includes other applications and persists between
  runs. Sampled peaks can miss spikes. Missing metrics are `unknown`, never fabricated zeros.
  Server logs include Metal/KV allocation details. No per-process GPU-memory measurement is claimed.

Do not compare contended timings to an idle run or cold page-cache work to a warm run. Record
other GPU/memory activity alongside any interpretation. This recipe does not evict page cache,
purge system memory, alter system tuning or kill unrelated applications.

## Configured capacity is not occupied context

Every run defaults to 262144 context (explicit 131072..262144 is allowed) and verifies the server
reports exactly the requested size.
`--fit off` and `--no-context-shift` prevent silent shrinking/sliding; overflow is an error.
`occupied_tokens = prompt_n + cache_n + predicted_n` makes a small workload visible. The initial
integration benchmark used only 16 prompt + 16 output tokens. It says nothing about full 128K
occupied-context speed. For larger trials, supply a larger prompt and leave space for the
requested output; increase `--request-timeout` explicitly if genuinely needed. Do not run an
unrequested long-context campaign just to validate setup.

## Memory pressure and failure

Q8 K/V is deliberate, not the smallest cache. Two context checkpoints and `cache-ram=0` bound
additional caches but do not guarantee that weights + KV + compute buffers + OS fit without swap.
The tested contended M2 Pro host accumulated about 6 GiB of system swap during the short checks.
Stop with Ctrl-C if latency or pressure becomes unacceptable. Close unrelated large workloads,
then rerun the **same** recipe. Do not silently reduce context or KV precision and call that a
successful capacity comparison. The default now requests the model's 262144 metadata limit,
which carries higher memory risk on this 32 GiB host. The newer tiny IQ3/MTP4/projector check
peaked near 7.5 GiB system swap; see [validation](validation.md). Neither allocation nor a short
request proves acceptable performance at full occupancy.

With the projector, the pinned runner disables `n_cache_reuse` chunk-shift reuse, not all live
prefix reuse. The short tool-result continuation reused 357 tokens and evaluated 22 new prompt
tokens. This is not a promise about 20K history, future eviction, cache lifetime, or Pi warming.

## Verified upstream seams

At official commit `1ba568c5ad5c6c718984bd2bf4e7c71485b706f9`:

- `common/arg.cpp`: flags, `--fit off`, checkpoint/cache options, `--spec-draft-n-max`,
  `--reasoning-budget` (`-1` unrestricted native default) and `--cors-origins localhost`.
- `tools/server/server-http.cpp`: localhost CORS origin handling (not authentication).
- `common/speculative.cpp`: `none` and `draft-mtp` modes.
- `src/models/qwen35.cpp`: single nextn block and MTP tensor requirements.
- `gguf-py/gguf/gguf_reader.py`, `constants.py`: GGUF v3 directory layout/tensor names.
- `tools/server/README.md`: streaming API, `/props`, `cache_prompt` and timing fields.
- `tools/server/server-common.h`: decode denominator excludes the first generated token.

https://github.com/ggml-org/llama.cpp/tree/1ba568c5ad5c6c718984bd2bf4e7c71485b706f9

The local runner's `--list-devices` exposed **MTL0**, not the guessed device name `Metal`.
Help/device checks are performed before each launch; unsupported runners fail rather than adapt
silently. Directory checks do not prove full tensor integrity; that requires a trusted artifact
checksum and validation by the loader.
