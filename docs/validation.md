# Endpoint validation

## Native router control check (2026-10-20)

Same official pin and M2 Pro / 32 GiB host as below, with concurrent downloads/desktop activity.
`make doctor` listed only the private Qwen3.8 control as unloaded. `make smoke` used the same
`router()` / `router_command()` path as `make serve`, with no top-level model or global sampling
flags. A tiny thinking-disabled chat request auto-loaded the existing IQ3 control and returned
nonempty text. `/props?model=qwen3.8-27b` reported 262144 context, one slot and vision enabled.
Rendered child arguments preserved Q8 K/V, all layers on MTL0, flash attention, fit off,
no context shift, two checkpoints, cache-ram 0, MTP4, low effort, budget1024 and output4096.
The request's timing payload showed four MTP proposals (zero accepted in this tiny answer).

At this pin `/props` copies sampling into a fresh task object, so its task-level `n_predict`,
`reasoning_format` and `speculative.types` are **not** reliable reports of launch settings.
This was checked in `get_res_props` in `tools/server/server-context.cpp`; rendered child
arguments and live draft counters supply the additional evidence, not those placeholder fields.

`POST /models/unload` returned success; `/models` returned to unloaded and the recorded child
PID no longer existed. The outer process group was then verified absent. No existing server
was killed. Native parsing of the public example listed all four section IDs unloaded without
loading the absent candidate weights. A deliberately unknown INI key failed startup with a
nonzero helper exit and no surviving server. Offline tests cover the production router entry,
no-global-model-flags precedence, explicit config, errors, and child-group cleanup.

This is a launch/integration check, not a speed comparison or full-context workload. No new
model-family capability, image-answer quality, Pi UI interaction, or multi-model eviction
performance is inferred. The scheduler's idle-LRU/busy-queue behavior is source-verified,
not a hardware switching trial. Private full logs and JSON live in the result directory printed
by each command; they are not publishable without review/redaction.


## IQ3 / MTP4 / 256K / projector (2026-10-20)

Environment: Apple M2 Pro, 32 GiB, Darwin 27.0.0, official llama.cpp commit
`1ba568c5ad5c6c718984bd2bf4e7c71485b706f9` (build 11768). Existing
`Qwen3.8-27B-UD-IQ3_XXS.gguf`, 10934860704 bytes; matching F16 projector size and SHA256
verified as documented in [setup](setup.md). No language weights or upstream code changed.
Other desktop/agent workloads remained active; exact GPU contention was not measured. These
are **contended-environment timings**, not intrinsic latency or comparative benchmark results.

One owned single-model server used the then-production (now legacy) `server()`/`command()` path with default serving settings:
262144 context, one slot, MTP draft4, Q8 K/V, all layers on MTL0, flash attention, two checkpoints,
no automatic fit, loopback39483, budget1024, low effort, output fallback4096, matching projector.
No silent context reduction or MTP fallback occurred. No concurrent llama-server was present.
The server process group was verified gone afterward.

Private local evidence: `results/endpoint-1792517765155359000/` contains `server.log`, `run.json`,
`props.json`, `embedded-template.jinja` and `validation.json`. Logs may contain local paths and
must be redacted before sharing. The extracted template came from the loaded GGUF via `/props`,
not an online template assumed to match. It supports low/medium/xhigh (also aliases high),
`enable_thinking`, `preserve_thinking`, image markers and tool messages.

| Check | Result |
|---|---|
| Readiness and `/props` | 262144 context, one slot, vision true; startup 4.94 s |
| Streamed reasoning + answer | Nonempty `reasoning_content`; 17 + 25 → 42; 5.21 s |
| Required streamed echo call | Valid ID/name and JSON argument `metal-ready`; 7.58 s |
| Tool-result continuation | Returned `metal-ready`; 3.19 s; cache_n357 / prompt_n22 |
| Synthetic image | Generated 64×64 solid-red PNG, no color hint in prompt; answer `Red`; 3.56 s |
| Thinking off | Explicit none/false overrides; `ready.` with no reasoning; 1.10 s |

Text/tool requests used 256 total tokens and 48 reasoning tokens; image used 96/16; off used
32 total tokens. Thus the check exercised real bounded reasoning/closure without waiting out
1024 thoughts. The server retained the configured default1024 throughout. Total generated
counts were 46, 51, 34, 20 and 3. MTP draft generated/accepted counts were 48/33, 44/39, 36/24,
24/13 and 8/0 respectively. This confirms speculation remained active for image input.

Sampled process RSS peak: 22887563264 bytes (21.3 GiB). System-wide swap started at
5406982144 bytes (5.0 GiB), ended at 8027373568 (7.5 GiB), peaked at 8035762176; delta about
2.44 GiB. RSS is not total unified GPU allocation; swap includes unrelated applications.

The runner warned that grounding needs at least 1024 image tokens and emitted non-consecutive
position warnings while processing the image. The small color response succeeded with live MTP
counters and no truncation. These warnings are retained in the log, not treated as evidence of
broad image correctness. No video, grounding, large-image, full256K occupied-context, long20K
cache, or actual Pi UI/harness session was tested. The earlier Q4 length-2 budget0 smoke is
separate historical evidence, not the basis for these reasoning/vision claims.

## Offline coverage

The stdlib tests exercise default model preference versus explicit-path freedom, serve MTP4
versus baseline controls, projector selection/missing paths/text-only, port/context/sampling/
reasoning command construction, benchmark budget and scheduling, and existing lifecycle checks.
Mock tests prove plumbing only; the hardware checks above are separate. `make check` remains
the offline final gate. No new test framework or package is required.
