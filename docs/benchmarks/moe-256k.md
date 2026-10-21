# Seven MoE quants at 256K capacity on M2 Pro

Measured 2026-10-20 on **Apple M2 Pro, 32 GiB unified memory**, macOS 27.0.1 (26A434).
Native llama.cpp source: `1ba568c5ad5c6c718984bd2bf4e7c71485b706f9`, unmodified throughout.
Pi: installed 1.0.3, using the existing working Node 22.22.2 executable.

## What I would choose

- **Qwen3.6 UD-Q3_K_XL with MTP4** is a good starting point for this local coding setup:
  58.3 tok/s on the short screen, 22.7 tok/s around 20K input, working image/tool/thinking
  checks, and a correct BFS file in 13.1 seconds after loading. Qwen IQ4 was close on decode
  and completed its single coding trial slightly faster; this is not enough to rank coding quality.
- **Gemma4 QAT UD-Q4_K_XL with MTP2** also works, including its separate draft head and
  image projector. It reached 53.9 tok/s short and 25.1 tok/s around 20K, but the 20K uncached
  prompt took about 89 seconds. Its coding trial took 31.8 seconds, largely generating more tokens.
- **Nemotron3.5 UD-IQ2_M, MTP off for longer inputs** was fastest on this synthetic 20K
  workload: 32.4 tok/s, roughly 49 seconds to first token uncached. It is text-only here.
  MTP4 helped the short screen but hurt the 8K/20K cases. Keep the off preset available.
- **Do not enable MTP for Nemotron UD-Q4_K_M on this configuration.** Both draft sizes failed
  with native Metal out-of-memory errors. The off preset worked, but was slower and used more memory.

These are narrow local observations, not a 25–35 tok/s guarantee for arbitrary sessions.
There is no evidence here for general coding-quality superiority or fully occupied 256K performance.

## Short decode screen

Canonical native `predicted_per_second`, accepted output tokens, excluding the first generated token
from the decode interval. Each successful new-model cell averages **four requests**: uncached/warm
in each of two repetitions; model/MTP order reverses for the second repetition. Each generated
128 actual tokens. The control has only one uncached/warm pair.

| Model / Unsloth quant | MTP off | MTP2 | MTP4 | Selected short mode |
|---|---:|---:|---:|---|
| Qwen3.6 35B-A3B / UD-Q2_K_XL | 43.0 | 46.9 | 50.5 | MTP4 |
| Qwen3.6 35B-A3B / UD-Q3_K_XL | 39.7 | 56.2 | 58.3 | MTP4 |
| Qwen3.6 35B-A3B / UD-IQ4_XS | 39.3 | 54.6 | 57.1 | MTP4 |
| Gemma4 26B-A4B QAT / UD-Q4_K_XL | 45.6 | 53.9 | 53.1 | MTP2 |
| Nemotron3.5 Lightning 30B-A3B / UD-IQ2_M | 39.7 | 46.8 | 47.6 | MTP4 |
| Nemotron3.5 Lightning 30B-A3B / UD-Q3_K_XL | 38.4 | 45.8 | 44.1 | MTP2 |
| Nemotron3.5 Lightning 30B-A3B / UD-Q4_K_M | 35.8 | **OOM** | **OOM** | Off only |
| Qwen3.8 27B / UD-IQ3_XXS control | Not run | Not run | 14.5 | Existing control |

The short input occupied 48 Qwen tokens, 49 Gemma tokens, or 52 Nemotron tokens after the actual
native chat template/tokenizer. Including output, these are **176–180 occupied tokens**, not 256K.
Warm requests reused 44/44/48 tokens respectively and evaluated 4/5/4 new prompt tokens.

| Selected mode | Uncached TTFT (s) | Warm TTFT (s) | Draft tokens accepted |
|---|---:|---:|---:|
| Qwen Q2 MTP4 | 0.288 | 0.056 | 62.5% |
| Qwen Q3 MTP4 | 0.279 | 0.052 | 76.2% |
| Qwen IQ4 MTP4 | 0.283 | 0.053 | 74.2% |
| Gemma Q4 MTP2 | 0.238 | 0.078 | 76.0% |
| Nemotron IQ2 MTP4 | 0.349 | 0.060 | 70.8% |
| Nemotron Q3 MTP2 | 0.338 | 0.059 | 74.5% |
| Nemotron Q4 off | 0.377 | 0.065 | Not applicable |
| Dense control MTP4 | 1.762 | 0.220 | 72.3% |

These TTFT values exclude model loading. A separate autoload/readiness request ran before each
experiment. Model switching and its lost KV cache are not free.

## Representative longer prefixes

Same synthetic ledger and question, native templates, two counterbalanced repetitions per configuration.
Each row averages four responses; all generated 128 tokens. Input lengths are measured by native
`/apply-template` and `/tokenize`, not estimated from character count. Whole ledger rows make the
actual count slightly exceed the target.

| Configuration | Actual input | Decode tok/s | Uncached TTFT (s) | Warm TTFT (s) | Warm cache / new prompt |
|---|---:|---:|---:|---:|---|
| Qwen Q3 off | 8,221 | 25.1 | 17.41 | 0.084 | 8,217 / 4 |
| Qwen Q3 MTP4 | 8,221 | 33.3 | 18.24 | 0.082 | 8,217 / 4 |
| Qwen Q3 off | 20,491 | 16.9 | 50.76 | 0.121 | 20,487 / 4 |
| Qwen Q3 MTP4 | 20,491 | 22.7 | 54.45 | 0.124 | 20,487 / 4 |
| Gemma Q4 off | 8,193 | 32.4 | 23.39 | 0.045 | 8,192 / 1 |
| Gemma Q4 MTP2 | 8,193 | 33.9 | 23.45 | 0.044 | 8,192 / 1 |
| Gemma Q4 off | 20,493 | 24.6 | 87.79 | 0.067 | 20,492 / 1 |
| Gemma Q4 MTP2 | 20,493 | 25.1 | 88.78 | 0.057 | 20,492 / 1 |
| Nemotron IQ2 off | 8,196 | 36.0 | 17.72 | 0.071 | 8,192 / 4 |
| Nemotron IQ2 MTP4 | 8,196 | 30.5 | 18.45 | 0.074 | 8,192 / 4 |
| Nemotron IQ2 off | 20,496 | 32.4 | 48.93 | 0.092 | 20,492 / 4 |
| Nemotron IQ2 MTP4 | 20,496 | 23.4 | 51.51 | 0.091 | 20,492 / 4 |

MTP is workload-dependent. At 20K, Nemotron's MTP4 acceptance was 41.4%, versus 70.8% short;
its decode rate fell below the off baseline. Gemma's gain at 20K was only about 2%.
Qwen Q3 retained about a 34% decode gain, but still took roughly 54 seconds before the first token
of an uncached 20K prompt. Warm repeated-prefix latency does not predict a fresh conversation's latency.

Maximum occupied input plus output here was **20,624 tokens**. Configured capacity was 262,144
for every case, including failures. A repetitive synthetic ledger is not a representative repository,
a retrieval/quality test, or proof of accuracy at this length.

## Actual Pi coding trials

Task: write `BFS.py` containing `TreeNode(val, left=None, right=None)` and iterative, in-place
`invert_tree_bfs(root)`. The exact public prompt is in `examples/compare-prompts.json`.
Each trial used a new temporary work directory and temporary agent directory containing only the
local provider configuration and dummy key. No personal Pi configuration was read or changed.

Documented CLI flags: `--print --mode json --no-session --offline --no-extensions --no-skills
--no-context-files --no-prompt-templates --no-themes --no-approve --tools read,write,edit`, with an
explicit `llamacpp-qwen38/<id>` and `--thinking low`. All requests used source-card sampling, not the
greedy performance-screen sampling. The trial deadline was 600 seconds, with whole-process-group
cleanup; the model response allowance was 4096 tokens, reasoning budget 1024 per thinking block.
Those are per-response/per-block controls, not a global token quota across tool turns.

| Quant / mode | Pi wall time (s) | Generated tokens, all turns | Tool calls | Independent result |
|---|---:|---:|---|---|
| Qwen Q2 / MTP4 | 15.81 | 568 | write | Pass |
| Qwen Q3 / MTP4 | 13.10 | 370 | write | Pass |
| Qwen IQ4 / MTP4 | 11.91 | 391 | write | Pass |
| Gemma Q4 / MTP2 | 31.82 | 1,267 | write | Pass |
| Nemotron IQ2 / MTP4 | 23.49 | 841 | write | Pass |
| Nemotron IQ2 / off | 17.40 | 470 | write | Pass |
| Nemotron Q3 / MTP2 | 28.10 | 1,001 | write, edit, read | Pass |
| Nemotron Q4 / off | 40.79 | 1,193 | write | Pass |

Wall time includes Pi startup, prompt evaluation, generation, tool writes, and final answer, but excludes
model autoload and the independent tests. Corresponding autoload/readiness requests took about 20–55
seconds; per-trial values are in the companion JSON. All returned exit code 0, with a nonempty file.
All emitted actual thinking content, although native usage reported zero separately counted reasoning
tokens; total output counts include the responses and must not be interpreted as answer-only counts.

Every generated file was read before execution: only benign class/function definitions and
`collections`, `typing`, or future-annotations imports; no network/filesystem/process calls or executable
examples. Each then passed independent checks for `None`, constructor children, a golden mirror,
50 deterministic random valid trees, every original node's value and swapped edge identities,
double inversion, and a 10,000-node chain. Source inspection confirmed queue/`popleft` breadth-first
iteration, rather than treating a deep-chain pass alone as proof of BFS.

No external repair prompt was needed. Nemotron Q3 made its own edit/read within its first trial; that
cost is included, not hidden as a clean one-write result. Each quant has one measured task, not a
statistical success rate. The later final-provider Qwen smoke is a configuration check, not another
sample in this table.

**This is not the full interactive orchestration harness.** There was no Bash, cloud model, MCP server,
extension, saved history, skill, or project context. Read/write/edit are still filesystem-capable tools,
not a security sandbox; the controlled prompt and inspected tool paths stayed within the fresh workspace.
All trial directories were removed after preserving private logs and code.

An initial isolated-launch batch never reached any model: its sanitized PATH selected a broken,
pre-existing Homebrew Node executable with a missing `libllhttp` dylib. Those seven failures are retained
separately in `results/moe-pi/`; they are harness-startup failures, not coding failures. The driver now
selects the already-working Node from its caller's PATH. No installation, dependency patch, or machine
configuration change was made. The successful trials above were fresh launches after that correction.

## Capabilities and memory

All seven quants passed separate streamed thinking-plus-answer and constrained tool-call/tool-result
checks with MTP off. The selected viable MTP variants passed those checks again. A generated 128×64
red-left/blue-right PNG was correctly described by all three Qwen quants and Gemma, both off and in
the selected MTP configuration. This is a minimal image sanity check, not OCR/general vision scoring.
Nemotron was configured text-only and its native props did not advertise vision.

Live draft counters, not model-card marketing or filename assumptions, verified MTP for Qwen, Gemma,
and Nemotron. Gemma used the matching separate `mtp-gemma-4-26B-A4B-it.gguf`; Qwen and Nemotron used
embedded heads. The older Qwen distributor card cautions about MTP with projectors; this particular
native pin passed both the text-MTP counter checks and image requests with its matching projector.
That does not prove every image token was speculatively decoded.

| Configuration | Sampled peak router + model RSS (GiB) | Largest case-end system swap growth (GiB) |
|---|---:|---:|
| Qwen Q2 MTP4 | 16.24 | 0.00 |
| Qwen Q3 MTP4 | 20.48 | 0.00 |
| Qwen IQ4 MTP4 | 20.92 | 0.00 |
| Gemma Q4 MTP2 | 18.04 | 0.00 |
| Nemotron IQ2 off | 18.64 | 0.00 |
| Nemotron IQ2 MTP4 | 19.98 | 0.00 |
| Nemotron Q3 MTP2 | 21.30 | 0.00 |
| Nemotron Q4 off | 23.76 | 0.11 |
| Nemotron Q4 MTP2, failed | 23.80 | 0.45 |
| Nemotron Q4 MTP4, failed | 24.83 | 0.00 |
| Dense control MTP4 | 21.00 | 0.60 |

Memory values cover the short/long performance runs, sampled once a second across all members of the
owned router process group, including its native model child. RSS is **not** Metal allocation size or
whole-system memory use; mapped weights, compression, eviction, shared pages, and missed sub-second
peaks limit interpretation. No Qwen KV-size formula was applied to Gemma's different architecture.
System swap was already about 6.4 GiB before the first capability run; a zero delta does not mean
zero memory pressure. No other llama-server process was allowed; other desktop/GPU activity was not
instrumented, so these are ordinary, potentially contended desktop measurements, not an isolated-lab ceiling.

Nemotron Q4 MTP2/4 produced `kIOGPUCommandBufferCallbackErrorOutOfMemory` and HTTP/SSE compute errors.
Each model process group was stopped; no context or cache-precision fallback was attempted. The original
two-repetition screen attempted each failing mode twice, transparently retained in the raw results.
The final driver skips subsequent repetitions after a failed case. No further Q4 MTP runs were made.
The pressure guard also stops its owned group if swap grows more than 2 GiB without streamed progress
for 60 seconds; it did not trigger in this campaign.

## Fixed runtime and sampling

Native router on `127.0.0.1:39483`, `models-max=1`, autoload, no concurrent model workloads. Every
candidate used `ctx-size=262144`, `parallel=1`, `n-gpu-layers=all`, `device=MTL0`, flash attention on,
Q8 K/V, fit off, no context shift, two context checkpoints and no RAM prompt-cache allocation.
Candidate draft contexts also requested Q8 K/V. Native rendered child arguments and live props were
retained; task-default props alone can misreport speculative/reasoning defaults on this pin.

For decode screens the semantic prompt, output target 128, seed 42, temperature 0, top-k 1,
top-p 1, min-p 0, repeat penalty 1 and presence penalty 0 were matched. Chat formatting is family-specific;
quant/MTP numerics can change continuations, so this is not a claim of bit-identical outputs.

“Cold” in artifact names means `cache_prompt=false` and native `cache_n=0` **after readiness**, not cold
SSD or first-ever model inference. “Warm” immediately repeats the same input with `cache_prompt=true`.
The OS file cache was not flushed. There is no cold-storage bandwidth claim.

Pi/native user-facing sampling follows the pinned distributor cards:

| Family | Thinking/coding temperature | top-p | top-k | Other controls |
|---|---:|---:|---:|---|
| Qwen3.6 precise coding | 0.6 | 0.95 | 20 | min-p 0, presence 0, repeat 1; preserve thinking |
| Gemma4 QAT | 1.0 | 0.95 | 64 | min-p 0, presence 0, repeat 1 |
| Nemotron3.5 | 1.0 | 0.95 | 0 (disabled) | min-p 0, presence 0, repeat 1 |

Qwen's Pi off override uses the card's non-thinking general-task values: temperature 0.7, top-p 0.8,
presence 1.5. This off override is not part of the low-thinking coding timing table.
The 1024 reasoning budget and 4096 response allowance are bounded local experiment settings, not the
cards' much larger recommended evaluation budgets.

## Ready configurations and reproduction

`local/models.ini` remains private and preserves the original `qwen3.8-27b` control unchanged, adding
only the tested modes below. All paths in it are relative symlinks under ignored `.runtime/models/`.
`examples/pi-models-moe.json` is the copyable public provider fragment; merge its model entries
into an existing provider rather than overwriting personal configuration. It includes the unchanged control and the validated candidate IDs:

- `qwen36-q2-k-xl-d4`, `qwen36-q3-k-xl-d4`, `qwen36-iq4-xs-d4`
- `gemma4-q4-k-xl-d2`
- `nemotron35-iq2-m-d0`, `nemotron35-iq2-m-d4`, `nemotron35-q3-k-xl-d2`, `nemotron35-q4-k-m-d0`

Keep provider `llamacpp-qwen38` and its existing port. Merge entries by ID, preserving unrelated
providers and models. Nothing automatically changes a default model or starts an always-on server.
Image input is declared only for candidates with the actual image check. No cache lifetime is declared.

The installed Pi schema and OpenAI-completions implementation were checked. New families use
`thinkingFormat: "chat-template"`, `chatTemplateKwargs.enable_thinking: {"$var":"thinking.enabled"}`,
and `supportsReasoningEffort: false`; Qwen also preserves thinking. This is a binary thinking switch,
not an invented low/medium/high capability map copied from the Qwen3.8 control. All non-off levels
use the same 1024 budget. The final provider fragment was exercised through the actual CLI.

To reproduce, first acquire the exact artifacts listed in `moe-256k-data.json` independently; the
comparison command never downloads or checks an inventory manifest at serve time. Start from the
native options in `example.ini`, give each quant/mode its own section ID, and point its relative model
symlink at the corresponding file. Use `spec-type=none` for off or `draft-mtp` with
`spec-draft-n-max=2/4`, `spec-draft-n-min=0`, `spec-draft-type-k=q8_0`, `spec-draft-type-v=q8_0`.
Gemma additionally needs `model-draft`, `spec-draft-ngl=all`, and `spec-draft-device=MTL0` for its head.
Qwen/Gemma need the matching `mmproj`; Nemotron uses `no-mmproj=true`. Paths are local choices.

Example commands after preparing ignored `local/compare.ini` and a local provider fragment:

```sh
python3 scripts/compare.py capabilities --models qwen36-q3-k-xl-d0 qwen36-q3-k-xl-d4 --output capability-run
python3 scripts/compare.py performance --models qwen36-q3-k-xl-d0 qwen36-q3-k-xl-d2 qwen36-q3-k-xl-d4 --repeats 2 --output short-run
python3 scripts/compare.py performance --models qwen36-q3-k-xl-d0 qwen36-q3-k-xl-d4 --sizes 8192 20480 --repeats 2 --request-timeout 900 --output context-run
python3 scripts/compare.py pi --config local/models.ini --pi-config examples/pi-models-moe.json --models qwen36-q3-k-xl-d4 --output coding-run
python3 -m unittest discover -s tests -p test_compare.py -v
python3 tests/replay_compare.py results/short-run results/context-run
```

Use a fresh output name; evidence directories are never silently overwritten. `--budget` is a finite
between-case scheduling budget, with separately bounded requests/trials and cleanup. Do not run a
second server or benchmark concurrently. Generated `BFS.py` is **not** executed by the driver. Read it,
reject unsafe code, then explicitly invoke `tests/check_bfs.py` with its reviewed SHA256. That checker
is not a sandbox. The code and raw results are retained before temporary workspaces are removed.

## Evidence and limits

Public aggregate data and exact asset revisions/hashes: `docs/benchmarks/moe-256k-data.json`.
Raw SSE/request bodies, rendered native argv/props, model-group RSS/swap samples, full server logs,
Pi JSON events and generated code remain in ignored `results/`:

- `moe-cap-qwen2`, `moe-cap-off`, `moe-cap-mtp`: capability checks.
- `moe-screen-first`, `moe-screen-rest`: all seven quants, off/MTP2/MTP4, planned two repetitions.
- `moe-context`: off/selected-MTP, 8K/20K, all three families, two repetitions.
- `moe-dense-control`: bounded comparable short control pair only.
- `moe-pi`: failed Node startup attempts; no inference.
- `moe-pi-working-node`, `moe-pi-rest`, `moe-pi-nemotron-long-choice`: successful coding trials and independent correctness JSON.
- `moe-final-profile-smoke`: final private INI/public provider integration check, excluded from timing aggregates.

Read-only replay checked **126 measured responses** against their raw SSE, actual tokenized request,
native context/Metal/Q8 arguments, warm/cold prefix counts and live MTP counters. No throughput was
computed from drafted-token counts. No raw Pi transcript should be published without redaction.

Not run: fully occupied 256K; long inputs for every quant/draft length; long dense-control cases;
multiple coding seeds/tasks; an interactive full-harness trial; broad vision or coding-quality suites;
repair-after-test-error prompts (none were needed). No model downloads, upstream modifications,
new dependencies, sysctl changes, user-application termination, commits or publication were performed
by the comparison driver. All owned native/Pi process groups were reaped, and the loopback port was
free at handoff.

Pinned distributor cards:

- https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/README.md
- https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/README.md
- https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/blob/f2d3fe3694501008786e81e5f20360cbf715496a/README.md

Model licenses are separate from the recipe license: Qwen/Gemma cards identify Apache 2.0;
Nemotron identifies OpenMDW 1.1. Hardware claims and benchmark scores in those cards are their
publishers' results, not measurements reproduced here.
