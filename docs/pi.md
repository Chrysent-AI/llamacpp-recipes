# Connect Pi to the local endpoint

## One provider, all measured models

[`examples/pi-models.json`](../examples/pi-models.json) is the complete, ready-to-copy JSON:
**one `llamacpp-qwen38` provider, one port, nine model/mode entries**, each with its own
sampling, thinking controls and input capabilities. It includes the dense control and all
validated MoE choices. [`examples/pi-models-moe.json`](../examples/pi-models-moe.json) is an
identical compatibility copy for existing links; use `pi-models.json` as the main example.
Neither file contains anyone's personal configuration or credentials.

1. Prepare your private `local/models.ini` from [`example.ini`](../example.ini), retaining only
   sections whose language weights and required sidecars are available. The INI section names
   must match Pi model IDs. Adjust paths only in your private copy.
2. Run `make serve` and leave that terminal open. It starts the native llama.cpp router at
   `http://127.0.0.1:39483/v1`, with at most one model resident. No model-path environment variable
   is required. Do not run another loader/server alongside it.
3. Pi's default configuration path is `~/.pi/agent/models.json`. For a new file, the complete
   example is a starting point. For an existing file, merge its provider or individual model
   entries by ID, preserving unrelated providers and settings. **Do not overwrite your file.**
4. Open `/model` to reload and select `llamacpp-qwen38/<model-id>`, then select `/thinking low`
   to enable bounded thinking. Saving a default model or thinking level is a separate choice.

The repository does not automatically edit Pi settings. The example's `apiKey: "local"` is a
non-secret placeholder that makes the provider selectable; it does not authenticate the server.
Keep the listener on loopback. See [setup and security limits](setup.md).

## Which entry to start with

These are **measured starting points, not universally optimal settings**. The table gives
accepted-output decode speed for the synthetic short and approximately 20K prompts, not total
request latency or general coding quality. See the [full comparison](benchmarks/moe-256k.md)
for methods, model-loading costs, cold/warm results and all quantizations.

| Model ID | Method | Short / ~20K tok/s | Intended starting use |
|---|---|---:|---|
| `qwen36-q3-k-xl-d4` | Embedded MTP, draft 4 | 58.3 / 22.7 | Coding with image support; longer-input behavior measured |
| `qwen36-iq4-xs-d4` | Embedded MTP, draft 4 | 57.1 / not measured | Higher-precision Qwen alternative; no long-context quality claim |
| `qwen36-q2-k-xl-d4` | Embedded MTP, draft 4 | 50.5 / not measured | Lower-memory Qwen alternative; evaluate task quality |
| `gemma4-q4-k-xl-d2` | Separate MTP head, draft 2 | 53.9 / 25.1 | Vision-capable alternative; slower uncached 20K prefill |
| `nemotron35-iq2-m-d0` | MTP off | 39.7 / 32.4 | Fastest measured longer text-only workload |
| `nemotron35-iq2-m-d4` | Embedded MTP, draft 4 | 47.6 / 23.4 | Short-input alternative; slower than off at 20K |
| `nemotron35-q3-k-xl-d2` | Embedded MTP, draft 2 | 45.8 / not measured | Higher-precision Nemotron comparison |
| `nemotron35-q4-k-m-d0` | MTP off | 35.8 / not measured | Q4 alternative; MTP2/4 failed Metal allocation |
| `qwen3.8-27b` | Dense control, MTP draft 4 | 14.5 / not measured | Original baseline, not the faster MoE path |

`d0` means **MTP off**, not thinking off. MTP is configured in the native INI, not by Pi's
thinking selector. Qwen3.6 and Nemotron use embedded MTP heads; Gemma needs its separate
matching drafter. Qwen and Gemma image entries also need their matching projector. Remove
`"image"` from a model's `input` list if you disable its projector. Nemotron is text-only here.

## Parameters and precedence

| Family | Thinking/coding temperature | top-p | top-k | Thinking interface |
|---|---:|---:|---:|---|
| Qwen3.8 dense control | 1.0 | 0.95 | 20 | Native low/medium/xhigh effort mapping |
| Qwen3.6 MoE | 0.6 | 0.95 | 20 | Binary chat-template thinking switch |
| Gemma 4 QAT MoE | 1.0 | 0.95 | 64 | Binary chat-template thinking switch |
| Nemotron 3.5 MoE | 1.0 | 0.95 | 0, disabled | Binary chat-template thinking switch |

Thinking defaults additionally use min-p 0, presence penalty 0 and repeat penalty 1. The
model cards' `repetition_penalty` is expressed as llama.cpp's `repeat_penalty`. Qwen entries
have their separately documented nonthinking sampling override. Public source-card links and
exact artifact revisions are in the comparison report; settings are not interchangeable across
all Unsloth model families.

The native INI controls model/projector/drafter loading, allocated context, KV precision and
server defaults. Pi's per-model `samplingParams` are sent with requests and override server
sampling defaults. Changing temperature only in INI will not change a Pi request that supplies
its own temperature. Keep `baseUrl`, model IDs, capabilities and `contextWindow` aligned with
the server. Do not put model-specific CLI flags on the shared router launch: they override INI.

All example entries declare **262144 context**, **4096 total output tokens** and a
**1024-token per-thinking-block budget**. Context is not output allowance. The total output
includes thoughts plus the answer and does not guarantee a complete answer if exhausted.
The thinking budget is not an immutable aggregate quota; server `--n-predict` is likewise a
client-overridable fallback. Avoid competing `max_tokens`, `max_completion_tokens` or
`n_predict` keys in `samplingParams`.

For Qwen3.8, Pi minimal/low map to low, medium to medium, and high/xhigh to xhigh; effort is an
instruction, not a token budget. The three new families instead use
`thinkingFormat: "chat-template"` and `thinking.enabled`: every non-off Pi level enables the
same bounded thinking behavior. They do not claim distinct native low/medium/high effort modes.
Use `/thinking off` to disable thinking. Keep the example's family-specific overrides intact:
Pi merges sampling shallowly, and nested template kwargs are replaced rather than deep-merged.
`--reasoning-format deepseek` keeps streamed thoughts separate in `reasoning_content`.

## Cache, verification and publication

The native router unloads an idle previous model when another is requested. **Unloading loses
its KV cache**; switching back requires loading and prompt evaluation again. Prefix reuse
requires matching token history. No cache lifetime or Pi cache-warming eligibility is declared.
Configured 256K capacity is not proof of fully occupied 256K performance: the comparison reached
20624 occupied tokens. Start with a non-sensitive task before using important files.

These examples were checked against Pi 1.0.3 and the pinned llama.cpp build. The measured Pi
coding trials were isolated read/write/edit-only sessions, not full interactive orchestration
or a filesystem sandbox. See [validation](validation.md) and [benchmark limits](benchmarks/moe-256k.md).

Share the public examples, not your full personal `models.json`, authentication files or raw
session logs. `local/`, `.runtime/` and `results/` remain ignored. Review staged files before
publishing; do not add personal home paths, named volumes, host/share names or private addresses.
Generic relative paths, localhost and public model names/revisions are intentional.
