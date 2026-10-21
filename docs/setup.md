# Setup and operations

## Prerequisites

Apple Silicon macOS, Xcode command-line tools with Metal support, CMake, Ninja, Git, Python 3.9+
and Make. The recipe does not install these globally. The tested host is M2 Pro with 32 GiB on
macOS 27.0.1. Start with other large models stopped and enough memory available; the real checks
on a contended host incurred swap. Never use concurrent model servers for comparison.

## Native router configuration

`make serve` launches the pinned runner without a top-level model, reading ignored
`local/models.ini`. No environment variable is required. Before the first launch:

1. Create `local/` and `.runtime/models/` (both ignored).
2. Create symlinks in `.runtime/models/` to your existing language model and matching projector.
   Use the generic names shown in `example.ini`; do not copy weights into the repository.
3. Copy `example.ini` to `local/models.ini`, retaining only the global section and sections
   for files actually available. Never overwrite an existing private INI without reviewing it.
4. Run `make doctor`, then `make serve`. Override with
   `make serve ARGS="--config local/other.ini"` when deliberately selecting a different INI.

INI paths resolve relative to the **repository root**, not the INI directory. The helper sets
that working directory even when invoked from elsewhere. Section names are request model IDs;
there is no Qwen-only metadata gate in router mode. The native loader validates each model
when loading it. Doctor starts the real router and lists models without deliberately loading
one; keep `load-on-startup = false` for this behavior. It validates native INI syntax, not full
weight integrity, MTP compatibility, or whether every candidate can fit in memory.

Only router settings go on the CLI: `--models-preset`, `--models-max 1`, `--models-autoload`,
loopback host, port and localhost CORS. Native precedence is CLI > model section > `[*]`.
Put context, sampling, projector, reasoning and draft settings in INI. The helper rejects
legacy model flags on router commands instead of silently ignoring or globally applying them.
Ambient `LLAMA_ARG_*` variables are removed. Native cached models can also appear in `/models`;
this helper does not create a registry or modify the native cache. Use only trusted local-file
presets: upstream `hf-repo`/URL options and native management endpoints can initiate downloads.
The supplied INI has none of those download options. INI is configuration, not a shell script.

`make smoke` uses the same router launch as serve: lists the selected model unloaded, sends a
16-token-limit request with thinking disabled, reads `/props?model=...`, checks 262144 context
and one slot, calls `/models/unload`, and verifies the native child exited. It records properties,
rendered model arguments and responses privately. Use `--model-id SECTION` for another validated
256K preset. A config enabling startup loads is intentionally incompatible with this smoke's
unloaded-first check. `--startup-timeout` bounds readiness/unload; `--request-timeout` bounds
HTTP requests including lazy loading. `make doctor` never substitutes the legacy model probe.

Native admission enforces one resident model. When a different model is requested it evicts an
idle LRU model, or queues while the resident is busy. Unload discards KV state, so switching back
requires re-prefill. No automatic memory policy, context reduction, engine patch or restart
persistence is promised. Inspect `GET /models` and model-specific properties for real state;
use `POST /models/unload` with `{"model":"qwen3.8-27b"}` to unload explicitly.

### Measured presets and evidence

`example.ini` contains the nine measured presets whose IDs match
`examples/pi-models-moe.json`: the Qwen3.8 control, three Qwen3.6 quants, Gemma 4 QAT,
and four Nemotron quant/mode combinations. Qwen3.6 uses precise-coding thinking sampling
(temp .6/top-p .95/top-k 20), Gemma uses standardized sampling (1/.95/64), and Nemotron
uses temp 1/top-p .95 with top-k/min-p disabled. These are model-card starting points,
not measured optimal sampling settings. The common 4096 output fallback/1024 reasoning
budget are local bounds, not the publishers' large-output evaluation recipes.

The measured Qwen3.6 presets enable MTP4 and their matching projector. Gemma enables MTP2,
its matching projector and a separate head via native `model-draft` (alias `spec-draft-model`).
Nemotron is text-only: IQ2 is offered both with MTP off and MTP4, Q3 with MTP2, and Q4
with MTP off because its MTP2/4 runs failed Metal allocation. Only copy sections whose
weights and required sidecars you have; the example never downloads them.

Model cards consulted (2026-10-20):
- https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF
- https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF
- https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF

Native syntax/precedence and load/unload endpoints were checked in the pinned
`tools/server/README.md`, `server-models.cpp`, `common/preset.cpp` and `common/arg.cpp`.
The [comparison report](benchmarks/moe-256k.md) records actual MTP, image and tool checks,
throughput, failures and limits. All cases used 262144 configured capacity, but maximum
measured occupancy was 20624 tokens; neither copying the example nor the short checks
prove full-context performance or memory headroom on another machine.

## Runner pin and legacy single-model selection

`scripts/recipe.py` pins official llama.cpp source at commit
`1ba568c5ad5c6c718984bd2bf4e7c71485b706f9`. There is no model manifest or required model
revision, filename, size or checksum. For **legacy single-model commands only**, set `MODEL_PATH` to an existing compatible Qwen3.8-27B
GGUF, including a renamed file or a flat path. Without it, discovery searches
`/Volumes/*/models/unsloth/**/Qwen3.8-27B*.gguf`, preferring the exact basename
`Qwen3.8-27B-UD-IQ3_XXS.gguf`. Exactly one preferred candidate is required; if none exists,
exactly one other candidate is accepted. Ambiguity fails with an explicit-path request.

Set `MODEL_PATH` in your shell to the existing file, or let mounted-volume discovery find it.
No helper downloads weights or scans arbitrary home folders. A local symlink in `.runtime/`
lets the runner use a repository-relative model name without moving or modifying model bytes.
Do not commit that symlink, local prompts or logs.

```sh
make build
make doctor-single ARGS="--mtp"
MODEL_PATH=/path/to/model.gguf make doctor-single ARGS="--mtp"
make dry-run
make serve-mtp ARGS="--dry-run --draft 2"
```

Build fetches only official source, checks out the full pinned commit and builds Release/Metal
with four workers. No upstream changes. No install step. Embedded UI builds/prebuilt UI and
OpenSSL are disabled; this is a local API runner. CMake/Ninja output is preserved in
`.runtime/build-*.log`. Existing runtime checkout must be clean and at the pin; otherwise the
build stops without resetting or deleting it. Git itself is used for the HTTPS source fetch.

For an explicit existing runner override, set `LLAMA_SERVER` to its executable path. `doctor`
and each launch check required help flags and the actual `MTL0` device. Records include the
runner's version, executable SHA256 and override status; an override is not claimed to be the
pinned build. Do not assume an older `--draft-max` flag works: the pin uses
`--spec-draft-n-max`. Ambient `LLAMA_ARG_*` variables are removed from child environments so they
cannot silently change flags or enable model downloads.

## Running and stopping

`make serve` runs until Ctrl-C, with full output in the displayed result directory's `server.log`.
`make serve-mtp` enables `draft-mtp`, default length 4; `serve-baseline` or
`make serve-mtp ARGS="--no-mtp"` explicitly disables it. Legacy `--draft 1` and `--draft 2` remain available.
The following single-model controls do not modify router INI presets.
The default is 262144 context at `127.0.0.1:39483` (outside the observed macOS ephemeral range
49152–65535). Override the port explicitly if occupied; no random fallback occurs.
Serving uses low effort, `--reasoning-budget 1024`, `--reasoning-format deepseek`, and a
client-overridable 4096 total output fallback. The budget applies per thinking block; use
`--reasoning-budget 0` to disable it or `-1` for unrestricted thinking. Smoke/bench default to
budget `0`, with explicit overrides available. Server sampling defaults are temperature 1,
top-p .95, top-k 20, min-p 0, presence penalty 0 and repeat penalty 1. Benchmark requests keep
their explicit deterministic sampling. Budget and projector for the legacy single-model commands are recorded in `run.json`; router
presets/rendered child arguments are recorded in `router.json` and smoke evidence.

There is no authentication. Loopback binding is **not authentication**: local clients can call
the API. `--cors-origins localhost` restricts browser response access to localhost origins, but
CORS is not an authorization boundary and does not protect against local clients or all requests.
Where a browser permits loopback access, a non-local webpage may still trigger management
writes such as model unloads or downloads without being allowed to read the response. Do not
treat localhost CORS as protection against those side effects. Do not reverse-proxy this listener
or expose it to the network casually. No user/global credentials are changed.

`make smoke` and `make bench*` start their own server; stop interactive serving first. A lock
prevents two recipe commands in this checkout. An existing process named `llama-server` is also
rejected, including one launched elsewhere. Other applications can still use the GPU: close them
manually. The port is checked before startup, and readiness verifies the reported context and
slot count. A failed launch stops, rather than changing context, KV precision, runner or model.
`--request-timeout` is a total HTTP request budget, including opening, headers and partial SSE
lines, not just an idle-read timeout. It uses POSIX `SIGALRM`/`ITIMER_REAL` on the CLI main thread;
embedded worker-thread calls or an already-active real-time timer fail before opening HTTP.
An existing active timer is left untouched; otherwise the previous signal handler is restored and
the recipe timer cleared on success, failure or interruption. No timeout worker threads are created.

Normal exit, request failure, Ctrl-C and SIGTERM terminate the owned process group, reap its
leader (SIGKILL escalation if needed), and verify the process group is gone. Native router
children inherit that group; smoke additionally verifies child exit after native unload. No process ownership can survive killing the Python owner
with SIGKILL or losing power; inspect process state before restarting after such an event.

## Projector and images

Legacy single-model serving automatically selects `mmproj-F16.gguf` beside the selected language-model file.
Router presets instead select the projector explicitly with `mmproj` (or `no-mmproj = true`).
`MMPROJ_PATH=/path/to/mmproj-F16.gguf` or `--mmproj-path` selects an existing projector explicitly,
including for smoke/bench. `--text-only` overrides both sources. An absent adjacent projector
prints a text-only notice; an explicit missing path fails. There is never an implicit download.

The locally validated public projector is from Unsloth Qwen3.8-27B-GGUF revision
`4ca720788d1e01f1bff70c033e0d0028fd02e502`, file `mmproj-F16.gguf`, **927607488 bytes**, SHA256:
`cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e`.

https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502/mmproj-F16.gguf

Download only this file manually if needed; do not overwrite an existing file. Verify size and
SHA256 before selecting it. It was checked before use in the local validation. Explicit alternate
projectors are not checksum-gated by the recipe and need their own compatibility verification.
The runner reported vision enabled and answered a real 64×64 PNG request correctly at MTP4/256K.
Large images and grounding accuracy were not tested. The native grounding warning recommends
at least 1024 image tokens for grounding tasks; the tiny color check is not such a task.

## Exact model/MTP verification

Every **legacy single-model** launch checks that the selected file exists and its metadata names `Qwen3.8-27B` with
architecture `qwen35`. MTP additionally requires
`qwen35.nextn_predict_layers=1` and real last-block projection, normalization, attention and FFN
tensor names. This is a bounded GGUF v3 directory reader, not a weight loader or tensor-integrity
validator. `run.json` records the selected filename, byte size and actual metadata under
`model_check`, not presumed provenance from a particular quantization. For optional integrity
verification, manually compare `shasum -a 256 /path/to/model.gguf` with the publisher's checksum
for that exact artifact; this full-file read is not part of doctor or serving.

The artifact is Qwen3.8 even though the GGUF architecture string is `qwen35`. Never substitute
Qwen3.6 because of a tutorial example. General Unsloth MTP documentation mixes Qwen3.6 examples
with newer references; its Qwen3.8 guide advertises MTP. Exact local tensor checks and real runner
results govern this recipe, not a generic guide's claim. Unsupported metadata/flags fail before
inference with no fallback and no extra draft-model/vision download.

## Validation

`make check` is offline: launch/config validation, GGUF directory parsing and MTP rejection,
readiness deadline/context checks, process-group cleanup, streamed framing/tool assembly,
benchmark arithmetic and counterbalanced scheduling. Mocks prove client plumbing only.
`make smoke-single` is the legacy real hardware text/forced-tool/result check. It validates a constrained
structured exchange, not broad natural tool-selection quality. Logs can contain local paths,
prompt text and model output; keep them private and inspect/redact before sharing.

`make check` is the documented offline gate; it is not run by the privacy hooks below.
Build and hardware smoke are distinct integration checks, not prerequisites for the offline
test runner.

## Repository-local privacy hooks

Run `make install-hooks` once per clone. It selects the tracked `.githooks/` directory using
only repository-local `core.hooksPath`; it never installs global configuration or packages.
It refuses conflicting local/global hook routing or existing default hooks rather than
silently bypassing them. Resolve integration deliberately if it refuses; do not remove global
protection to make this installer succeed. Re-running with this project's routing is safe.

Install gitleaks separately (8.30.1 tested). The POSIX-sh `pre-commit` hook scans staged changes;
`commit-msg` scans the proposed message. Both redact scanner findings. If gitleaks is missing,
they print a notice and allow the commit: **protection is inactive**, not fail-closed.
No Python pre-commit framework is used. The staged hook falls back to the older `protect`
command when the modern command/flags are unsupported; older versions are not fully tested.

`.gitleaks.toml` extends upstream default secret rules with generic home-directory path,
UNC-share path, and selected consumer-email patterns. GitHub noreply identities are allowlisted.
No personal literals are embedded. These patterns cannot guarantee detection of all secrets or
PII (including arbitrary names, hostnames, email domains, or every path spelling); review diffs
and messages yourself. False positives are possible. Prefer rewording; use scanner allowlists
or inline `gitleaks:allow` only after reviewing the exact finding. Hooks are local opt-in checks,
not a server-side policy, and do not scan all history or replace `make check`.

To uninstall routing, first confirm `git config --local --get core.hooksPath` prints exactly
`.githooks`, then run `git config --local --unset core.hooksPath`. Do not unset another hook
chain. This affects this checkout only and leaves tracked hook/config files available to others;
any inherited hook routing becomes effective again. No files under the home directory need
changing.
