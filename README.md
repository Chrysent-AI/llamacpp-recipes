# Local llama.cpp Metal router

A local-only native multi-model router targeting Apple M2 Pro / 32 GiB. Existing weights stay
external. The helper never downloads weights, edits Pi configuration, installs global packages,
or patches upstream. Model-specific settings live in a native INI file, not a JSON registry.

## Measured on M2 Pro

Measured October 20, 2026 on **Apple M2 Pro, 32 GiB unified memory**, macOS 27.0.1,
using **llama.cpp with Metal; these are not MLX measurements**. The unmodified runner is
build 11768, commit `1ba568c5ad5c6c718984bd2bf4e7c71485b706f9`. All cases configured
262,144 tokens, Q8 K/V, all layers on Metal, flash attention and one slot, with no automatic
context shrinking. Other desktop/GPU activity was not instrumented: these are potentially
contended desktop measurements, not an isolated-lab ceiling.

Selected configurations below show **decode tokens/second**, not end-to-end response speed.
Short inputs were 48–52 tokens; longer inputs were synthetic ledger prefixes near 8K and 20K.
Every response generated 128 tokens. Each new-model cell averages four uncached/warm responses
across two repetitions; the dense control averages only two. An em dash means not measured.

| Model | Unsloth quant | Mode | Short | ~8K | ~20K |
|---|---|---|---:|---:|---:|
| Qwen3.6 35B-A3B | UD-Q2_K_XL | MTP4 | 50.5 | — | — |
| Qwen3.6 35B-A3B | UD-Q3_K_XL | MTP4 | 58.3 | 33.3 | 22.7 |
| Qwen3.6 35B-A3B | UD-IQ4_XS | MTP4 | 57.1 | — | — |
| Gemma4 26B-A4B QAT | UD-Q4_K_XL | MTP2 | 53.9 | 33.9 | 25.1 |
| Nemotron3.5 Lightning 30B-A3B | UD-IQ2_M | Off | 39.7 | 36.0 | 32.4 |
| Nemotron3.5 Lightning 30B-A3B | UD-Q3_K_XL | MTP2 | 45.8 | — | — |
| Nemotron3.5 Lightning 30B-A3B | UD-Q4_K_M | Off | 35.8 | — | — |
| Qwen3.8 27B (dense control) | UD-IQ3_XXS | MTP4 | 14.5 | — | — |

MoE models activate a subset of their expert weights per token, unlike dense models; total
parameter count alone does not predict decode cost. That helps explain the appeal of these
MoE options, but this comparison does not isolate architecture from quantization or model quality.
MTP2/4 means speculative multi-token prediction with a maximum draft length of 2/4. Its benefit
is context-dependent: Nemotron IQ2 reached **47.6 tok/s with MTP4 short**, but only 23.4 at 20K,
versus 32.4 with MTP off. Nemotron Q4's MTP2 and MTP4 runs failed with Metal out-of-memory errors.

For the table's Qwen Q3, Gemma and Nemotron IQ2 configurations, uncached 20K time to first token
was roughly **54 / 89 / 49 seconds**, excluding model loading. Immediate repeated-prefix requests
were about 0.124 / 0.057 / 0.092 seconds because they reused the exact prefix cache, not because
fresh sessions are that fast. Maximum occupied input plus output was **20,624 tokens**:
262,144 is configured capacity, not demonstrated full-context performance or a throughput promise.

All eight isolated Pi BFS coding trials passed independent checks, but each used only read/write/edit
and one small task—not Bash, extensions or a full interactive harness with a 20K project prompt.
The report's trial times exclude model autoload and independent tests; they are not a coding-quality
ranking. Streamed thinking and constrained tool exchanges passed; Qwen/Gemma passed a minimal image
check, while Nemotron is text-only here. Reported RSS is not exact Metal allocation or whole-system use.
See the [full comparison and reproduction steps](docs/benchmarks/moe-256k.md) and
[aggregate data and exact model assets](docs/benchmarks/moe-256k-data.json) for all modes and limits.

## Quickstart

```sh
make build
# First configure ignored local/models.ini and model symlinks: see docs/setup.md.
make doctor                        # native INI validation, list unloaded models, stop
make dry-run                       # inspect router-only flags
make serve                         # private INI, loopback 39483, one resident model
# Stop with Ctrl-C before running a smoke or legacy benchmark.
make smoke                         # tiny autoload request, 256K props, native unload
```

`make serve` reads **`local/models.ini`** by default, with no model-path environment variable.
Use `make serve ARGS="--config local/other.ini"` for an explicit alternate configuration.
[`example.ini`](example.ini) contains **nine presets** with generic repository-relative paths:
the Qwen3.8 control, three Qwen3.6 quants, Gemma 4 QAT, and four Nemotron quant/mode combinations
(including IQ2 with MTP off and MTP4). Their IDs match
[`examples/pi-models-moe.json`](examples/pi-models-moe.json).
Keep the shared defaults and copy only model sections whose weights and required sidecars are
available; adjust paths in your private copy. The example does not install models. Merge only the
needed Pi entries into your own configuration, preserving unrelated entries; nothing here edits it
automatically. See [setup](docs/setup.md#native-router-configuration) before starting.

The router uses `--models-max 1 --models-autoload`. It loads the requested model on demand,
evicts an idle least-recently-used model when needed, and queues requests when the current
model is busy. **Unloading loses its KV cache**; returning to that model requires loading and
prompt evaluation again. This is native llama.cpp behavior, not a memory manager in Python.

The Qwen3.8 control preset keeps **262144 context**, MTP draft length **4**, matching projector,
Q8 K/V, all layers on Metal (`MTL0`), flash attention, one slot, two context checkpoints,
no RAM prompt-cache pool, no context shifting and no automatic fit/shrink. Low reasoning effort,
1024-token per-block budget and 4096 total output fallback are request-overridable defaults.
Other families have their own sampling and capabilities; do not copy Qwen's thinking knobs
blindly. Put model-specific flags in INI: native router CLI flags override all presets.

One Pi provider/port can offer several model IDs corresponding to the INI section names.
See [Pi connection](docs/pi.md) for the current control example and per-model configuration.
No user/global settings or credentials are changed by this repository.

## Validation and limits

The native router has passed unloaded listing → short Qwen3.8 request → 262144-context/vision
properties → native unload with child-exit verification on the pinned runner. MTP proposals
were observed. This proves integration, **not full 256K occupied performance, broad vision
quality, a switching benchmark, or an MTP speed advantage**. See [router validation](docs/validation.md).
The separate [seven-quant comparison](docs/benchmarks/moe-256k.md) includes measured off/MTP decode,
8K/20K prefixes, image/tool checks and actual isolated Pi BFS trials, with explicit failures and limits.

Legacy Qwen-only controls remain explicit, with the prior `MODEL_PATH` / mounted-volume
discovery and optional projector controls. They do not use the router INI:

```sh
make serve-mtp                     # legacy single Qwen, MTP4, 256K
make serve-baseline                # legacy single Qwen, MTP off
make doctor-single ARGS="--mtp"
make smoke-single ARGS="--mtp"     # legacy text + streamed tool/result check
make bench-mtp ARGS="--repeats 2 --drafts 1 2 4 --tokens 128"
make check                         # offline stdlib gate; no weights loaded
```

Legacy smoke/bench use explicit deterministic sampling and reasoning budget 0 by default;
see [measurement/tuning](docs/tuning.md). Every model-running command owns a process group
and repository lock, and refuses an already-running `llama-server`. Logs remain in ignored
`results/`; builds and local symlinks in ignored `.runtime/`. Nothing is committed or published.

## Contributing and license

Run `make install-hooks` once per clone to opt in to repository-local gitleaks checks of
staged content and commit messages. Install gitleaks separately (8.30.1 tested); without it,
the hooks warn and **pass without scanning**. No global Git configuration or Python pre-commit
framework is needed. See [hook setup and limits](docs/setup.md#repository-local-privacy-hooks)
for uninstall instructions and detection limits. Run `make check` separately before committing.

This recipe is MIT licensed; see [LICENSE](LICENSE). Upstream llama.cpp and model/projector
weights retain their own licenses; this license does not relicense them.
