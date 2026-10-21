# Connect Pi to the local endpoint

Run `make serve` and leave that terminal open. It starts the **native llama.cpp router** from
ignored `local/models.ini`, at the same provider URL/port as the earlier single-model control.
Nothing in this repository automatically edits Pi settings or credentials. Requests select
models by INI section ID; do not run a second loader/server alongside this owned router.

One provider can contain multiple entries in its `models` array, each with its own ID, context,
output allowance, verified input capabilities and sampling parameters. Add an entry only after
its local preset/files and capabilities are validated. Keep the provider name and base URL;
do not duplicate the provider for each quantization. [`example.ini`](../example.ini) shows native
preset names and candidate-specific server sampling, while the JSON below deliberately remains
the verified Qwen3.8 control only. Do not copy Qwen's thinking map or image capability onto other
families without checking them. Native single-resident switching unloads the previous idle model
and loses its KV cache; selecting it again requires reload and prompt evaluation.

A ready-to-copy control configuration is available at [`examples/pi-models.json`](../examples/pi-models.json).
The separately tested MoE candidate fragment is [`examples/pi-models-moe.json`](../examples/pi-models-moe.json);
read the [comparison report](benchmarks/moe-256k.md) before selecting quants/MTP modes. It retains this
control and uses verified binary thinking-template switches for the new families, not this control's
effort mapping. Both examples contain only
public local-provider settings, not anyone's personal Pi configuration.
For a new configuration, use it as a starting point; if `models.json` already exists, merge only
its `llamacpp-qwen38` provider into the existing `providers` object—do not overwrite the file.
Pi's default configuration path is `~/.pi/agent/models.json`.

Adjust `baseUrl` to your server's address and port, `models[0].id` to its served alias, and
`contextWindow` to the configured server capacity. `maxTokens` is the total output allowance;
`reasoning_budget_tokens` is the separate per-thinking-block budget. Remove `"image"` from
`input` when serving without a matching projector. These settings are for Qwen3.8-27B; other
Unsloth model families may need different templates, sampling and thinking controls.

Merge the following provider into the existing agent-directory `models.json`, preserving all
other providers. Open `/model` to reload and select `llamacpp-qwen38/qwen3.8-27b`, then select
`/thinking low`. Saving a default model or thinking level is a separate user choice.

```json
{
  "providers": {
    "llamacpp-qwen38": {
      "baseUrl": "http://127.0.0.1:39483/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [{
        "id": "qwen3.8-27b",
        "name": "Qwen3.8 27B local Metal",
        "reasoning": true,
        "input": ["text", "image"],
        "contextWindow": 262144,
        "maxTokens": 4096,
        "thinkingLevelMap": {
          "off": "none", "minimal": "low", "low": "low",
          "medium": "medium", "high": "xhigh", "xhigh": "xhigh"
        },
        "compat": {
          "supportsReasoningEffort": true,
          "maxTokensField": "max_tokens"
        },
        "samplingParams": {
          "temperature": 1.0, "top_p": 0.95, "top_k": 20,
          "min_p": 0.0, "presence_penalty": 0.0, "repeat_penalty": 1.0,
          "reasoning_budget_tokens": 1024,
          "chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}
        },
        "samplingParamsByThinkingLevel": {
          "off": {
            "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
            "chat_template_kwargs": {"enable_thinking": false, "preserve_thinking": true}
          }
        }
      }]
    }
  }
}
```

The dummy key is non-secret; the loopback server does not authenticate clients. The alias matches
the server. Image input is verified with the matching projector, not with text-only serving: if
you remove `mmproj` and set `no-mmproj = true` in its preset, remove `"image"` from that model's input list.
See [setup](setup.md) for projector selection and provenance.

## Reasoning and output limits

The embedded local GGUF template accepts low/medium/xhigh effort (and aliases high to xhigh).
Effort is an instruction, **not a numeric budget**. Pi minimal/low map to low and high/xhigh to
xhigh; these are not six distinct native capabilities. Server low effort is a fallback and the
request's `reasoning_effort` overrides it. No forced server `enable_thinking` kwarg contradicts
Pi's off setting. The off-level kwargs above explicitly replace the thinking-enabled kwargs:
Pi merges sampling shallowly, so omit neither the false flag nor the preserved-history choice.

The 1024 budget is approximate and applies per thinking block, not an immutable aggregate
quota. The 4096 total output allowance includes thinking plus answer; it does not guarantee
an answer if exhausted. Server `--n-predict 4096` is a client-overridable fallback, not a hard
limit against arbitrary clients. Do not put competing output-limit fields in `samplingParams`.
`--reasoning-format deepseek` streams thoughts separately as `reasoning_content`.

Sampling follows the distributor's thinking and nonthinking recommendations, with its
`repetition_penalty` translated to the runner's `repeat_penalty`:
https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/README.md

This uses installed Pi 1.0.3 `docs/models.md`, `docs/configuration.md` and the installed
OpenAI-completions/schema implementation; no provider extension or model-gating manifest is
needed. A real direct API check passed streamed thinking + answer, constrained tool/result,
thinking-off, and a synthetic image at IQ3/MTP4/256K. This is not a live Pi UI session or broad
vision/tool-selection quality test. See [validation evidence](validation.md).

No cache lifetime or Pi cache-warming eligibility is declared. Live-prefix reuse was observed
on a short tool continuation; this does not promise full-history or 20K reuse. Start with a
non-sensitive conversation: model tool calls are untrusted executable instructions. Review
Pi permissions and project trust before letting this local model operate on important files.
