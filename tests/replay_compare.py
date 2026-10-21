"""Read-only replay of retained native benchmark evidence; never starts a model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare
import recipe


def replay(folder):
    count = 0
    for path in sorted(folder.glob("*/performance.jsonl")):
        run = path.parent
        result = json.loads((run / "result.json").read_text())
        props = json.loads((run / "props.json").read_text())
        loaded = json.loads((run / "loaded.json").read_text())
        mtp = compare.validate_native(loaded, result["model"], props)
        for line in path.read_text().splitlines():
            stored = json.loads(line)
            name = f'{stored["target_input"]}-{stored["warm"]}'
            events = []
            done = False
            for raw in (run / (name + ".sse")).read_bytes().splitlines():
                if not raw.startswith(b"data: "):
                    continue
                if raw[6:] == b"[DONE]":
                    done = True
                    continue
                event = json.loads(raw[6:])
                events.append(event)
                done |= event.get("stop") is True
            if not done:
                raise AssertionError("Stored response is incomplete")
            measured = recipe.metrics(events, stored["ttft_seconds"], stored["request_seconds"])
            for key, value in measured.items():
                if stored[key] != value:
                    raise AssertionError(f"Metric mismatch: {path} {key}")
            request = json.loads((run / (name + ".request.json")).read_text())
            if len(request["prompt"]) != stored["tokenized_input"]:
                raise AssertionError("Tokenizer input count mismatch")
            if stored["timings"]["cache_n"] + stored["timings"]["prompt_n"] != stored["tokenized_input"]:
                raise AssertionError("Native occupied input differs from tokenizer count")
            if request["cache_prompt"] != stored["warm"]:
                raise AssertionError("Cache treatment mismatch")
            if not stored["warm"] and stored["timings"]["cache_n"]:
                raise AssertionError("Cold request unexpectedly reused prefix")
            if mtp and not (isinstance(measured["draft_tokens"], int) and measured["draft_tokens"] > 0):
                raise AssertionError("MTP flag without actual draft work")
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps({str(p): replay(p) for p in args.folders}, indent=2))
