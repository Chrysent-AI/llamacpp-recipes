"""Bounded native-router experiments; raw evidence stays in ignored results/."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import zlib

import recipe as r

FIXTURE = r.ROOT / "examples/compare-prompts.json"


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class ChildrenMonitor(r.Monitor):
    """Sample the owned process group, including native model children."""
    def __init__(self, pid):
        super().__init__(pid)
        self.progress = time.monotonic()
        self.aborted = None
        self.progress_file = None
        self.progress_bytes = 0

    def run(self):
        while not self.stop.is_set():
            if self.progress_file and self.progress_file.exists():
                size = self.progress_file.stat().st_size
                if size != self.progress_bytes:
                    self.progress_bytes = size
                    self.progress = time.monotonic()
            sample = r.memory(self.pid)
            try:
                rows = r.capture(["ps", "-axo", "pid=,pgid=,rss="]).splitlines()
                members = [list(map(int, row.split())) for row in rows]
                group = [row for row in members if row[1] == self.pid]
                sample["rss_bytes"] = sum(row[2] * 1024 for row in group)
                sample["pids"] = [row[0] for row in group]
            except (OSError, ValueError, subprocess.SubprocessError):
                sample["rss_bytes"] = None
            sample["elapsed"] = time.monotonic()
            self.samples.append(sample)
            initial = self.samples[0]["system_swap_bytes"]
            current = sample["system_swap_bytes"]
            if (initial is not None and current is not None and current - initial > 2 * 1024**3
                    and time.monotonic() - self.progress > 60):
                self.aborted = "swap grew >2 GiB and no streamed progress for 60 seconds"
                try:
                    os.killpg(self.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return
            self.stop.wait(1)


def recorded_stream(args, path, endpoint, body, monitor):
    """Preserve every SSE line, including an incomplete response on failure."""
    started, first, events, done = time.monotonic(), None, [], False
    monitor.progress = started
    save(path.with_suffix(".request.json"), body)
    with path.open("wb") as raw, r.request_deadline(args.request_timeout, started):
        with r.api(args.port, endpoint, body, timeout=args.request_timeout) as response:
            for line in response:
                raw.write(line)
                raw.flush()
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                events.append(event)
                if "error" in event:
                    raise ValueError(str(event["error"]))
                delta = (event.get("choices") or [{}])[0].get("delta", {})
                if event.get("content") or any(delta.get(k) for k in ("content", "tool_calls", "reasoning_content")):
                    monitor.progress = time.monotonic()
                    if first is None:
                        first = monitor.progress - started
                if event.get("stop") is True:
                    done = True
    if not done:
        raise ValueError("Incomplete SSE response; inspect preserved raw stream")
    return events, first, time.monotonic() - started


def chat(args, folder, name, model, monitor, messages, **extra):
    return recorded_stream(args, folder / (name + ".sse"), "/v1/chat/completions", {
        "model": model, "messages": messages, "stream": True, "temperature": 0,
        "seed": 42, "max_tokens": 256, "reasoning_budget_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False}, **extra}, monitor)


def red_blue_png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    row = b"\0" + b"\xff\0\0" * 64 + b"\0\0\xff" * 64
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 128, 64, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * 64)) + chunk(b"IEND", b""))


def capabilities(args, folder, model, monitor, props):
    result = {}
    events, _, _ = chat(args, folder, "thinking", model, monitor,
                        [{"role": "user", "content": "What is 17 times 19? Briefly check and answer."}],
                        chat_template_kwargs={"enable_thinking": True})
    thought = "".join(c.get("delta", {}).get("reasoning_content", "") or ""
                      for e in events for c in e.get("choices", []))
    result["thinking_answer"] = {"reasoning_chars": len(thought), "answer": r.text_from(events),
                                 "pass": bool(thought) and "323" in r.text_from(events)}
    tool = {"type": "function", "function": {"name": "echo", "description": "Echo value",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                           "required": ["value"], "additionalProperties": False}}}
    messages = [{"role": "user", "content": "Call echo with value metal-ready, then report the result."}]
    events, _, _ = chat(args, folder, "tool", model, monitor, messages, tools=[tool], tool_choice="required")
    calls = {}
    for event in events:
        for choice in event.get("choices", []):
            for part in choice.get("delta", {}).get("tool_calls", []):
                call = calls.setdefault(part["index"], {"id": "", "type": "function",
                                                        "function": {"name": "", "arguments": ""}})
                if part.get("id"):
                    call["id"] = part["id"]
                for key in ("name", "arguments"):
                    call["function"][key] += part.get("function", {}).get(key, "") or ""
    valid = len(calls) == 1
    call = next(iter(calls.values()), {})
    try:
        valid = valid and bool(call["id"]) and call["function"]["name"] == "echo" and json.loads(
            call["function"]["arguments"]) == {"value": "metal-ready"}
    except (KeyError, ValueError):
        valid = False
    result["tool_call"] = {"pass": valid, "calls": list(calls.values())}
    if valid:
        messages += [{"role": "assistant", "content": None, "tool_calls": [call]},
                     {"role": "tool", "tool_call_id": call["id"], "content": "metal-ready"}]
        events, _, _ = chat(args, folder, "tool-result", model, monitor, messages,
                            tools=[tool], tool_choice="none")
        result["tool_result"] = {"pass": "metal-ready" in r.text_from(events), "answer": r.text_from(events)}
    if props.get("modalities", {}).get("vision"):
        image = base64.b64encode(red_blue_png()).decode()
        events, _, _ = chat(args, folder, "image", model, monitor, [{"role": "user", "content": [
            {"type": "text", "text": "Name the left and right colors in this image, in that order."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}]}])
        answer = r.text_from(events).lower()
        result["image"] = {"pass": "red" in answer and "blue" in answer and answer.index("red") < answer.index("blue"),
                           "answer": answer}
    else:
        result["image"] = {"pass": False, "status": "not advertised by native props"}
    save(folder / "capabilities.json", result)
    return result


def prompt_tokens(args, model, text):
    rendered = r.json_request(args, "/apply-template", {"model": model,
        "messages": [{"role": "user", "content": text}],
        "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    return r.json_request(args, "/tokenize", {"model": model, "content": rendered,
                                            "add_special": True})["tokens"]


def validate_native(loaded, model, props):
    residents = [m for m in loaded["data"] if m["status"]["value"] in ("loaded", "loading", "sleeping")]
    if len(residents) != 1 or residents[0]["id"] != model:
        raise ValueError("Expected exactly the requested resident model")
    argv = residents[0]["status"]["args"]
    expected = {"--ctx-size": "262144", "--parallel": "1", "--n-gpu-layers": "all",
                "--device": "MTL0", "--cache-type-k": "q8_0", "--cache-type-v": "q8_0",
                "--fit": "off", "--flash-attn": "on"}
    if any(flag not in argv or argv[argv.index(flag) + 1] != value for flag, value in expected.items()):
        raise ValueError("Native child arguments differ from fixed capacity/precision/Metal contract")
    if ("--no-context-shift" not in argv or props["default_generation_settings"]["n_ctx"] != 262144
            or props["total_slots"] != 1):
        raise ValueError("Native context/slot contract differs; no fallback")
    return "--spec-type" in argv and argv[argv.index("--spec-type") + 1] == "draft-mtp"


def performance(args, folder, model, monitor, mtp):
    fixture = json.loads(FIXTURE.read_text())
    for size in args.sizes:
        text = fixture["short"]
        tokens = prompt_tokens(args, model, text)
        if size:
            low, high = 1, size
            while low < high:
                rows = (low + high) // 2
                text = fixture["context_header"] + "".join(fixture["context_row"].format(i=i)
                    for i in range(rows)) + fixture["context_question"]
                tokens = prompt_tokens(args, model, text)
                if len(tokens) < size:
                    low = rows + 1
                else:
                    high = rows
            text = fixture["context_header"] + "".join(fixture["context_row"].format(i=i)
                for i in range(low)) + fixture["context_question"]
            tokens = prompt_tokens(args, model, text)
        (folder / f"prompt-{size}.txt").write_text(text)
        for warm in (False, True):
            events, first, elapsed = recorded_stream(args, folder / f"{size}-{warm}.sse", "/completion", {
                "model": model, "prompt": tokens, "n_predict": args.tokens, "stream": True,
                "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0,
                "repeat_penalty": 1, "presence_penalty": 0, "seed": 42,
                "cache_prompt": warm}, monitor)
            record = {"target_input": size, "tokenized_input": len(tokens), "warm": warm,
                      **r.metrics(events, first, elapsed)}
            if mtp and not (isinstance(record["draft_tokens"], int) and record["draft_tokens"] > 0):
                raise ValueError("Configured MTP did not produce live draft counters")
            with (folder / "performance.jsonl").open("a") as output:
                output.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


def pi_trial(args, folder, model, monitor):
    config = json.loads(Path(args.pi_config).read_text())
    provider = config["providers"]["llamacpp-qwen38"]
    selected = next(m for m in provider["models"] if m["id"] == model)
    config = {"providers": {"llamacpp-qwen38": {**provider, "models": [selected]}}}
    with tempfile.TemporaryDirectory(prefix="local-bfs-") as temporary:
        root = Path(temporary)
        agent, work = root / "agent", root / "work"
        agent.mkdir()
        work.mkdir()
        save(agent / "models.json", config)
        # No inherited provider secrets, OPI hooks, or user configuration discovery.
        node = shutil.which("node")
        if node is None:
            raise ValueError("Pi trials require the existing working Node executable")
        env = {"PATH": str(Path(node).parent) + ":" + os.defpath + ":/opt/homebrew/bin", "HOME": str(root),
               "PI_CODING_AGENT_DIR": str(agent), "PI_OFFLINE": "1", "PI_TELEMETRY": "0"}
        command = [shutil.which("pi"), "--print", "--mode", "json", "--no-session", "--offline",
                   "--no-extensions", "--no-skills", "--no-context-files", "--no-prompt-templates",
                   "--no-themes", "--no-approve", "--tools", "read,write,edit", "--model",
                   "llamacpp-qwen38/" + model, "--thinking", "low", "--", json.loads(FIXTURE.read_text())["coding"]]
        started, expired = time.monotonic(), False
        monitor.progress_file = folder / "pi.jsonl"
        with (folder / "pi.jsonl").open("wb") as output:
            proc = subprocess.Popen(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                expired = True
            finally:
                r.terminate(proc)
        result = {"seconds": time.monotonic() - started, "exit": proc.returncode, "timeout": expired,
                  "file_present": (work / "BFS.py").is_file(), "file_bytes": 0,
                  "correctness": "pending independent review"}
        if result["file_present"]:
            shutil.copyfile(work / "BFS.py", folder / "BFS.py")
            result["file_bytes"] = (folder / "BFS.py").stat().st_size
        events = []
        for line in (folder / "pi.jsonl").read_text().split("\n"):
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        result["tool_calls"] = [e.get("toolName") for e in events if e.get("type") == "tool_execution_start"]
        result["usage"] = [e["message"].get("usage") for e in events if e.get("type") == "message_end"
                           and e.get("message", {}).get("role") == "assistant"]
        save(folder / "pi-summary.json", result)
        return result


def case(args, folder, model):
    result = {"model": model, "phase": args.phase, "status": "failed"}
    monitor = None
    try:
        with r.router(args, folder) as (_, proc):
            monitor = ChildrenMonitor(proc.pid)
            monitor.thread.start()
            try:
                events, first, elapsed = chat(args, folder, "ready", model, monitor,
                    [{"role": "user", "content": "Reply with the word ready."}], max_tokens=16,
                    reasoning_budget_tokens=0)
                result["autoload_request_seconds"] = elapsed
                if not r.text_from(events).strip():
                    raise ValueError("No text in readiness smoke")
                props = r.json_request(args, "/props?" + urllib.parse.urlencode({"model": model}))
                save(folder / "props.json", props)
                loaded = r.json_request(args, "/models")
                save(folder / "loaded.json", loaded)
                mtp = validate_native(loaded, model, props)
                if args.phase == "capabilities":
                    result["capabilities"] = capabilities(args, folder, model, monitor, props)
                elif args.phase == "performance":
                    performance(args, folder, model, monitor, mtp)
                else:
                    result["pi"] = pi_trial(args, folder, model, monitor)
                if not r.json_request(args, "/models/unload", {"model": model}).get("success"):
                    raise ValueError("Native unload failed")
                result["status"] = "completed"
            finally:
                monitor.stop.set()
                monitor.thread.join(timeout=65)
        # owned() verified the process group is gone; independently check the endpoint is free.
        r.ensure_available(args)
        result["cleanup"] = "owned group gone and port available"
        if args.phase == "pi":
            trial = result["pi"]
            if trial["timeout"] or trial["exit"] != 0 or not trial["file_present"] or trial["file_bytes"] == 0:
                result["status"] = "failed"
                result["error"] = "Pi execution failed or produced no nonempty BFS.py; inspect pi-summary.json"
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        if isinstance(exc, urllib.error.HTTPError):
            (folder / "http-error.txt").write_bytes(exc.read())
    finally:
        if monitor:
            result["memory"] = monitor.report()
            result["pressure_abort"] = monitor.aborted
            save(folder / "memory.json", monitor.samples)
        save(folder / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["capabilities", "performance", "pi"])
    parser.add_argument("--config", default="local/compare.ini")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="+", default=[0])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pi-config", default="local/compare-pi.json")
    parser.add_argument("--budget", type=float, default=7200)
    parser.add_argument("--request-timeout", type=float, default=600)
    args = parser.parse_args()
    args.port, args.startup_timeout = 39483, 60
    if (not re.fullmatch(r"[a-zA-Z0-9_-]+", args.output) or args.repeats < 1
            or args.tokens < 1 or args.tokens > 4096 or any(n < 0 or n > 20480 for n in args.sizes)
            or not 0 < args.budget <= 7200 or not 0 < args.request_timeout <= 1200
            or any(not re.fullmatch(r"[a-zA-Z0-9_.-]+", m) for m in args.models)):
        parser.error("Invalid bounded experiment arguments")
    folder = r.ROOT / "results" / args.output
    folder.mkdir()
    save(folder / "plan.json", vars(args))
    started, failed = time.monotonic(), set()
    with r.exclusive():
        for repetition in range(args.repeats):
            models = args.models if repetition % 2 == 0 else list(reversed(args.models))
            for model in models:
                if model in failed:
                    continue
                if time.monotonic() - started >= args.budget:
                    save(folder / "budget-stop.json", {"seconds": time.monotonic() - started,
                                                       "next_model": model, "repetition": repetition})
                    return
                run = folder / f"{repetition}-{model}"
                run.mkdir()
                if case(args, run, model)["status"] == "failed":
                    failed.add(model)
                    r.ensure_available(args)
                save(folder / "failed-models.json", sorted(failed))


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, r.interrupted)
    main()
