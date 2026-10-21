"""Local, single-owner llama.cpp Metal recipe. Python standard library only."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from gguf_meta import inspect, require_mtp

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/ggml-org/llama.cpp.git"
SOURCE_COMMIT = "1ba568c5ad5c6c718984bd2bf4e7c71485b706f9"
ALIAS = "qwen3.8-27b"
DEFAULT_CONTEXT = 262144
DEFAULT_PORT = 39483
PREFERRED_MODEL = "Qwen3.8-27B-UD-IQ3_XXS.gguf"
RUNTIME = ROOT / ".runtime"
SOURCE = RUNTIME / "llama.cpp"
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path():
    supplied = os.environ.get("MODEL_PATH")
    if supplied:
        path = Path(supplied).expanduser()
    else:
        matches = [p for p in Path("/Volumes").glob("*/models/unsloth/**/Qwen3.8-27B*.gguf")
                   if p.is_file()]
        preferred = [p for p in matches if p.name == PREFERRED_MODEL]
        if preferred:
            matches = preferred
        if len(matches) != 1:
            raise ValueError("Set MODEL_PATH to an existing Qwen3.8-27B GGUF; discovery requires exactly one candidate (no download is automatic)")
        path = matches[0]
    if not path.is_file():
        raise ValueError("MODEL_PATH must name an existing GGUF file")
    return path.resolve()


def projector_path(args, model):
    if args.text_only:
        return None
    if args.mmproj_path:
        path = Path(args.mmproj_path).expanduser()
        if not path.is_file():
            raise ValueError("MMPROJ_PATH must name an existing projector GGUF")
        return path.resolve()
    if args.action.startswith("serve"):
        adjacent = model.parent / "mmproj-F16.gguf"
        if adjacent.is_file():
            return adjacent.resolve()
    return None


def selected_draft(args):
    return args.draft if args.mtp else 0


def model_info(path, mtp=False):
    metadata, tensors = inspect(path)
    if metadata.get("general.name") != "Qwen3.8-27B" or metadata.get("general.architecture") != "qwen35":
        raise ValueError("Wrong model metadata: expected Qwen3.8-27B / qwen35")
    mtp_tensors = require_mtp(metadata, tensors) if mtp else sorted(n for n in tensors if ".nextn." in n)
    return {"filename": path.name, "bytes": path.stat().st_size,
            "metadata": metadata, "mtp_tensors": mtp_tensors}


def runner():
    return str(Path(os.environ.get("LLAMA_SERVER", str(RUNTIME / "build/bin/llama-server"))).expanduser().resolve())


def clean_env():
    # Prevent ambient llama.cpp flags from changing the recipe or enabling downloads.
    return {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}


def command(args, draft):
    cmd = [runner(), "--model", ".runtime/model.gguf", "--alias", ALIAS,
           "--host", "127.0.0.1", "--cors-origins", "localhost", "--port", str(args.port), "--ctx-size", str(args.context),
           "--parallel", "1", "--n-gpu-layers", "all", "--device", "MTL0",
           "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           "--ctx-checkpoints", "2", "--cache-ram", "0", "--fit", "off",
           "--no-context-shift", "--jinja", "--reasoning-budget", str(args.reasoning_budget),
           "--reasoning-format", "deepseek", "--reasoning-effort", "low",
           "--chat-template-kwargs", '{"preserve_thinking":true}',
           "--n-predict", "4096", "--temp", "1", "--top-p", "0.95", "--top-k", "20",
           "--min-p", "0", "--presence-penalty", "0", "--repeat-penalty", "1"]
    projector = getattr(args, "projector", None)
    cmd += ["--mmproj", str(projector)] if projector else ["--no-mmproj"]
    cmd += ["--spec-type", "draft-mtp" if draft else "none"]
    if draft:
        cmd += ["--spec-draft-n-max", str(draft), "--spec-draft-n-min", "0"]
    return cmd


def router_command(args):
    config = Path(args.config).expanduser()
    if not (ROOT / config).is_file():
        raise ValueError("Missing router INI; create local/models.ini from example.ini or pass --config")
    # Model flags belong in INI: native router CLI has higher precedence than presets.
    return [runner(), "--models-preset", str(config), "--models-max", "1",
            "--models-autoload", "--host", "127.0.0.1", "--port", str(args.port),
            "--cors-origins", "localhost"]


def capture(cmd):
    return subprocess.check_output(cmd, cwd=ROOT, env=clean_env(), stderr=subprocess.STDOUT,
                                   timeout=30, text=True)


def runner_info(args, draft, cmd=None):
    help_text = capture([runner(), "--help"])
    required = [x for x in (cmd or command(args, draft))[1:] if x.startswith("--")]
    if any(flag not in help_text for flag in required) or (draft and "draft-mtp" not in help_text):
        raise ValueError("Runner lacks required flags; make build at the pinned commit; no fallback")
    devices = capture([runner(), "--list-devices"])
    if "MTL0:" not in devices:
        raise ValueError("No MTL0 Metal device exposed by runner; inspect --list-devices")
    version = capture([runner(), "--version"])
    return {"version": version.strip(), "devices": devices.strip(), "binary_sha256": digest(runner()),
            "override": bool(os.environ.get("LLAMA_SERVER")), "expected_source": SOURCE_COMMIT}


def terminate(proc, grace=8):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # A reaped leader does not prove its children exited.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=grace)
    deadline = time.monotonic() + grace
    while True:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Owned process group still exists after cleanup")
        threading.Event().wait(min(0.05, max(0, deadline - time.monotonic())))


@contextlib.contextmanager
def owned(cmd, log):
    with open(log, "wb") as output:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=clean_env(), stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield proc
        finally:
            terminate(proc)


@contextlib.contextmanager
def exclusive():
    RUNTIME.mkdir(exist_ok=True)
    with open(RUNTIME / "owner.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another recipe command owns the runner; stop it first") from None
        yield


def build():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("This recipe builds Apple Silicon Metal only")
    commands = []
    if not SOURCE.exists():
        commands += [["git", "clone", "--filter=blob:none", "--no-checkout", SOURCE_URL, str(SOURCE)],
                     ["git", "-C", str(SOURCE), "checkout", "--detach", SOURCE_COMMIT]]
    else:
        if capture(["git", "-C", str(SOURCE), "rev-parse", "HEAD"]).strip() != SOURCE_COMMIT:
            raise ValueError("Existing runtime checkout is not pinned; leave it untouched and inspect")
        if capture(["git", "-C", str(SOURCE), "status", "--porcelain"]).strip():
            raise ValueError("Runtime source is dirty; refusing to build modified upstream")
    commands += [["cmake", "-S", str(SOURCE), "-B", str(RUNTIME / "build"), "-G", "Ninja",
                  "-DCMAKE_BUILD_TYPE=Release", "-DGGML_METAL=ON", "-DLLAMA_OPENSSL=OFF",
                  "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF",
                  "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"],
                 ["cmake", "--build", str(RUNTIME / "build"), "--target", "llama-server", "-j", "4"]]
    for index, cmd in enumerate(commands):
        log = RUNTIME / f"build-{time.time_ns()}-{index}.log"
        print("Build log:", log.relative_to(ROOT), flush=True)
        with owned(cmd, log) as proc:
            if proc.wait(timeout=900):
                raise ValueError("Build failed; inspect preserved log")


def api(port, endpoint, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{endpoint}", data=data,
                                 headers={"Content-Type": "application/json"})
    return HTTP.open(req, timeout=timeout)


def wait_ready(proc, args):
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ValueError("Server exited during startup; inspect server log (no context fallback)")
        try:
            with api(args.port, "/health", timeout=1) as response:
                if json.load(response).get("status") == "ok":
                    with api(args.port, "/props") as props:
                        settings = json.load(props)
                    actual = settings["default_generation_settings"]["n_ctx"]
                    if actual != args.context or settings["total_slots"] != 1:
                        raise ValueError("Server context/slot settings differ from requested recipe")
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            threading.Event().wait(min(0.2, max(0, deadline - time.monotonic())))
    raise TimeoutError("Server readiness deadline exceeded; inspect preserved log")


def memory(pid):
    rss, swap = None, None
    try:
        rss = int(capture(["ps", "-o", "rss=", "-p", str(pid)]).strip()) * 1024
        text = capture(["sysctl", "-n", "vm.swapusage"])
        match = re.search(r"used = ([\d.]+)M", text)
        swap = float(match[1]) * 1024 * 1024 if match else None
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return {"rss_bytes": rss, "system_swap_bytes": swap}


class Monitor:
    def __init__(self, pid):
        self.pid, self.samples, self.stop = pid, [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            self.samples.append(memory(self.pid))
            self.stop.wait(1)

    def report(self):
        result = {}
        for key in ("rss_bytes", "system_swap_bytes"):
            values = [s[key] for s in self.samples if s[key] is not None]
            result[key] = {"start": values[0], "end": values[-1], "delta": values[-1] - values[0],
                           "sampled_peak": max(values)} if values else "unknown"
        return result


def ensure_available(args):
    existing = subprocess.run(["pgrep", "-x", "llama-server"], stdout=subprocess.DEVNULL,
                              timeout=10)
    if existing.returncode != 1:
        raise ValueError("A llama-server already exists or process check failed; inspect before starting")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))


def json_request(args, endpoint, body=None):
    with request_deadline(args.request_timeout, time.monotonic()), \
         api(args.port, endpoint, body, timeout=args.request_timeout) as response:
        return json.load(response)


def wait_router(proc, args):
    started = time.monotonic()
    with request_deadline(args.startup_timeout, started):
        while time.monotonic() - started < args.startup_timeout:
            if proc.poll() is not None:
                raise ValueError("Router exited during startup; inspect server.log for INI errors")
            try:
                with api(args.port, "/models", timeout=1) as response:
                    models = json.load(response)
                if not models.get("data"):
                    raise ValueError("Router lists no models; check the INI sections and model paths")
                return models
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                threading.Event().wait(min(0.2, max(0, args.startup_timeout - (time.monotonic() - started))))
    raise TimeoutError("Router readiness deadline exceeded")


@contextlib.contextmanager
def router(args, folder):
    cmd = router_command(args)
    ensure_available(args)
    info = {"runner": runner_info(args, 0, cmd), "models_max": 1}
    with owned(cmd, folder / "server.log") as proc:
        info["models"] = wait_router(proc, args)
        (folder / "router.json").write_text(json.dumps(info, indent=2) + "\n")
        print("Router ready: http://127.0.0.1:" + str(args.port), flush=True)
        yield info, proc


def router_smoke(args, folder, initial, proc):
    model = args.model_id
    before = next((m for m in initial["data"] if m["id"] == model), None)
    if before is None or before["status"]["value"] != "unloaded":
        raise ValueError("Smoke requires the selected model to start unloaded")
    # Exercise request-triggered autoload, not a separate load manager or single-model server.
    events, _, _ = stream(args, "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "Reply with the word ready."}],
        "stream": True, "max_tokens": 16, "temperature": 0,
        "reasoning_budget_tokens": 0, "chat_template_kwargs": {"enable_thinking": False}})
    if not text_from(events).strip():
        raise ValueError("Router smoke produced no text")
    query = urllib.parse.urlencode({"model": model, "autoload": "false"})
    props = json_request(args, "/props?" + query)
    loaded = json_request(args, "/models")
    (folder / "smoke.json").write_text(json.dumps({"events": events, "props": props,
                                                   "loaded": loaded}, indent=2) + "\n")
    if props["default_generation_settings"]["n_ctx"] != DEFAULT_CONTEXT or props["total_slots"] != 1:
        raise ValueError("Router smoke expected 262144 context and one slot; no fallback")
    residents = [m for m in loaded["data"] if m["status"]["value"] in ("loading", "loaded", "sleeping")]
    if len(residents) != 1 or residents[0]["id"] != model:
        raise ValueError("Router resident model differs from selected model")
    # Record direct native children before unload; subprocesses inherit our owned group.
    children = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True, timeout=10)
    if children.returncode != 0:
        raise ValueError("Cannot identify loaded router child")
    pids = [int(pid) for pid in children.stdout.split()]
    if not json_request(args, "/models/unload", {"model": model}).get("success"):
        raise ValueError("Native unload failed")
    started = time.monotonic()
    with request_deadline(args.startup_timeout, started):
        while True:
            with api(args.port, "/models", timeout=1) as response:
                after = json.load(response)
            entry = next(m for m in after["data"] if m["id"] == model)
            alive = False
            for pid in pids:
                try:
                    os.kill(pid, 0)
                    alive = True
                except ProcessLookupError:
                    pass
            if entry["status"]["value"] == "unloaded" and not alive:
                (folder / "unloaded.json").write_text(json.dumps(after, indent=2) + "\n")
                print("Router request, context and native child unload verified", flush=True)
                return
            threading.Event().wait(0.05)


@contextlib.contextmanager
def server(args, draft, folder):
    ensure_available(args)
    path = model_path()
    args.projector = projector_path(args, path)
    model = model_info(path, mtp=bool(draft))
    identity = runner_info(args, draft)
    link = RUNTIME / "model.gguf"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise ValueError("Expected .runtime/model.gguf to be a recipe-owned symlink")
    link.symlink_to(path)
    info = {"model_check": model, "runner": identity, "context": args.context,
            "draft": draft, "reasoning_budget": args.reasoning_budget,
            "projector": ({"filename": args.projector.name, "bytes": args.projector.stat().st_size}
                          if args.projector else None),
            "platform": {"system": platform.system(), "release": platform.release(),
                                        "machine": platform.machine()}, "memory": "unknown"}
    print("Vision: projector selected" if args.projector else
          "Text-only: no projector selected (set MMPROJ_PATH or place mmproj-F16.gguf beside the model)", flush=True)
    started = time.monotonic()
    with owned(command(args, draft), folder / "server.log") as proc:
        monitor = Monitor(proc.pid)
        monitor.thread.start()
        try:
            wait_ready(proc, args)
            info["startup_seconds"] = time.monotonic() - started
            print("Ready: http://127.0.0.1:" + str(args.port), flush=True)
            yield info, proc
        finally:
            monitor.stop.set()
            monitor.thread.join(timeout=65)
            info["memory"] = monitor.report()
            (folder / "run.json").write_text(json.dumps(info, indent=2) + "\n")


class _RequestExpired(BaseException):
    """Bypass urllib's OSError translation while its opening cleanup unwinds."""


@contextlib.contextmanager
def request_deadline(seconds, started):
    # The macOS CLI owns the main thread. Never steal an embedding caller's timer.
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        raise ValueError("Request deadlines require the main thread and POSIX setitimer")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError("Request deadline cannot replace an existing real-time timer")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise _RequestExpired()

    remaining = seconds - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("Request deadline exceeded")
    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            yield
        except _RequestExpired:
            raise TimeoutError("Request deadline exceeded") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def stream(args, endpoint, payload):
    started, first, events = time.monotonic(), None, []
    with request_deadline(args.request_timeout, started), \
         api(args.port, endpoint, payload, timeout=args.request_timeout) as response:
        done = False
        for raw in response:
            if time.monotonic() - started > args.request_timeout:
                raise TimeoutError("Request deadline exceeded")
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise ValueError("Server returned error: " + str(event["error"]))
            events.append(event)
            delta = (event.get("choices") or [{}])[0].get("delta", {})
            if first is None and (event.get("content") or delta.get("content") or delta.get("tool_calls")):
                first = time.monotonic() - started
            if event.get("stop") is True:
                done = True
        if not done:
            raise ValueError("Incomplete streamed response")
    return events, first, time.monotonic() - started


def metrics(events, ttft, elapsed):
    final = events[-1]
    timing = next((e["timings"] for e in reversed(events) if "timings" in e), {})
    n, ms = timing.get("predicted_n"), timing.get("predicted_ms")
    rate = timing.get("predicted_per_second")
    if not n or not ms or rate is None or ttft is None or final.get("truncated"):
        raise ValueError("Missing generation timing/text or truncated context; not a valid benchmark")
    # The pinned server excludes the first token from its decode interval/rate.
    return {"ttft_seconds": ttft, "request_seconds": elapsed, "output_tokens": n,
            "accepted_output_tokens_per_second": rate,
            "end_to_end_tokens_per_second": n / elapsed, "timings": timing,
            "draft_tokens": timing.get("draft_n", "unknown"),
            "draft_accepted": timing.get("draft_n_accepted", "unknown"),
            "occupied_tokens": timing.get("prompt_n", 0) + timing.get("cache_n", 0) + n}


def chat(args, messages, **extra):
    payload = {"model": ALIAS, "messages": messages, "stream": True,
               "temperature": 0, "seed": 42, "max_tokens": 256, **extra}
    return stream(args, "/v1/chat/completions", payload)[0]


def text_from(events):
    return "".join(c.get("delta", {}).get("content", "") or ""
                   for e in events for c in e.get("choices", []))


def smoke(args):
    text = text_from(chat(args, [{"role": "user", "content": "Reply with the word ready."}]))
    if not text.strip():
        raise ValueError("Text smoke produced no text")
    tool = {"type": "function", "function": {"name": "echo", "description": "Echo the given value",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                           "required": ["value"], "additionalProperties": False}}}
    messages = [{"role": "user", "content": "Call echo with value metal-ready, then report its result."}]
    events = chat(args, messages, tools=[tool], tool_choice="required")
    calls = {}
    for e in events:
        for c in e.get("choices", []):
            for part in c.get("delta", {}).get("tool_calls", []):
                call = calls.setdefault(part["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if part.get("id"):
                    call["id"] = part["id"]
                for key in ("name", "arguments"):
                    call["function"][key] += part.get("function", {}).get(key, "") or ""
    if len(calls) != 1:
        raise ValueError("Expected exactly one streamed tool call")
    call = next(iter(calls.values()))
    value = json.loads(call["function"]["arguments"])
    if not call["id"] or call["function"]["name"] != "echo" or value != {"value": "metal-ready"}:
        raise ValueError("Tool smoke returned invalid name/id/arguments")
    messages += [{"role": "assistant", "content": None, "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(value)}]
    reply = text_from(chat(args, messages, tools=[tool], tool_choice="none"))
    if "metal-ready" not in reply:
        raise ValueError("Tool-result response did not include the result")
    return {"text": text, "tool_call": call, "tool_result_reply": reply}


def schedule(repeats, drafts):
    for repetition in range(repeats):
        for draft in drafts:
            for mode in ([0, draft] if repetition % 2 == 0 else [draft, 0]):
                yield repetition, draft, mode


def benchmark(args, folder):
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else (
        "Write a concise explanation of why deterministic tests are useful, with three Python examples.")
    if not prompt.strip():
        raise ValueError("Benchmark prompt is empty")
    payload = {"prompt": prompt, "n_predict": args.tokens, "stream": True,
               "temperature": 0, "seed": 42, "top_k": 1, "top_p": 1, "min_p": 0,
               "cache_prompt": False}
    meta = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_bytes": len(prompt.encode()),
            "reasoning_budget": args.reasoning_budget,
            "sampling": {k: v for k, v in payload.items() if k != "prompt"},
            "cold_definition": "first request after process start; OS file cache uncontrolled",
            "warm_definition": "repeated identical prompt with cache_prompt=true", "records": []}
    plan = schedule(args.repeats, args.drafts) if args.action == "bench-mtp" else (
        (i, 0, 0) for i in range(args.repeats))
    try:
        for i, pair, draft in plan:
            run = folder / f"repeat-{i}-pair-{pair}-draft-{draft}"
            run.mkdir()
            with server(args, draft, run):
                for warm in (False, True):
                    events, ttft, elapsed = stream(args, "/completion", {**payload, "cache_prompt": warm})
                    result = metrics(events, ttft, elapsed)
                    result.update(repetition=i, pair=pair, draft=draft, warm=warm, run=run.name)
                    meta["records"].append(result)
                    (run / ("warm.json" if warm else "cold.json")).write_text(json.dumps(events, indent=2))
                    print(json.dumps(result), flush=True)
    finally:
        (folder / "benchmark.json").write_text(json.dumps(meta, indent=2) + "\n")


def arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("action", choices=["doctor", "doctor-single", "build", "serve", "serve-mtp", "serve-baseline", "smoke", "smoke-single", "bench", "bench-mtp"])
    p.add_argument("--config", default="local/models.ini", help="Native router INI, paths relative to repository root")
    p.add_argument("--model-id", default=ALIAS, help="Router smoke model section ID (256K preset)")
    p.add_argument("--context", type=int, default=int(os.environ.get("CONTEXT", DEFAULT_CONTEXT)))
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--draft", type=int, choices=[1, 2, 4], default=4)
    p.add_argument("--drafts", type=int, nargs="+", choices=[1, 2, 4], default=[1, 2, 4])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--prompt-file")
    mtp = p.add_mutually_exclusive_group()
    mtp.add_argument("--mtp", action="store_true", default=None, help="Legacy single-model MTP (default for serve-mtp)")
    mtp.add_argument("--no-mtp", dest="mtp", action="store_false", help="Explicit baseline")
    p.add_argument("--mmproj-path", default=os.environ.get("MMPROJ_PATH"), help="Existing projector; no downloads")
    p.add_argument("--text-only", action="store_true", help="Disable projector, including adjacent discovery")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reasoning-budget", type=int, help="Native budget: -1 unlimited, 0 disabled, positive token budget")
    p.add_argument("--startup-timeout", type=float, default=300)
    p.add_argument("--request-timeout", type=float, default=600)
    a = p.parse_args(argv)
    a.router_mode = a.action in ("serve", "doctor", "smoke")
    model_flags = {"--context", "--draft", "--mtp", "--no-mtp", "--mmproj-path",
                   "--text-only", "--reasoning-budget"}
    if a.router_mode and any(token.split("=", 1)[0] in model_flags for token in argv):
        p.error("Set model-specific options in the router INI; use serve-mtp/serve-baseline or smoke-single for legacy flags")
    if a.mtp is None:
        a.mtp = a.action in ("serve", "serve-mtp")
    if a.reasoning_budget is None:
        a.reasoning_budget = 0 if a.action in ("smoke", "smoke-single", "bench", "bench-mtp") else 1024
    if a.reasoning_budget < -1:
        p.error("reasoning budget must be >= -1")
    if a.context < 131072 or a.context > 262144:
        p.error("context must be 131072..262144; never silently shrink")
    if a.repeats < 1 or a.tokens < 1 or a.tokens >= a.context or not 1 <= a.port <= 65535:
        p.error("invalid repeats, tokens or port")
    if (not all(math.isfinite(t) and t > 0 for t in (a.startup_timeout, a.request_timeout))
            or len(set(a.drafts)) != len(a.drafts)):
        p.error("timeouts must be positive and drafts unique")
    return a


def main(argv=None):
    args = arguments(argv)
    draft = selected_draft(args)
    if args.dry_run:
        if args.router_mode:
            print(shlex.join([Path(runner()).name, *router_command(args)[1:]]))
            return
        if not args.text_only and (args.mmproj_path or args.action.startswith("serve")):
            args.projector = projector_path(args, model_path())
        print(shlex.join([Path(runner()).name, *command(args, draft)[1:]]))
        return
    with exclusive():
        if args.action == "build":
            build()
            return
        if args.action == "doctor-single":
            print(json.dumps({"tools": {t: bool(shutil.which(t)) for t in ("git", "cmake", "ninja", "xcodebuild")},
                              "model": model_info(model_path(), args.mtp)}, indent=2))
            print(json.dumps(runner_info(args, draft), indent=2))
            return
        folder = ROOT / "results" / str(time.time_ns())
        folder.mkdir(parents=True)
        print("Results:", folder.relative_to(ROOT), flush=True)
        if args.router_mode:
            with router(args, folder) as (info, proc):
                if args.action == "doctor":
                    print(json.dumps({"models": [{"id": m["id"], "status": m["status"]["value"]}
                                                  for m in info["models"]["data"]]}, indent=2))
                elif args.action == "smoke":
                    router_smoke(args, folder, info["models"], proc)
                elif proc.wait():
                    raise ValueError("Router failed; inspect preserved log")
        elif args.action.startswith("bench"):
            benchmark(args, folder)
        else:
            with server(args, draft, folder) as (_, proc):
                if args.action == "smoke-single":
                    (folder / "smoke.json").write_text(json.dumps(smoke(args), indent=2))
                    print("Text and streamed tool/result smoke passed")
                elif proc.wait():
                    raise ValueError("Server failed; inspect preserved log")


def interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; owned server process group reaped.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as exc:
        print("Error:", exc, file=sys.stderr)
        sys.exit(1)
