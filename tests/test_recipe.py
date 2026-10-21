import contextlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recipe as r
from gguf_meta import inspect, require_mtp


class ModelSelectionTests(unittest.TestCase):
    metadata = {"general.name": "Qwen3.8-27B", "general.architecture": "qwen35"}

    def test_explicit_q3_and_renamed_files_accept_actual_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("Qwen3.8-27B-UD-Q3_K_XL.gguf", "renamed.bin"):
                path = Path(directory) / name
                path.write_bytes(b"fixture")
                with patch.dict(r.os.environ, {"MODEL_PATH": str(path)}), \
                     patch.object(r, "inspect", return_value=(self.metadata, set())), \
                     patch.object(Path, "glob", side_effect=AssertionError("Explicit paths need no discovery")):
                    selected = r.model_path()
                    self.assertEqual(selected, path.resolve())
                    self.assertEqual(r.model_info(selected), {"filename": name, "bytes": 7,
                                     "metadata": self.metadata, "mtp_tensors": []})

    def test_missing_or_directory_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in (Path(directory), Path(directory) / "missing.gguf"):
                with patch.dict(r.os.environ, {"MODEL_PATH": str(path)}):
                    with self.assertRaisesRegex(ValueError, "existing GGUF file"):
                        r.model_path()

    def test_discovery_requires_one_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            q3, q4 = [Path(directory) / name for name in ("Qwen3.8-27B-Q3.gguf", "Qwen3.8-27B-Q4.gguf")]
            q3.touch()
            q4.touch()
            with patch.dict(r.os.environ, {"MODEL_PATH": ""}):
                for candidates in ([], [q3, q4]):
                    with patch.object(Path, "glob", return_value=candidates):
                        with self.assertRaisesRegex(ValueError, "exactly one"):
                            r.model_path()
                with patch.object(Path, "glob", return_value=[q3]):
                    self.assertEqual(r.model_path(), q3.resolve())

    def test_discovery_prefers_iq3_but_ambiguous_preferred_files_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            preferred = Path(directory) / r.PREFERRED_MODEL
            other = Path(directory) / "Qwen3.8-27B-Q4.gguf"
            preferred.touch()
            other.touch()
            with patch.dict(r.os.environ, {"MODEL_PATH": ""}), \
                 patch.object(Path, "glob", return_value=[other, preferred]):
                self.assertEqual(r.model_path(), preferred.resolve())
            with patch.dict(r.os.environ, {"MODEL_PATH": ""}), \
                 patch.object(Path, "glob", return_value=[preferred, preferred]):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    r.model_path()

    def test_wrong_metadata_and_missing_mtp_rejected(self):
        for key in self.metadata:
            with patch.object(r, "inspect", return_value=({**self.metadata, key: "wrong"}, set())):
                with self.assertRaisesRegex(ValueError, "Wrong model metadata"):
                    r.model_info(Path("unused"))
        with patch.object(r, "inspect", return_value=(self.metadata, set())):
            with self.assertRaisesRegex(ValueError, "one nextn block"):
                r.model_info(Path("unused"), mtp=True)

    def test_import_does_not_read_manifest(self):
        spec = importlib.util.spec_from_file_location("recipe_without_manifest", r.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch.object(Path, "read_text", side_effect=AssertionError("No manifest reads")):
            spec.loader.exec_module(module)
        self.assertFalse((r.ROOT / "recipes/qwen3.8-27b.json").exists())
        self.assertEqual(module.ALIAS, "qwen3.8-27b")


class RecipeTests(unittest.TestCase):
    def test_command_preserves_baseline_and_only_mtp_changes(self):
        args = r.arguments(["serve-mtp"])
        baseline = r.command(args, 0)
        draft = r.command(args, 2)
        for flag, value in [("--ctx-size", "262144"), ("--cache-type-k", "q8_0"),
                            ("--cache-type-v", "q8_0"), ("--parallel", "1"),
                            ("--cache-ram", "0"), ("--fit", "off"), ("--device", "MTL0"),
                            ("--flash-attn", "on"), ("--n-gpu-layers", "all"),
                            ("--cors-origins", "localhost"), ("--reasoning-budget", "1024"),
                            ("--port", "39483"), ("--n-predict", "4096"),
                            ("--reasoning-format", "deepseek"), ("--reasoning-effort", "low"),
                            ("--temp", "1"), ("--top-p", "0.95"), ("--top-k", "20"),
                            ("--min-p", "0"), ("--presence-penalty", "0"), ("--repeat-penalty", "1")]:
            self.assertEqual(baseline[baseline.index(flag) + 1], value)
        self.assertIn("--no-context-shift", baseline)
        self.assertIn("--no-mmproj", baseline)
        self.assertEqual(baseline[-2:], ["--spec-type", "none"])
        self.assertEqual(draft[:-5], baseline[:-1])
        self.assertEqual(draft[-5:], ["draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-n-min", "0"])

    def test_default_modes_and_explicit_baseline(self):
        for action, expected in (("serve-mtp", 4), ("serve-baseline", 0),
                                 ("smoke-single", 0), ("bench", 0), ("bench-mtp", 0)):
            self.assertEqual(r.selected_draft(r.arguments([action])), expected)
        self.assertEqual(r.selected_draft(r.arguments(["serve-mtp", "--no-mtp"])), 0)
        self.assertEqual(r.selected_draft(r.arguments(["smoke-single", "--mtp"])), 4)
        kwargs = r.command(r.arguments(["serve-mtp"]), 4)
        self.assertEqual(json.loads(kwargs[kwargs.index("--chat-template-kwargs") + 1]),
                         {"preserve_thinking": True})

    def test_projector_selection_and_explicit_text_only(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(r.os.environ, {"MMPROJ_PATH": ""}):
            model = Path(directory) / "model.gguf"
            projector = Path(directory) / "mmproj-F16.gguf"
            args = r.arguments(["serve-mtp"])
            self.assertIsNone(r.projector_path(args, model))
            projector.touch()
            args.projector = r.projector_path(args, model)
            self.assertEqual(args.projector, projector.resolve())
            cmd = r.command(args, 4)
            self.assertEqual(cmd[cmd.index("--mmproj") + 1], str(projector.resolve()))
            self.assertNotIn("--no-mmproj", cmd)
            self.assertIsNone(r.projector_path(r.arguments(["serve-mtp", "--text-only"]), model))
            self.assertIsNone(r.projector_path(r.arguments(["bench"]), model))
            explicit = r.arguments(["smoke-single", "--mmproj-path", str(projector)])
            self.assertEqual(r.projector_path(explicit, model), projector.resolve())
            explicit.mmproj_path = str(Path(directory) / "missing")
            with self.assertRaisesRegex(ValueError, "existing projector"):
                r.projector_path(explicit, model)

    def test_invalid_inputs_fail_before_launch(self):
        for extra in (["--context", "4096"], ["--tokens", "0"], ["--draft", "3"],
                      ["--startup-timeout", "0"], ["--drafts", "2", "2"],
                      ["--reasoning-budget", "-2"], ["--request-timeout", "nan"],
                      ["--request-timeout", "inf"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    r.arguments(["serve-mtp", *extra])

    def test_environment_overrides_cannot_change_recipe(self):
        with patch.dict(r.os.environ, {"LLAMA_ARG_MODEL_URL": "unwanted", "PATH": "test"}):
            self.assertNotIn("LLAMA_ARG_MODEL_URL", r.clean_env())
            self.assertEqual(r.clean_env()["PATH"], "test")

    def test_schedule_counterbalances_each_pair(self):
        self.assertEqual(list(r.schedule(2, [1, 2])),
                         [(0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 2),
                          (1, 1, 1), (1, 1, 0), (1, 2, 2), (1, 2, 0)])

    def test_benchmark_counts_emitted_not_drafted_tokens(self):
        result = r.metrics([{"stop": True, "timings": {"predicted_n": 12, "predicted_ms": 3000,
                            "predicted_per_second": 11 / 3, "prompt_n": 10, "cache_n": 6, "draft_n": 40, "draft_n_accepted": 8}}], 1, 5)
        self.assertEqual(result["accepted_output_tokens_per_second"], 11 / 3)
        self.assertEqual(result["end_to_end_tokens_per_second"], 2.4)
        self.assertEqual(result["occupied_tokens"], 28)
        self.assertEqual(result["draft_accepted"], 8)

    def test_invalid_benchmark_is_not_a_success(self):
        with self.assertRaises(ValueError):
            r.metrics([{"truncated": True, "timings": {"predicted_n": 12, "predicted_ms": 3000}}], 1, 5)
        with self.assertRaises(ValueError):
            r.metrics([{}], None, 5)

    def test_stream_ignores_role_for_ttft_and_requires_done(self):
        args = r.arguments(["smoke"])
        chunks = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                  b'data: {"choices":[{"delta":{"content":"ready"}}]}\n', b'data: [DONE]\n']
        with patch.object(r, "api", return_value=io.BytesIO(b"".join(chunks))), \
             patch.object(r.time, "monotonic", side_effect=[0, 0, 1, 2, 3, 4, 5]):
            events, ttft, elapsed = r.stream(args, "/v1/chat/completions", {})
        self.assertEqual(ttft, 3)
        self.assertEqual(elapsed, 5)
        self.assertEqual(r.text_from(events), "ready")
        with patch.object(r, "api", return_value=io.BytesIO(chunks[1])):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                r.stream(args, "/v1/chat/completions", {})

    def test_reasoning_defaults_and_overrides_match_between_modes(self):
        for action in ("serve-mtp", "smoke-single", "bench", "bench-mtp"):
            default = 1024 if action.startswith("serve") else 0
            self.assertEqual(r.arguments([action]).reasoning_budget, default)
            for budget in (-1, 0, 64):
                with self.subTest(action=action, budget=budget):
                    args = r.arguments([action, "--reasoning-budget", str(budget)])
                    baseline, mtp = r.command(args, 0), r.command(args, 2)
                    self.assertEqual(baseline[baseline.index("--reasoning-budget") + 1], str(budget))
                    self.assertEqual(mtp[:-5], baseline[:-1])

    def test_benchmark_records_chosen_budget_for_every_pair(self):
        args = r.arguments(["bench-mtp", "--repeats", "1", "--drafts", "2", "--reasoning-budget", "64"])
        commands = []

        @contextlib.contextmanager
        def server(args, draft, folder):
            commands.append(r.command(args, draft))
            yield {}, None

        events = [{"stop": True, "timings": {"predicted_n": 2, "predicted_ms": 1,
                                             "predicted_per_second": 1000}}]
        with tempfile.TemporaryDirectory() as directory, patch.object(r, "server", server), \
             patch.object(r, "stream", return_value=(events, 0.1, 0.2)), \
             contextlib.redirect_stdout(io.StringIO()):
            r.benchmark(args, Path(directory))
            meta = json.loads((Path(directory) / "benchmark.json").read_text())
        self.assertEqual(meta["reasoning_budget"], 64)
        self.assertEqual(len(meta["records"]), 4)
        self.assertEqual(commands[1][:-5], commands[0][:-1])

    def test_run_metadata_records_chosen_budget(self):
        args = r.arguments(["serve-mtp", "--reasoning-budget", "64"])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model = folder / "selected-Q3.gguf"
            model.write_bytes(b"selected")
            runtime = folder / "runtime"
            runtime.mkdir()
            with patch.object(r.subprocess, "run", return_value=Mock(returncode=1)), \
                 patch.object(r.socket, "socket"), patch.object(r, "model_path", return_value=model), \
                 patch.object(r, "inspect", return_value=(ModelSelectionTests.metadata, set())), patch.object(r, "runner_info", return_value={}), \
                 patch.object(r, "RUNTIME", runtime), patch.object(r, "owned") as owned, \
                 patch.object(r, "Monitor") as monitor, patch.object(r, "wait_ready"), \
                 contextlib.redirect_stdout(io.StringIO()):
                monitor.return_value.report.return_value = {}
                owned.return_value.__enter__.return_value = Mock(pid=123)
                with r.server(args, 0, folder):
                    pass
            recorded = json.loads((folder / "run.json").read_text())
            self.assertEqual(recorded["reasoning_budget"], 64)
            self.assertEqual(recorded["model_check"], {"filename": "selected-Q3.gguf", "bytes": 8,
                             "metadata": ModelSelectionTests.metadata, "mtp_tensors": []})
            self.assertNotIn("recipe", recorded)

    def test_deadline_restores_state_on_error_and_interrupt(self):
        previous = signal.getsignal(signal.SIGALRM)
        for error in (ValueError("bad response"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                with patch.object(r, "api", side_effect=error):
                    with self.assertRaises(type(error)):
                        r.stream(r.arguments(["smoke"]), "/completion", {})
                self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
                self.assertIs(signal.getsignal(signal.SIGALRM), previous)

    def test_deadline_arms_remaining_budget_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGALRM)
        with patch.object(r.time, "monotonic", return_value=12), \
             patch.object(r.signal, "setitimer") as timer:
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                with r.request_deadline(5, started=10):
                    signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
        self.assertEqual(timer.call_args_list[0].args, (signal.ITIMER_REAL, 3))
        self.assertEqual(timer.call_args_list[-1].args, (signal.ITIMER_REAL, 0))
        self.assertIs(signal.getsignal(signal.SIGALRM), previous)
        with patch.object(r.time, "monotonic", return_value=15):
            with self.assertRaises(TimeoutError):
                with r.request_deadline(5, started=10):
                    self.fail("Expired budget must not open HTTP")

    def test_deadline_fails_safe_without_stealing_existing_timer_or_worker_thread(self):
        with patch.object(r.signal, "getitimer", return_value=(10, 2)), \
             patch.object(r.signal, "setitimer") as timer, patch.object(r.signal, "signal") as handler:
            with self.assertRaisesRegex(ValueError, "existing"):
                r.stream(r.arguments(["smoke"]), "/completion", {})
            timer.assert_not_called()
            handler.assert_not_called()
        with patch.object(r.threading, "current_thread", return_value=object()), \
             patch.object(r, "api") as api:
            with self.assertRaisesRegex(ValueError, "main thread"):
                r.stream(r.arguments(["smoke"]), "/completion", {})
            api.assert_not_called()

    def test_smoke_reassembles_tool_fragments_and_submits_result(self):
        def event(delta):
            return {"choices": [{"delta": delta}]}
        responses = [[event({"content": "ready"})],
                     [event({"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "echo", "arguments": '{"value":'}}]}),
                      event({"tool_calls": [{"index": 0, "function": {"arguments": '"metal-ready"}'}}]})],
                     [event({"content": "metal-ready"})]]
        with patch.object(r, "chat", side_effect=responses) as chat:
            result = r.smoke(r.arguments(["smoke"]))
        self.assertEqual(result["tool_call"]["function"]["arguments"], '{"value":"metal-ready"}')
        self.assertEqual(chat.call_args_list[1].kwargs["tool_choice"], "required")
        self.assertEqual(chat.call_args.args[1][-1]["role"], "tool")
        self.assertEqual(chat.call_args.args[1][-1]["tool_call_id"], "call-1")

    def test_readiness_checks_actual_context(self):
        proc = Mock()
        proc.poll.return_value = None
        args = r.arguments(["serve-mtp"])
        responses = [io.BytesIO(b'{"status":"ok"}'), io.BytesIO(
            b'{"default_generation_settings":{"n_ctx":4096},"total_slots":1}')]
        with patch.object(r, "api", side_effect=responses):
            with self.assertRaisesRegex(ValueError, "context/slot"):
                r.wait_ready(proc, args)

    def test_readiness_deadline_without_waiting_production_duration(self):
        with patch.object(r.time, "monotonic", side_effect=[0, 301]):
            with self.assertRaises(TimeoutError):
                r.wait_ready(Mock(), r.arguments(["serve-mtp"]))

    def test_owned_process_reaped_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "exercise cleanup"):
                with r.owned([sys.executable, "-c", "import signal; signal.pause()"], Path(directory) / "server.log") as proc:
                    raise RuntimeError("exercise cleanup")
            self.assertIsNotNone(proc.poll())
            with self.assertRaises(ProcessLookupError):
                r.os.killpg(proc.pid, 0)

    def test_cleanup_kills_group_even_when_leader_exited(self):
        proc = Mock(pid=999999)
        with patch.object(r.os, "killpg", side_effect=[None, None, ProcessLookupError()]) as kill:
            r.terminate(proc, grace=0.01)
        self.assertEqual(kill.call_args_list[0].args, (proc.pid, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (proc.pid, signal.SIGKILL))
        self.assertEqual(proc.wait.call_count, 2)

    def test_memory_unknown_is_not_zero(self):
        monitor = r.Monitor(0)
        self.assertEqual(monitor.report()["rss_bytes"], "unknown")
        monitor.samples = [{"rss_bytes": 2, "system_swap_bytes": 10},
                           {"rss_bytes": 7, "system_swap_bytes": 8}]
        self.assertEqual(monitor.report()["rss_bytes"]["sampled_peak"], 7)
        self.assertEqual(monitor.report()["system_swap_bytes"]["delta"], -2)

    def test_repository_lock_rejects_overlapping_owner(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(r, "RUNTIME", Path(directory)):
            with r.exclusive():
                with self.assertRaisesRegex(ValueError, "Another recipe"):
                    with r.exclusive():
                        self.fail("Second owner must not enter")
            with r.exclusive():
                pass

    def test_owned_process_reaped_on_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                with r.owned([sys.executable, "-c", "import signal; signal.pause()"], Path(directory) / "server.log") as proc:
                    raise KeyboardInterrupt
            self.assertIsNotNone(proc.poll())
            with self.assertRaises(ProcessLookupError):
                r.os.killpg(proc.pid, 0)

    def test_missing_flags_rejected_without_starting_server(self):
        with patch.object(r, "capture", return_value="--help"):
            with self.assertRaisesRegex(ValueError, "lacks required flags"):
                r.runner_info(r.arguments(["serve-mtp"]), 2)


class StreamHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.finished = threading.Event()
        cls.disconnected = threading.Event()
        cls.stop = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                try:
                    if self.path == "/headers":
                        self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Trickle: ")
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        if self.path == "/success":
                            self.wfile.write(b'data: {"content":"ready","stop":true}\n\ndata: [DONE]\n\n')
                            return
                        self.wfile.write(b'data: {"content":"')
                    # Deliberate byte pacing exercises a peer that never finishes a line.
                    for _ in range(100):  # Finite peer lifetime also bounds a deadline regression.
                        if cls.stop.wait(0.005):
                            break
                        self.wfile.write(b"x")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    cls.disconnected.set()
                finally:
                    cls.finished.set()

        cls.http = HTTPServer(("127.0.0.1", 0), Handler)
        cls.worker = threading.Thread(target=cls.http.serve_forever, kwargs={"poll_interval": 0.01})
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.stop.set()
        cls.http.shutdown()
        cls.http.server_close()
        cls.worker.join(timeout=2)
        if cls.worker.is_alive():
            raise AssertionError("HTTP fixture thread leaked")
        if cls.http.socket.fileno() != -1:
            raise AssertionError("HTTP fixture socket leaked")

    def test_opening_and_partial_line_deadlines_close_connection(self):
        previous = signal.getsignal(signal.SIGALRM)
        for endpoint in ("/headers", "/partial"):
            with self.subTest(endpoint=endpoint):
                self.finished.clear()
                self.disconnected.clear()
                args = r.arguments(["smoke", "--port", str(self.http.server_port), "--request-timeout", "0.08"])
                with self.assertRaisesRegex(TimeoutError, "deadline"):
                    r.stream(args, endpoint, {})
                self.assertTrue(self.finished.wait(2), "HTTP handler did not finish")
                self.assertTrue(self.disconnected.is_set(), "Client connection leaked")
                self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
                self.assertIs(signal.getsignal(signal.SIGALRM), previous)

    def test_successful_sse_restores_timer(self):
        args = r.arguments(["smoke", "--port", str(self.http.server_port), "--request-timeout", "1"])
        events, first, elapsed = r.stream(args, "/success", {})
        self.assertEqual(events, [{"content": "ready", "stop": True}])
        self.assertIsNotNone(first)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))


class PiConfigTests(unittest.TestCase):
    def test_documented_provider_limits_and_thinking_overrides(self):
        text = (r.ROOT / "docs/pi.md").read_text()
        config = json.loads(text.split("```json\n", 1)[1].split("```", 1)[0])
        provider = config["providers"]["llamacpp-qwen38"]
        self.assertEqual(provider["baseUrl"], f"http://127.0.0.1:{r.DEFAULT_PORT}/v1")
        self.assertEqual(provider["api"], "openai-completions")
        model, = provider["models"]
        self.assertEqual(model["id"], r.ALIAS)
        self.assertEqual(model["contextWindow"], r.DEFAULT_CONTEXT)
        self.assertEqual(model["maxTokens"], 4096)
        self.assertTrue(model["reasoning"])
        self.assertEqual(model["input"], ["text", "image"])
        self.assertEqual(model["thinkingLevelMap"], {"off": "none", "minimal": "low", "low": "low",
                                                   "medium": "medium", "high": "xhigh", "xhigh": "xhigh"})
        defaults = model["samplingParams"]
        self.assertEqual(defaults["reasoning_budget_tokens"], 1024)
        for key, value in {"temperature": 1, "top_p": .95, "top_k": 20, "min_p": 0,
                           "presence_penalty": 0, "repeat_penalty": 1}.items():
            self.assertEqual(defaults[key], value)
        self.assertFalse({"max_tokens", "n_predict", "max_completion_tokens"} & defaults.keys())
        off = {**defaults, **model["samplingParamsByThinkingLevel"]["off"]}
        self.assertEqual(off["chat_template_kwargs"], {"enable_thinking": False, "preserve_thinking": True})
        self.assertEqual((off["temperature"], off["top_p"], off["presence_penalty"]), (.7, .8, 1.5))


class GGUFTests(unittest.TestCase):
    def test_reads_directory_without_weight_data(self):
        def string(value):
            raw = value.encode()
            return struct.pack("<Q", len(raw)) + raw
        data = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
        data += string("general.architecture") + struct.pack("<I", 8) + string("qwen35")
        data += string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + string("one") + string("two")
        data += string("blk.64.nextn.eh_proj.weight") + struct.pack("<IQQIQ", 2, 2, 2, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            path.write_bytes(data)
            metadata, names = inspect(path)
            self.assertEqual(metadata, {"general.architecture": "qwen35"})
            self.assertIn("blk.64.nextn.eh_proj.weight", names)
            path.write_bytes(data[:20])
            with self.assertRaises(ValueError):
                inspect(path)

    def test_mtp_requires_real_projection_and_attention_tensors(self):
        with self.assertRaisesRegex(ValueError, "MTP tensors missing"):
            require_mtp({"general.architecture": "qwen35", "qwen35.block_count": 65,
                         "qwen35.nextn_predict_layers": 1}, {"blk.64.nextn.eh_proj.weight"})
        with self.assertRaisesRegex(ValueError, "one nextn block"):
            require_mtp({"general.architecture": "qwen35"}, set())


if __name__ == "__main__":
    unittest.main()
