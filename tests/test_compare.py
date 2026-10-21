import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import copy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare


class StreamTests(unittest.TestCase):
    def test_stream_preserves_raw_and_canonical_rate(self):
        event = {"content": "ready", "stop": True, "timings": {
            "predicted_n": 3, "predicted_ms": 100, "predicted_per_second": 20,
            "prompt_n": 8, "cache_n": 2, "draft_n": 4, "draft_n_accepted": 2}}
        data = b"data: " + json.dumps(event).encode() + b"\n\n"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.sse"
            with patch.object(compare.r, "api", return_value=io.BytesIO(data)):
                events, first, elapsed = compare.recorded_stream(
                    SimpleNamespace(port=1, request_timeout=1), path, "/completion", {}, SimpleNamespace(progress=0))
            self.assertEqual(path.read_bytes(), data)
            result = compare.r.metrics(events, first, elapsed)
            self.assertEqual(result["accepted_output_tokens_per_second"], 20)
            self.assertEqual(result["occupied_tokens"], 13)

    def test_incomplete_stream_is_retained_not_success(self):
        data = b'data: {"content":"unfinished"}\n'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.sse"
            with patch.object(compare.r, "api", return_value=io.BytesIO(data)):
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    compare.recorded_stream(SimpleNamespace(port=1, request_timeout=1), path,
                                            "/completion", {}, SimpleNamespace(progress=0))
            self.assertEqual(path.read_bytes(), data)

    def test_image_fixture_is_png(self):
        data = compare.red_blue_png()
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IHDR", data)

    def test_tokenizer_receives_template_not_character_estimate(self):
        args = SimpleNamespace()
        with patch.object(compare.r, "json_request", side_effect=[{"prompt": "rendered"}, {"tokens": [4, 5]}]) as request:
            self.assertEqual(compare.prompt_tokens(args, "model", "semantic text"), [4, 5])
        self.assertEqual(request.call_args.args[2]["content"], "rendered")
        self.assertTrue(request.call_args.args[2]["add_special"])


class ContractTests(unittest.TestCase):
    def setUp(self):
        argv = ["server", "--no-context-shift", "--ctx-size", "262144", "--parallel", "1",
                "--n-gpu-layers", "all", "--device", "MTL0", "--cache-type-k", "q8_0",
                "--cache-type-v", "q8_0", "--fit", "off", "--flash-attn", "on", "--spec-type", "draft-mtp"]
        self.loaded = {"data": [{"id": "test", "status": {"value": "loaded", "args": argv}}]}
        self.props = {"default_generation_settings": {"n_ctx": 262144}, "total_slots": 1}

    def test_fixed_native_contract(self):
        self.assertTrue(compare.validate_native(self.loaded, "test", self.props))

    def test_reduced_context_rejected(self):
        self.loaded["data"][0]["status"]["args"][3] = "8192"
        with self.assertRaisesRegex(ValueError, "fixed capacity"):
            compare.validate_native(self.loaded, "test", self.props)

    def test_extra_resident_rejected(self):
        self.loaded["data"].append(copy.deepcopy(self.loaded["data"][0]))
        with self.assertRaisesRegex(ValueError, "exactly"):
            compare.validate_native(self.loaded, "test", self.props)

    def test_pressure_guard_signals_owned_group_without_waiting(self):
        monitor = compare.ChildrenMonitor(12345)
        monitor.progress = 0
        monitor.samples = [{"system_swap_bytes": 0}]
        sample = {"rss_bytes": 1, "system_swap_bytes": 3 * 1024**3}
        with patch.object(compare.r, "memory", return_value=sample), \
             patch.object(compare.r, "capture", return_value="12345 12345 20\n12346 12345 100\n77 77 9000\n"), \
             patch.object(compare.time, "monotonic", return_value=61), \
             patch.object(compare.os, "killpg") as kill:
            monitor.run()
        kill.assert_called_once_with(12345, compare.signal.SIGTERM)
        self.assertEqual(monitor.samples[-1]["rss_bytes"], 120 * 1024)
        self.assertIsNotNone(monitor.aborted)

    def test_pi_failures_preserve_evidence_cleanup_and_suppress_repeats(self):
        content = b'raise AssertionError("generated code must not execute")\n# \xc3\xa9\n'
        for outcome in ("timeout", "nonzero", "missing", "empty", "healthy"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = root / "provider.json"
                config.write_text(json.dumps({"providers": {"llamacpp-qwen38": {
                    "models": [{"id": "test"}]}}}))
                lifecycle, workspaces = [], []
                processes = []
                raw = b'{"type":"tool_execution_start","toolName":"write"}\n'

                @contextmanager
                def router(args, folder):
                    lifecycle.append("router start")
                    try:
                        yield None, SimpleNamespace(pid=12345)
                    finally:
                        lifecycle.append("group cleanup")

                def request(args, endpoint, body=None):
                    if endpoint.startswith("/props?"):
                        return self.props
                    if endpoint == "/models":
                        return self.loaded
                    self.assertEqual(endpoint, "/models/unload")
                    self.assertEqual(body, {"model": "test"})
                    lifecycle.append("unload")
                    return {"success": True}

                def spawn(command, **kwargs):
                    work = kwargs["cwd"]
                    workspaces.append(work)
                    if outcome != "missing":
                        (work / "BFS.py").write_bytes(b"" if outcome == "empty" else content)
                    kwargs["stdout"].write(raw)
                    proc = MagicMock(returncode=7 if outcome == "nonzero" else 0)
                    if outcome == "timeout":
                        proc.wait.side_effect = compare.subprocess.TimeoutExpired(command, 600)
                    processes.append(proc)
                    return proc

                monitor = MagicMock(samples=[], aborted=None)
                monitor.report.return_value = {}
                argv = ["compare.py", "pi", "--models", "test", "--repeats", "2",
                        "--output", "trial", "--pi-config", str(config)]
                with patch.object(sys, "argv", argv), \
                     patch.object(compare.r, "ROOT", root), \
                     patch.object(compare.r, "exclusive"), \
                     patch.object(compare.r, "router", side_effect=router), \
                     patch.object(compare.r, "json_request", side_effect=request), \
                     patch.object(compare.r, "ensure_available", side_effect=lambda args: lifecycle.append("port check")), \
                     patch.object(compare, "ChildrenMonitor", return_value=monitor), \
                     patch.object(compare, "chat", return_value=([{"choices": [{"delta": {"content": "ready"}}]}], 0, 0)), \
                     patch.object(compare.shutil, "which", side_effect=lambda name: "/fake/" + name), \
                     patch.object(compare.subprocess, "Popen", side_effect=spawn), \
                     patch.object(compare.r, "terminate") as terminate, \
                     patch.object(sys, "stdout", io.StringIO()):
                    (root / "results").mkdir()
                    compare.main()
                run = root / "results/trial/0-test"
                result = json.loads((run / "result.json").read_text())
                self.assertIn("pi", result, result)
                summary = json.loads((run / "pi-summary.json").read_text())
                healthy = outcome == "healthy"
                self.assertEqual(result["status"], "completed" if healthy else "failed")
                self.assertEqual(result["pi"], summary)
                self.assertEqual(summary["timeout"], outcome == "timeout")
                self.assertEqual(summary["exit"], 7 if outcome == "nonzero" else 0)
                self.assertEqual(summary["file_present"], outcome != "missing")
                self.assertEqual(summary["file_bytes"], 0 if outcome in ("missing", "empty") else len(content))
                self.assertEqual(summary["correctness"], "pending independent review")
                self.assertEqual((run / "pi.jsonl").read_bytes(), raw)
                if outcome != "missing":
                    self.assertEqual((run / "BFS.py").read_bytes(), b"" if outcome == "empty" else content)
                self.assertEqual(result["cleanup"], "owned group gone and port available")
                self.assertEqual(lifecycle[:4], ["router start", "unload", "group cleanup", "port check"])
                self.assertEqual(len(processes), 2 if healthy else 1)
                self.assertEqual(terminate.call_count, len(processes))
                self.assertEqual(monitor.stop.set.call_count, len(processes))
                self.assertEqual(monitor.thread.join.call_count, len(processes))
                for proc in processes:
                    proc.wait.assert_called_once_with(timeout=600)
                self.assertTrue(all(not work.exists() for work in workspaces))
                self.assertEqual((root / "results/trial/1-test").exists(), healthy)
                self.assertEqual(json.loads((root / "results/trial/failed-models.json").read_text()),
                                 [] if healthy else ["test"])

    def test_pi_trial_isolated_and_reaped(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = folder / "provider.json"
            config.write_text(json.dumps({"providers": {"llamacpp-qwen38": {
                "apiKey": "local", "models": [{"id": "test"}]}}}))
            args = SimpleNamespace(pi_config=str(config))
            workspaces = []
            class Process:
                returncode = 0
                def wait(self, timeout):
                    return 0
            def spawn(command, **kwargs):
                work = kwargs["cwd"]
                workspaces.append(work)
                (work / "BFS.py").write_text("# retained without execution\n")
                event = {"type": "message_end", "message": {"role": "assistant", "usage": {"output": 1},
                         "content": [{"type": "text", "text": "one\u2028two"}]}}
                kwargs["stdout"].write((json.dumps(event, ensure_ascii=False) + "\n").encode())
                self.assertEqual(command[command.index("--tools") + 1], "read,write,edit")
                self.assertIn("--no-session", command)
                self.assertTrue(kwargs["start_new_session"])
                self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
                self.assertTrue(kwargs["env"]["PATH"].startswith("/working-node"))
                return Process()
            with patch.object(compare.shutil, "which", side_effect=lambda name: "/working-node/node" if name == "node" else "/working-pi/pi"), \
                 patch.object(compare.subprocess, "Popen", side_effect=spawn), \
                 patch.object(compare.r, "terminate") as terminate:
                result = compare.pi_trial(args, folder, "test", SimpleNamespace())
            self.assertTrue(result["file_present"])
            self.assertEqual(result["correctness"], "pending independent review")
            self.assertEqual(result["usage"], [{"output": 1}])
            self.assertFalse(workspaces[0].exists())
            terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
