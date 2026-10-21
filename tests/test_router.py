"""Router entry point and native configuration boundary (no weights/network)."""
import configparser
import contextlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recipe as r


class RouterTests(unittest.TestCase):
    def test_default_and_override_config_command_has_only_router_options(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(r, "ROOT", Path(directory)):
            root = Path(directory)
            (root / "local").mkdir()
            (root / "local/models.ini").write_text("version = 1\n")
            for action in ("serve", "doctor", "smoke"):
                cmd = r.router_command(r.arguments([action]))
                self.assertEqual(cmd[1:], ["--models-preset", "local/models.ini", "--models-max", "1",
                                           "--models-autoload", "--host", "127.0.0.1", "--port", "39483",
                                           "--cors-origins", "localhost"])
            alternate = root / "alternate.ini"
            alternate.touch()
            cmd = r.router_command(r.arguments(["serve", "--config", str(alternate)]))
            self.assertEqual(cmd[2], str(alternate))
            with self.assertRaisesRegex(ValueError, "Missing router INI"):
                r.router_command(r.arguments(["serve", "--config", "missing.ini"]))
            with self.assertRaisesRegex(ValueError, "Missing router INI"):
                r.router_command(r.arguments(["serve", "--config", "."]))

    def test_router_flags_reject_silent_model_cli_overrides(self):
        for extra in (["--cont", "131072"], ["--context", "131072"], ["--mtp"], ["--no-mtp"], ["--draft", "2"],
                      ["--reasoning-budget=0"], ["--text-only"], ["--mmproj-path", "other.gguf"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    r.arguments(["serve", *extra])
        with patch.dict(r.os.environ, {"MODEL_PATH": "not-a-file", "MMPROJ_PATH": "absent",
                                       "LLAMA_ARG_MODEL": "unwanted"}):
            args = r.arguments(["serve"])
            with patch.object(Path, "is_file", return_value=True), \
                 patch.object(r, "model_info", side_effect=AssertionError("No Qwen-only gate")):
                self.assertNotIn("--model", r.router_command(args))
            self.assertNotIn("LLAMA_ARG_MODEL", r.clean_env())

    def test_public_example_aliases_and_model_specific_settings(self):
        config = configparser.ConfigParser(interpolation=None)
        config.read_string("[header]\n" + (r.ROOT / "example.ini").read_text())
        shared = config["*"]
        self.assertEqual(shared["ctx-size"], "262144")
        self.assertEqual(shared["fit"], "off")
        self.assertEqual(shared["load-on-startup"], "false")
        qwen = config[r.ALIAS]
        self.assertEqual(qwen["spec-type"], "draft-mtp")
        self.assertEqual(qwen["spec-draft-n-max"], "4")
        self.assertEqual(qwen["reasoning-effort"], "low")
        self.assertIn("mmproj", qwen)
        self.assertNotIn("temp", shared)
        self.assertNotIn("reasoning-effort", shared)
        provider = json.loads((r.ROOT / "examples/pi-models-moe.json").read_text())["providers"]["llamacpp-qwen38"]
        model_ids = {model["id"] for model in provider["models"]}
        self.assertEqual(set(config.sections()) - {"header", "*"}, model_ids)
        self.assertEqual(len(model_ids), 9)
        for name in ("qwen36-q2-k-xl-d4", "qwen36-q3-k-xl-d4", "qwen36-iq4-xs-d4"):
            self.assertEqual(config[name]["temp"], "0.6")
            self.assertEqual(config[name]["spec-type"], "draft-mtp")
            self.assertEqual(config[name]["spec-draft-n-max"], "4")
            self.assertIn("mmproj", config[name])
        gemma = config["gemma4-q4-k-xl-d2"]
        self.assertEqual(gemma["top-k"], "64")
        self.assertEqual(gemma["spec-type"], "draft-mtp")
        self.assertEqual(gemma["spec-draft-n-max"], "2")
        self.assertIn("model-draft", gemma)
        for name, draft in (("nemotron35-iq2-m-d0", None), ("nemotron35-iq2-m-d4", "4"),
                            ("nemotron35-q3-k-xl-d2", "2"), ("nemotron35-q4-k-m-d0", None)):
            self.assertEqual(config[name]["no-mmproj"], "true")
            self.assertEqual(config[name]["spec-type"], "draft-mtp" if draft else "none")
            if draft:
                self.assertEqual(config[name]["spec-draft-n-max"], draft)
            else:
                self.assertNotIn("spec-draft-n-max", config[name])
        for name in model_ids:
            for key in ("model", "mmproj", "model-draft"):
                if key in config[name]:
                    self.assertFalse(Path(config[name][key]).is_absolute())
                    self.assertTrue(config[name][key].startswith(".runtime/models/"))
            if name != r.ALIAS and config[name]["spec-type"] == "draft-mtp":
                self.assertEqual(config[name]["spec-draft-type-k"], "q8_0")
                self.assertEqual(config[name]["spec-draft-type-v"], "q8_0")
        ignored = (r.ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("local/", ignored)
        self.assertIn(".runtime/", ignored)

    def test_main_serve_and_smoke_use_router_not_legacy_model_gate(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(r, "ROOT", Path(directory)), \
             patch.object(r, "exclusive"), patch.object(r, "router") as router, \
             patch.object(r, "server", side_effect=AssertionError("Wrong production seam")), \
             patch.object(r, "model_path", side_effect=AssertionError("No discovery for router")), \
             patch.object(r, "router_smoke") as smoke, contextlib.redirect_stdout(io.StringIO()):
            proc = Mock()
            proc.wait.return_value = 0
            router.return_value.__enter__.return_value = ({"models": {"data": []}}, proc)
            r.main(["serve"])
            proc.wait.assert_called_once()
            r.main(["smoke"])
            smoke.assert_called_once()
            self.assertEqual(router.call_count, 2)

    def test_readiness_uses_models_without_loading_default_model(self):
        proc = Mock()
        proc.poll.return_value = None
        payload = {"data": [{"id": "not-qwen", "status": {"value": "unloaded"}}]}
        with patch.object(r, "api", return_value=io.BytesIO(json.dumps(payload).encode())) as api:
            self.assertEqual(r.wait_router(proc, r.arguments(["doctor"])), payload)
        self.assertEqual(api.call_args.args[1], "/models")
        proc.poll.return_value = 1
        with self.assertRaisesRegex(ValueError, "INI errors"):
            r.wait_router(proc, r.arguments(["doctor"]))
        proc.poll.return_value = None
        with patch.object(r, "api", return_value=io.BytesIO(b'{"data": []}')):
            with self.assertRaisesRegex(ValueError, "no models"):
                r.wait_router(proc, r.arguments(["doctor"]))
        with patch.object(r.time, "monotonic", side_effect=[0, 301]):
            with self.assertRaises(TimeoutError):
                r.wait_router(proc, r.arguments(["doctor"]))

    def test_existing_server_and_process_probe_failure_refuse_launch(self):
        for status in (0, 2):
            with patch.object(r.subprocess, "run", return_value=Mock(returncode=status)):
                with self.assertRaisesRegex(ValueError, "inspect before starting"):
                    r.ensure_available(r.arguments(["serve"]))

    def test_smoke_routes_selected_id_and_verifies_native_unload(self):
        args = r.arguments(["smoke", "--model-id", "other-family"])
        initial = {"data": [{"id": args.model_id, "status": {"value": "unloaded"}}]}
        loaded = {"data": [{"id": args.model_id, "status": {"value": "loaded"}}]}
        props = {"default_generation_settings": {"n_ctx": 262144}, "total_slots": 1}
        events = [{"choices": [{"delta": {"content": "ready"}}]}]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(r, "stream", return_value=(events, 0.1, 0.2)) as stream, \
             patch.object(r, "json_request", side_effect=[props, loaded, {"success": True}]) as request, \
             patch.object(r, "api", return_value=io.BytesIO(json.dumps(initial).encode())), \
             patch.object(r.subprocess, "run", return_value=Mock(returncode=0, stdout="999999\n")), \
             patch.object(r.os, "kill", side_effect=ProcessLookupError()) as kill, \
             contextlib.redirect_stdout(io.StringIO()):
            r.router_smoke(args, Path(directory), initial, Mock(pid=999998))
            self.assertEqual(stream.call_args.args[2]["model"], "other-family")
            self.assertIn("model=other-family", request.call_args_list[0].args[1])
            self.assertEqual(request.call_args.args[1:], ("/models/unload", {"model": "other-family"}))
            self.assertTrue((Path(directory) / "unloaded.json").is_file())
            kill.assert_called_once_with(999999, 0)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))

    def test_smoke_fails_unavailable_or_preloaded_before_request(self):
        for models in ([], [{"id": r.ALIAS, "status": {"value": "loaded"}}]):
            with patch.object(r, "stream") as stream:
                with self.assertRaisesRegex(ValueError, "start unloaded"):
                    r.router_smoke(r.arguments(["smoke"]), Path("unused"), {"data": models}, Mock())
                stream.assert_not_called()

    def test_smoke_rejects_silent_context_reduction(self):
        initial = {"data": [{"id": r.ALIAS, "status": {"value": "unloaded"}}]}
        props = {"default_generation_settings": {"n_ctx": 4096}, "total_slots": 1}
        events = [{"choices": [{"delta": {"content": "ready"}}]}]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(r, "stream", return_value=(events, 0.1, 0.2)), \
             patch.object(r, "json_request", side_effect=[props, initial]):
            with self.assertRaisesRegex(ValueError, "262144 context"):
                r.router_smoke(r.arguments(["smoke"]), Path(directory), initial, Mock())
            self.assertTrue((Path(directory) / "smoke.json").is_file())

    def test_cleanup_reports_lingering_group(self):
        with patch.object(r.os, "killpg"), patch.object(r.time, "monotonic", side_effect=[0, 1]):
            with self.assertRaisesRegex(RuntimeError, "still exists"):
                r.terminate(Mock(pid=999999), grace=0.01)

    def test_owned_native_style_child_group_is_gone_on_failure(self):
        # One real fixture tests descendants, not just the reaped leader.
        child_code = "import signal; signal.pause()"
        code = ("import subprocess,sys,signal; "
                "p=subprocess.Popen([sys.executable,'-c',sys.argv[1]]); "
                "signal.signal(signal.SIGTERM,lambda *a:(p.wait(),sys.exit(0))); "
                "print(p.pid,flush=True); signal.pause()")
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            with self.assertRaisesRegex(ValueError, "exercise"):
                with r.owned([sys.executable, "-c", code, child_code], log) as proc:
                    # Bound readiness on a log signal rather than a guessed sleep.
                    with r.request_deadline(3, r.time.monotonic()):
                        while not log.read_text().strip():
                            r.threading.Event().wait(0.01)
                    child = int(log.read_text().strip())
                    raise ValueError("exercise")
            self.assertIsNotNone(proc.poll())
            with self.assertRaises(ProcessLookupError):
                r.os.kill(child, 0)
            with self.assertRaises(ProcessLookupError):
                r.os.killpg(proc.pid, 0)


if __name__ == "__main__":
    unittest.main()
