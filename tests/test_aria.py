import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from aria.assistant import Assistant
from aria.backend import BackendError, OllamaBackend
from aria.cli import main
from aria.config import Config, ConfigurationError, load_dotenv
from aria.storage import SessionStore, StorageError
from aria.voice import LocalVoice, VoiceError


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.fail_after_first_chunk = False

    def stream(self, messages):
        self.calls.append(messages)
        yield "reply "
        if self.fail_after_first_chunk:
            raise BackendError("temporary failure")
        yield str(len(self.calls))


class AssistantTests(unittest.TestCase):
    def test_context_is_bounded_but_saved_history_is_full(self):
        backend = FakeBackend()
        assistant = Assistant(backend, history_turns=1)
        assistant.ask("first")
        assistant.ask("second")
        assistant.ask("third")
        self.assertEqual([item["role"] for item in backend.calls[2]],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(backend.calls[2][1]["content"], "second")
        self.assertEqual(len(assistant.history), 6)
        assistant.reset()
        self.assertEqual(assistant.history, [])

    def test_partial_reply_does_not_change_history(self):
        backend = FakeBackend()
        assistant = Assistant(backend, history_turns=2)
        backend.fail_after_first_chunk = True
        with self.assertRaises(BackendError):
            list(assistant.stream_reply("try"))
        self.assertEqual(assistant.history, [])


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path.cwd() / "tests" / "_aria_test_sessions"
        self.store = SessionStore(self.directory)

    def tearDown(self):
        if self.directory.exists():
            for path in self.directory.iterdir():
                if path.is_file():
                    path.unlink()
            self.directory.rmdir()

    def test_roundtrip_and_names(self):
        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        self.store.save("demo", messages)
        self.assertEqual(self.store.load("demo"), messages)
        self.assertEqual(self.store.names(), ["demo"])
        self.store.save("demo", [])
        self.assertEqual(self.store.load("demo"), [])

    def test_invalid_names_and_corrupt_history(self):
        with self.assertRaises(StorageError):
            self.store.save("../outside", [])
        self.directory.mkdir(exist_ok=True)
        (self.directory / "bad.json").write_text('{"messages":[{"role":"system","content":"x"}]}', encoding="utf-8")
        with self.assertRaises(StorageError):
            self.store.load("bad")


class ConfigTests(unittest.TestCase):
    def test_dotenv_does_not_override_process_environment(self):
        with patch.dict(os.environ, {"ARIA_LOCAL_MODEL": "qwen3:4b-instruct"}, clear=True), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value="# comment\nARIA_LOCAL_MODEL=gemma3:1b\nARIA_HISTORY_TURNS='3'\nARIA_API_KEY=old-secret\n"):
            load_dotenv(Path(".env"))
            self.assertEqual(os.environ["ARIA_LOCAL_MODEL"], "qwen3:4b-instruct")
            self.assertEqual(os.environ["ARIA_HISTORY_TURNS"], "3")
            self.assertNotIn("ARIA_API_KEY", os.environ)

    def test_invalid_configuration(self):
        with patch.dict(os.environ, {"ARIA_HISTORY_TURNS": "-1"}, clear=True):
            with self.assertRaises(ConfigurationError):
                Config.from_env()

    def test_remote_address_and_cloud_model_are_rejected(self):
        with patch.dict(os.environ, {"ARIA_OLLAMA_URL": "https://example.com"}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "local HTTP"):
                Config.from_env()
        with patch.dict(os.environ, {"ARIA_LOCAL_MODEL": "test:cloud"}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "downloaded local model"):
                Config.from_env()

    def test_no_api_key_is_required(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Config.from_env().model, "qwen3:4b-instruct")


class LocalHandler(BaseHTTPRequestHandler):
    requests = []
    post_status = 200
    chunks = [
        {"message": {"content": "Hello "}, "done": False},
        {"message": {"content": "from fake model"}, "done": False},
        {"message": {"content": ""}, "done": True},
    ]
    models = [{"name": "test-model"}]

    def do_GET(self):
        self.requests.append(("GET", self.path, dict(self.headers), None))
        self._reply(200, {"models": self.models})

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        self.requests.append(("POST", self.path, dict(self.headers), json.loads(self.rfile.read(size))))
        if self.post_status != 200:
            self._reply(self.post_status, {"error": "model unavailable"})
            return
        payload = b"".join(json.dumps(chunk).encode("utf-8") + b"\n" for chunk in self.chunks)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _reply(self, status, body):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), LocalHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        LocalHandler.requests = []
        LocalHandler.post_status = 200
        LocalHandler.chunks = [
            {"message": {"content": "Hello "}, "done": False},
            {"message": {"content": "from fake model"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        LocalHandler.models = [{"name": "test-model"}]
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.config = Config("test-model", self.url, 2, 10)

    def test_stream_request_and_cli_once(self):
        with patch.dict(os.environ, {"ARIA_LOCAL_MODEL": "test-model", "ARIA_OLLAMA_URL": self.url}), \
             patch("aria.cli.SessionStore") as store_class:
            store_class.return_value.load.return_value = []
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(["--once", "Hi"])
            store_class.return_value.save.assert_called_once()
        self.assertEqual(status, 0)
        self.assertIn("Hello from fake model", output.getvalue())
        self.assertEqual(LocalHandler.requests[0][0:2], ("GET", "/api/tags"))
        method, path, headers, body = LocalHandler.requests[1]
        self.assertEqual((method, path), ("POST", "/api/chat"))
        self.assertNotIn("Authorization", headers)
        self.assertEqual(body["model"], "test-model")
        self.assertTrue(body["stream"])
        self.assertEqual(body["messages"][-1], {"role": "user", "content": "Hi"})
        self.assertEqual(body["messages"][0]["role"], "system")

    def test_http_error_and_incomplete_stream(self):
        backend = OllamaBackend(self.config)
        LocalHandler.post_status = 404
        with self.assertRaisesRegex(BackendError, "HTTP 404.*model unavailable"):
            backend.complete([{"role": "user", "content": "Hi"}])
        LocalHandler.post_status = 200
        LocalHandler.chunks = [{"message": {"content": "partial"}, "done": False}]
        with self.assertRaisesRegex(BackendError, "before completing"):
            backend.complete([{"role": "user", "content": "Hi"}])

    def test_uninstalled_model_is_not_sent_for_inference(self):
        LocalHandler.models = []
        with self.assertRaisesRegex(BackendError, "not installed"):
            OllamaBackend(self.config).complete([{"role": "user", "content": "Hi"}])
        self.assertEqual([item[0] for item in LocalHandler.requests], ["GET"])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path.cwd() / "tests" / "_aria_test_sessions"
        self.store = SessionStore(self.directory)

    def tearDown(self):
        if self.directory.exists():
            for path in self.directory.iterdir():
                if path.is_file():
                    path.unlink()
            self.directory.rmdir()

    def test_interactive_chat_is_saved_and_resumed(self):
        output = io.StringIO()
        with patch("aria.cli.SessionStore", return_value=self.store), \
             patch("aria.cli.OllamaBackend", return_value=FakeBackend()), \
             patch("builtins.input", side_effect=["hello", "/exit"]), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(["--session", "example"]), 0)
        self.assertIn("ARIA > reply 1", output.getvalue())
        self.assertEqual(self.store.load("example"), [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "reply 1"},
        ])
        with patch("aria.cli.SessionStore", return_value=self.store), \
             patch("aria.cli.OllamaBackend", return_value=FakeBackend()), \
             patch("builtins.input", side_effect=["/exit"]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--session", "example"]), 0)


class VoiceTests(unittest.TestCase):
    def test_missing_model_has_clear_error(self):
        with self.assertRaisesRegex(VoiceError, "Vosk speech model not found"):
            LocalVoice(Path("missing-vosk-model"))

    def test_offline_voice_reads_one_utterance_and_speaks(self):
        class InputStream:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]

            def __enter__(self):
                self.callback(b"audio", 5, None, None)
                return self

            def __exit__(self, *_args):
                pass

        recognizer = MagicMock()
        recognizer.AcceptWaveform.return_value = True
        recognizer.Result.return_value = '{"text":"hello aria"}'
        speaker = MagicMock()
        modules = {
            "sounddevice": SimpleNamespace(query_devices=lambda **_kwargs: {"default_samplerate": 16000},
                                           RawInputStream=InputStream),
            "vosk": SimpleNamespace(Model=lambda _path: object(),
                                    KaldiRecognizer=lambda _model, _rate: recognizer),
            "pyttsx3": SimpleNamespace(init=lambda: speaker),
        }
        with patch.object(Path, "is_dir", return_value=True), patch.dict(sys.modules, modules):
            voice = LocalVoice(Path("model"))
            self.assertEqual(voice.listen(), "hello aria")
            voice.say("hello back")
        speaker.say.assert_called_once_with("hello back")
        speaker.runAndWait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
