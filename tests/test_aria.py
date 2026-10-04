import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from aria.assistant import Assistant
from aria.backend import BackendError, OllamaBackend
from aria.cli import main
from aria.config import Config, ConfigurationError, load_dotenv


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.fail = False

    def complete(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise BackendError("temporary failure")
        return f"reply {len(self.calls)}"


class AssistantTests(unittest.TestCase):
    def test_history_is_bounded_and_resettable(self):
        backend = FakeBackend()
        assistant = Assistant(backend, history_turns=1)
        assistant.ask("first")
        assistant.ask("second")
        assistant.ask("third")
        self.assertEqual([item["role"] for item in backend.calls[2]],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(backend.calls[2][1]["content"], "second")
        self.assertEqual(len(assistant.history), 2)
        assistant.reset()
        self.assertEqual(assistant.history, [])

    def test_failed_request_does_not_change_history(self):
        backend = FakeBackend()
        assistant = Assistant(backend, history_turns=2)
        backend.fail = True
        with self.assertRaises(BackendError):
            assistant.ask("try")
        self.assertEqual(assistant.history, [])


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
    body = {"message": {"content": "Hello from fake model"}}
    models = [{"name": "test-model"}]

    def do_GET(self):
        self.requests.append(("GET", self.path, dict(self.headers), None))
        self._reply(200, {"models": self.models})

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        self.requests.append(("POST", self.path, dict(self.headers), json.loads(self.rfile.read(size))))
        self._reply(self.post_status, self.body)

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
        LocalHandler.body = {"message": {"content": "Hello from fake model"}}
        LocalHandler.models = [{"name": "test-model"}]
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.config = Config("test-model", self.url, 2, 10)

    def test_request_and_cli_once(self):
        with patch.dict(os.environ, {"ARIA_LOCAL_MODEL": "test-model", "ARIA_OLLAMA_URL": self.url}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(["--once", "Hi"])
        self.assertEqual(status, 0)
        self.assertIn("Hello from fake model", output.getvalue())
        self.assertEqual(LocalHandler.requests[0][0:2], ("GET", "/api/tags"))
        method, path, headers, body = LocalHandler.requests[1]
        self.assertEqual((method, path), ("POST", "/api/chat"))
        self.assertNotIn("Authorization", headers)
        self.assertEqual(body["model"], "test-model")
        self.assertFalse(body["stream"])
        self.assertEqual(body["messages"][-1], {"role": "user", "content": "Hi"})
        self.assertEqual(body["messages"][0]["role"], "system")

    def test_http_error_and_bad_response(self):
        backend = OllamaBackend(self.config)
        LocalHandler.post_status = 404
        LocalHandler.body = {"error": "model unavailable"}
        with self.assertRaisesRegex(BackendError, "HTTP 404.*model unavailable"):
            backend.complete([{"role": "user", "content": "Hi"}])
        LocalHandler.post_status = 200
        LocalHandler.body = {"message": {}}
        with self.assertRaisesRegex(BackendError, "no usable text"):
            backend.complete([{"role": "user", "content": "Hi"}])

    def test_uninstalled_model_is_not_sent_for_inference(self):
        LocalHandler.models = []
        with self.assertRaisesRegex(BackendError, "not installed"):
            OllamaBackend(self.config).complete([{"role": "user", "content": "Hi"}])
        self.assertEqual([item[0] for item in LocalHandler.requests], ["GET"])


if __name__ == "__main__":
    unittest.main()
