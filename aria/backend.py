"""Local-only Ollama client using Python's standard library."""

import json
from urllib import error, request

from .config import Config


class BackendError(RuntimeError):
    """A local model request failed or returned an unusable response."""


class OllamaBackend:
    def __init__(self, config: Config):
        self.config = config
        self._model_checked = False

    def complete(self, messages: list[dict[str, str]]) -> str:
        return "".join(self.stream(messages)).strip()

    def stream(self, messages: list[dict[str, str]]):
        if not self._model_checked:
            self._check_local_model()
        payload = json.dumps({
            "model": self.config.model,
            "messages": messages,
            "stream": True,
        }).encode("utf-8")
        req = request.Request(
            f"{self.config.ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                saw_done = False
                saw_text = False
                for line in response:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            raise ValueError("invalid chunk")
                        if item.get("error"):
                            raise BackendError(f"Ollama error: {str(item['error'])[:300]}")
                        content = item.get("message", {}).get("content", "")
                        if not isinstance(content, str):
                            raise ValueError("invalid content")
                        if content:
                            saw_text = True
                            yield content
                        if item.get("done") is True:
                            saw_done = True
                            break
                    except (ValueError, TypeError, AttributeError) as exc:
                        raise BackendError("Ollama returned an invalid streaming response.") from exc
                if not saw_done or not saw_text:
                    raise BackendError("Ollama ended the reply before completing it.")
        except error.HTTPError as exc:
            raise BackendError(f"Ollama HTTP {exc.code}{_http_detail(exc)}") from exc
        except (error.URLError, ConnectionError) as exc:
            raise BackendError("Cannot reach Ollama on this computer. Start Ollama and try again.") from exc
        except TimeoutError as exc:
            raise BackendError("The local model timed out. Try a smaller model or raise ARIA_LOCAL_TIMEOUT_SECONDS.") from exc
        except (OSError, UnicodeError) as exc:
            raise BackendError("Ollama's reply was interrupted.") from exc

    def _check_local_model(self) -> None:
        req = request.Request(f"{self.config.ollama_url}/api/tags", method="GET")
        data = self._get_json(req)
        try:
            names = {item["name"] for item in data["models"]}
        except (KeyError, TypeError) as exc:
            raise BackendError("Ollama returned an invalid local model list.") from exc
        if self.config.model not in names:
            raise BackendError(
                f"Local model '{self.config.model}' is not installed. "
                f"Run: ollama pull {self.config.model}"
            )
        self._model_checked = True

    def _get_json(self, req: request.Request) -> dict:
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                data = json.load(response)
        except error.HTTPError as exc:
            raise BackendError(f"Ollama HTTP {exc.code}{_http_detail(exc)}") from exc
        except (error.URLError, ConnectionError) as exc:
            raise BackendError("Cannot reach Ollama on this computer. Start Ollama and try again.") from exc
        except TimeoutError as exc:
            raise BackendError("The local model timed out. Try a smaller model or raise ARIA_LOCAL_TIMEOUT_SECONDS.") from exc
        except (ValueError, UnicodeError, OSError) as exc:
            raise BackendError("Ollama returned an invalid response.") from exc
        if not isinstance(data, dict):
            raise BackendError("Ollama returned an invalid response.")
        return data


def _http_detail(exc: error.HTTPError) -> str:
    try:
        data = json.loads(exc.read(4096))
        message = data.get("error", "")
        if isinstance(message, str) and message:
            return f": {message[:300]}"
    except (ValueError, OSError, AttributeError, TypeError):
        pass
    return ". Check that Ollama is running and the model is installed."
