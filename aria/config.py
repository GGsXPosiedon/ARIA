"""Environment-based configuration with optional local .env loading."""

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """A required setting is missing or invalid."""


LOCAL_SETTINGS = {"ARIA_LOCAL_MODEL", "ARIA_OLLAMA_URL", "ARIA_LOCAL_TIMEOUT_SECONDS", "ARIA_HISTORY_TURNS", "ARIA_VOSK_MODEL_PATH"}


def load_dotenv(path: Path) -> None:
    """Load local ARIA settings without overriding existing environment values.

    This deliberately supports a small, predictable subset of dotenv syntax.
    """
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"Cannot read {path}: {exc}") from exc
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigurationError(f"Invalid .env line {number}: expected KEY=value")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key.isidentifier():
            raise ConfigurationError(f"Invalid .env key on line {number}")
        if key not in LOCAL_SETTINGS:
            continue
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ConfigurationError(f"Unclosed quote on .env line {number}")
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Config:
    model: str
    ollama_url: str
    timeout_seconds: float
    history_turns: int

    @classmethod
    def from_env(cls) -> "Config":
        model = os.getenv("ARIA_LOCAL_MODEL", "qwen3:4b-instruct").strip()
        ollama_url = os.getenv("ARIA_OLLAMA_URL", "http://127.0.0.1:11434").strip().rstrip("/")
        if not model:
            raise ConfigurationError("ARIA_LOCAL_MODEL must not be empty.")
        if model.endswith(":cloud"):
            raise ConfigurationError("ARIA_LOCAL_MODEL must name a downloaded local model.")
        try:
            parsed = urlsplit(ollama_url)
            hostname = parsed.hostname
        except ValueError as exc:
            raise ConfigurationError("ARIA_OLLAMA_URL is not a valid local HTTP address.") from exc
        if (parsed.scheme != "http" or hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ConfigurationError("ARIA_OLLAMA_URL must be a local HTTP address, such as http://127.0.0.1:11434.")
        try:
            if parsed.port is None:
                raise ValueError("missing port")
        except ValueError as exc:
            raise ConfigurationError("ARIA_OLLAMA_URL needs a valid port.") from exc
        try:
            timeout = float(os.getenv("ARIA_LOCAL_TIMEOUT_SECONDS", "120"))
            turns = int(os.getenv("ARIA_HISTORY_TURNS", "10"))
        except ValueError as exc:
            raise ConfigurationError("ARIA_LOCAL_TIMEOUT_SECONDS and ARIA_HISTORY_TURNS must be numbers.") from exc
        if not 0 < timeout <= 300:
            raise ConfigurationError("ARIA_LOCAL_TIMEOUT_SECONDS must be between 0 and 300.")
        if not 0 <= turns <= 100:
            raise ConfigurationError("ARIA_HISTORY_TURNS must be between 0 and 100.")
        return cls(model, ollama_url, timeout, turns)
