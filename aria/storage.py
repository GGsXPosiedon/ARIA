"""Local, human-readable conversation files."""

import json
import os
from pathlib import Path
import re
import tempfile


class StorageError(RuntimeError):
    """A conversation could not be read or saved."""


_NAME = re.compile(r"[A-Za-z0-9_-]{1,40}\Z")


class SessionStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def _path(self, name: str) -> Path:
        if not _NAME.fullmatch(name):
            raise StorageError("Session name must use 1–40 letters, numbers, _ or -.")
        return self.directory / f"{name}.json"

    def names(self) -> list[str]:
        try:
            return sorted(path.stem for path in self.directory.glob("*.json")
                          if _NAME.fullmatch(path.stem))
        except OSError as exc:
            raise StorageError(f"Cannot list conversations: {exc}") from exc

    def load(self, name: str) -> list[dict[str, str]]:
        path = self._path(name)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise StorageError(f"Cannot read conversation '{name}': {exc}") from exc
        messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(messages, list) or len(messages) % 2:
            raise StorageError(f"Conversation '{name}' has an invalid format.")
        for index, message in enumerate(messages):
            role = "user" if index % 2 == 0 else "assistant"
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] != role or not isinstance(message["content"], str)
                    or not message["content"].strip()):
                raise StorageError(f"Conversation '{name}' has an invalid format.")
        return messages

    def save(self, name: str, messages: list[dict[str, str]]) -> None:
        path = self._path(name)
        temp_path = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             prefix=".aria-", suffix=".tmp", dir=self.directory,
                                             delete=False) as handle:
                temp_path = Path(handle.name)
                json.dump({"version": 1, "messages": messages}, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except OSError as exc:
            raise StorageError(f"Cannot save conversation '{name}': {exc}") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
