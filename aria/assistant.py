"""Session conversation state and context management."""

from .backend import OllamaBackend
from .prompt import SYSTEM_PROMPT


class Assistant:
    def __init__(self, backend: OllamaBackend, history_turns: int):
        self.backend = backend
        self.history_turns = history_turns
        self.history: list[dict[str, str]] = []

    def ask(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("Message cannot be empty.")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.history_turns:
            messages.extend(self.history[-2 * self.history_turns :])
        messages.append({"role": "user", "content": text})
        answer = self.backend.complete(messages)
        # Commit only successful exchanges; failed requests can be retried cleanly.
        self.history.extend(({"role": "user", "content": text}, {"role": "assistant", "content": answer}))
        if self.history_turns:
            self.history = self.history[-2 * self.history_turns :]
        else:
            self.history.clear()
        return answer

    def reset(self) -> None:
        self.history.clear()
