"""Interactive command-line interface."""

import argparse
from pathlib import Path
import sys

from . import __version__
from .assistant import Assistant
from .backend import BackendError, OllamaBackend
from .config import Config, ConfigurationError, load_dotenv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARIA v0.1 text assistant")
    parser.add_argument("--once", metavar="MESSAGE", help="Ask one question and exit")
    parser.add_argument("--version", action="version", version=f"ARIA {__version__}")
    args = parser.parse_args(argv)
    try:
        load_dotenv(Path.cwd() / ".env")
        config = Config.from_env()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    assistant = Assistant(OllamaBackend(config), config.history_turns)
    if args.once is not None:
        try:
            print(assistant.ask(args.once))
            return 0
        except (BackendError, ValueError) as exc:
            print(f"ARIA error: {exc}", file=sys.stderr)
            return 1

    print(f"ARIA v{__version__} · {config.model} · type /help for commands")
    while True:
        try:
            text = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not text:
            continue
        command = text.lower()
        if command in ("/exit", "/quit"):
            print("Goodbye.")
            return 0
        if command == "/reset":
            assistant.reset()
            print("ARIA > Conversation cleared.")
            continue
        if command == "/help":
            print("Commands: /help, /reset, /exit")
            continue
        try:
            print(f"ARIA > {assistant.ask(text)}")
        except BackendError as exc:
            print(f"ARIA error: {exc}", file=sys.stderr)
