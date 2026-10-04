"""Interactive command-line interface for local text and optional voice chat."""

import argparse
from pathlib import Path
import sys

from . import __version__
from .assistant import Assistant
from .backend import BackendError, OllamaBackend
from .config import Config, ConfigurationError, load_dotenv
from .storage import SessionStore, StorageError
from .voice import LocalVoice, VoiceError, model_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARIA v0.2 local assistant")
    parser.add_argument("--once", metavar="MESSAGE", help="Ask one question and exit")
    parser.add_argument("--session", default="default", metavar="NAME", help="Saved conversation name")
    parser.add_argument("--voice", action="store_true", help="Use offline microphone and spoken replies")
    parser.add_argument("--version", action="version", version=f"ARIA {__version__}")
    args = parser.parse_args(argv)
    if args.once is not None and args.voice:
        parser.error("--once and --voice cannot be used together")
    try:
        load_dotenv(Path.cwd() / ".env")
        config = Config.from_env()
        store = SessionStore(Path.cwd() / ".aria_data")
        assistant = Assistant(OllamaBackend(config), config.history_turns)
        assistant.history = store.load(args.session)
        voice = LocalVoice(model_path()) if args.voice else None
    except (ConfigurationError, StorageError, VoiceError) as exc:
        print(f"ARIA setup error: {exc}", file=sys.stderr)
        return 2

    session = args.session
    if args.once is not None:
        try:
            answer = assistant.ask(args.once)
            print(answer)
            store.save(session, assistant.history)
            return 0
        except (BackendError, ValueError, StorageError) as exc:
            print(f"ARIA error: {exc}", file=sys.stderr)
            return 1

    mode = "voice" if voice else "text"
    print(f"ARIA v{__version__} · {config.model} · {mode} · session: {session} · /help for commands")
    while True:
        try:
            raw = input("Press Enter to talk, or type a message/command > " if voice else "You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not raw and voice:
            try:
                print("Listening...", flush=True)
                raw = voice.listen()
                print(f"You said > {raw}")
            except VoiceError as exc:
                print(f"ARIA voice error: {exc}", file=sys.stderr)
                continue
        if not raw:
            continue
        command, _, argument = raw.partition(" ")
        command = command.lower()
        argument = argument.strip()
        if command in ("/exit", "/quit"):
            print("Goodbye.")
            return 0
        if command == "/help":
            print("Commands: /help, /sessions, /use NAME, /new NAME, /save, /reset, /exit")
            continue
        if command == "/sessions":
            try:
                print("Saved conversations: " + (", ".join(store.names()) or "none yet"))
            except StorageError as exc:
                print(f"ARIA error: {exc}", file=sys.stderr)
            continue
        if command in ("/use", "/new"):
            try:
                if not argument:
                    raise StorageError(f"Usage: {command} NAME")
                # Validates the name before checking existence.
                path = store._path(argument)
                if command == "/use" and not path.is_file():
                    raise StorageError(f"Conversation '{argument}' does not exist. Use /new {argument}.")
                if command == "/new" and path.exists():
                    raise StorageError(f"Conversation '{argument}' already exists. Use /use {argument}.")
                history = store.load(argument) if command == "/use" else []
                if command == "/new":
                    store.save(argument, history)
                assistant.history = history
                session = argument
                print(f"ARIA > Switched to {session} ({len(history) // 2} exchanges).")
            except StorageError as exc:
                print(f"ARIA error: {exc}", file=sys.stderr)
            continue
        if command == "/save":
            try:
                store.save(session, assistant.history)
                print(f"ARIA > Saved {session}.")
            except StorageError as exc:
                print(f"ARIA error: {exc}", file=sys.stderr)
            continue
        if command == "/reset":
            try:
                store.save(session, [])
                assistant.reset()
                print("ARIA > Conversation cleared.")
            except StorageError as exc:
                print(f"ARIA error: {exc}", file=sys.stderr)
            continue
        if raw.startswith("/"):
            print("ARIA > Unknown command. Type /help.")
            continue
        try:
            print("ARIA > ", end="", flush=True)
            chunks = []
            for chunk in assistant.stream_reply(raw):
                chunks.append(chunk)
                print(chunk, end="", flush=True)
            print()
            store.save(session, assistant.history)
            if voice:
                voice.say("".join(chunks).strip())
        except (BackendError, ValueError, StorageError, VoiceError) as exc:
            print(f"\nARIA error: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nARIA > Reply interrupted. Message was not saved.")


if __name__ == "__main__":
    raise SystemExit(main())
