# ARIA v0.2 — local AI assistant

ARIA runs a downloaded language model on your own computer through Ollama. It needs no OpenAI account or API key. Version 0.2 adds saved conversations, replies that appear as they are generated, and optional offline voice input and speech output. Text mode needs only Python's standard library.

## Windows quick start

1. Install [Ollama for Windows](https://ollama.com/download/windows) and open it.
2. Download a local model in PowerShell (if you already did this for v0.1, skip it):

   ```powershell
   ollama pull qwen3:4b-instruct
   ```

3. Open PowerShell in this project folder and run:

   ```powershell
   python -m aria
   ```

Python 3.10 or newer is required. Type a message at `You >`; ARIA prints its answer as the model generates it. Conversations are saved automatically to `.aria_data/` in this project folder. The default conversation resumes when you run ARIA again.

If your PC struggles with the 4B model, `ollama pull gemma3:1b` and set `ARIA_LOCAL_MODEL=gemma3:1b` in `.env`.

## Conversations

At the ARIA prompt:

| Command | Action |
| --- | --- |
| `/sessions` | List saved conversations |
| `/new NAME` | Create a new conversation and switch to it |
| `/use NAME` | Resume a saved conversation |
| `/save` | Save the current conversation again |
| `/reset` | Clear the current conversation |
| `/help` | Show commands |
| `/exit` | Quit |

You can start directly in a named conversation with `python -m aria --session project`. The name can contain 1–40 letters, numbers, underscores, or hyphens. A one-off prompt also saves to the chosen session: `python -m aria --once "Hello, ARIA"`. ARIA keeps the full conversation on disk, but sends only the last `ARIA_HISTORY_TURNS` exchanges to Ollama for each reply. A reply interrupted before completion is not saved. The `.aria_data/` folder is ignored by Git; keep a separate backup if your conversations matter.

## Optional offline voice

Voice uses [Vosk](https://alphacephei.com/vosk/) for local speech recognition, your microphone through `sounddevice`, and Windows speech output through `pyttsx3`/SAPI5. The speech model and packages need an initial download; afterward the voice path runs locally. Text mode works without any voice packages.

1. Install the optional packages from this project folder:

   ```powershell
   python -m pip install -r requirements-voice.txt
   ```

2. Download `vosk-model-small-en-us-0.15` from the [official Vosk model list](https://alphacephei.com/vosk/models). Extract it so the folder is `models\vosk-model-small-en-us-0.15` under this project. The downloaded ZIP is about 40 MB. The `models/` folder is ignored by Git.
3. Connect a working microphone, then run:

   ```powershell
   python -m aria --voice
   ```

Press Enter to talk. ARIA listens for one utterance, prints what it heard, streams the reply to the screen, then speaks the completed reply. You can still type messages and commands. Set `ARIA_VOSK_MODEL_PATH` in `.env` if you put the Vosk model somewhere else. Voice recognition defaults to English; choose a different Vosk language model and path for another language. `/exit` quits. Spoken replies use the voices installed in Windows.

## Local-only settings

Copy `.env.example` to `.env` if you want to change defaults. Existing process environment variables override `.env`. The `.env` file is ignored by Git. ARIA reads only the local settings below and ignores old cloud API keys.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ARIA_LOCAL_MODEL` | `qwen3:4b-instruct` | Installed Ollama model |
| `ARIA_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama loopback address |
| `ARIA_LOCAL_TIMEOUT_SECONDS` | `120` | Ollama request timeout, max 300 |
| `ARIA_HISTORY_TURNS` | `10` | Prior exchanges sent to Ollama, 0–100 |
| `ARIA_VOSK_MODEL_PATH` | `models/vosk-model-small-en-us-0.15` | Local speech model folder |

ARIA accepts only a loopback Ollama address, checks that the selected model is installed locally, and sends no authentication header. Installing Ollama and downloading models need internet; inference and saved chats are local. To disable Ollama's optional cloud features too, set `OLLAMA_NO_CLOUD=1` in your Windows user environment and restart Ollama, as described in [Ollama's FAQ](https://docs.ollama.com/faq).

## Project structure

```text
aria/
  __main__.py    Entry point for python -m aria
  cli.py         Text/voice commands and streaming display
  config.py      Local configuration and .env loading
  backend.py     Ollama model check and streaming chat client
  assistant.py   Conversation context and reply handling
  storage.py     Local JSON conversations and safe saves
  voice.py       Optional offline microphone and speech output
  prompt.py      ARIA personality/system prompt
tests/
  test_aria.py   Offline tests with a fake local Ollama server
.env.example     Optional settings template
requirements.txt          Text mode has no extra packages
requirements-voice.txt    Optional voice packages
```

## Test and troubleshoot

Run the offline tests from this folder:

```powershell
python -m unittest discover -s tests -v
```

If ARIA cannot reach Ollama, open the Ollama app and retry. If it reports a missing text model, run the suggested `ollama pull` command. If voice reports a missing model, verify the extracted folder and `ARIA_VOSK_MODEL_PATH`. If voice reports a microphone error, check that Windows allows microphone access and that the right input device is selected. The Vosk model and speech packages are separate from the Ollama text model.

This setup runs a **pretrained model downloaded to your PC**. Training a language model from scratch is a separate project.
