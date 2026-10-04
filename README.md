# ARIA v0.1 — fully local

ARIA is a text assistant that runs a downloaded language model on your own
computer through Ollama. No OpenAI account, cloud model, or API key is needed.
Once Ollama and the model are downloaded, chat works without internet. ARIA
has no voice, GUI, vision, computer-control tools, or persistent memory.

## Windows quick start

1. Install [Ollama for Windows](https://ollama.com/download/windows) and open it.
   Ollama runs in the background and serves the local model at
   `http://127.0.0.1:11434`.
2. In a fresh PowerShell window, download the recommended text model:

   ```powershell
   ollama pull qwen3:4b-instruct
   ```

   The model download is about 2.5 GB. If your PC has limited RAM or runs it
   slowly, use `ollama pull gemma3:1b` instead and set
   `ARIA_LOCAL_MODEL=gemma3:1b` in `.env` as described below.
3. In PowerShell, enter this ARIA project directory and run:

   ```powershell
   python -m aria
   ```

   Type a message at `You >`. `/reset` clears this session's context;
   `/exit` quits. For a single question, use
   `python -m aria --once "Hello, ARIA"`.

Python 3.10 or newer is required. The Python project has no third-party
dependencies; `requirements.txt` records this. You do not need to create a
virtual environment or install Python packages to run it from this directory.

## Local-only behavior

ARIA accepts only loopback addresses for Ollama. Before each session's first
chat request, it asks Ollama for the list of installed local models and refuses
to continue if the chosen model is absent. It sends no authentication header.
The model runs on your computer; initial installation and model download need
internet. To also disable Ollama's optional cloud features, set
`OLLAMA_NO_CLOUD=1` in your Windows user environment and restart Ollama, as
described in [Ollama's FAQ](https://docs.ollama.com/faq). ARIA never uses
Ollama cloud models.

This setup uses a **pretrained model downloaded to your PC**. Training a new
language model from scratch is a separate, much larger project.

## Optional configuration

ARIA works without a `.env` file. To change the model or timeout, copy
`.env.example` to `.env` and edit it. Existing process environment variables
take precedence over `.env`. The parser accepts `KEY=value` lines, optional
quotes, and whole-line `#` comments. It does not expand variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ARIA_LOCAL_MODEL` | `qwen3:4b-instruct` | Installed Ollama model name |
| `ARIA_OLLAMA_URL` | `http://127.0.0.1:11434` | Local Ollama address only |
| `ARIA_LOCAL_TIMEOUT_SECONDS` | `120` | Request timeout, up to 300 seconds |
| `ARIA_HISTORY_TURNS` | `10` | Prior exchanges sent each time, from 0 to 100 |

An old `.env` from ARIA's previous API version may contain `ARIA_API_KEY`,
`OPENAI_API_KEY`, `ARIA_MODEL`, `ARIA_BASE_URL`, or `ARIA_TIMEOUT_SECONDS`. This version ignores those
settings. You may delete those lines; ARIA will never send requests to that
old cloud endpoint.

## Project structure

```text
aria/
  __main__.py    Entry point for python -m aria
  cli.py         Text prompt and commands
  config.py      Local-only configuration and .env loading
  backend.py     Ollama /api/tags and /api/chat client
  assistant.py   In-session conversation history
  prompt.py      ARIA personality/system prompt
tests/
  test_aria.py   Offline tests with a fake local Ollama server
.env.example     Optional settings template
requirements.txt
```

## Tests and troubleshooting

Run the offline tests from this directory:

```powershell
python -m unittest discover -s tests -v
```

If ARIA says it cannot reach Ollama, open the Ollama app and try again. If
it says the local model is not installed, run the displayed `ollama pull`
command. A smaller model can help if responses are too slow or time out.
