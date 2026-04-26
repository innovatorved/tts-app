# TTS app

Text/PDF → speech with **Kokoro** or **Chatterbox**. Jobs are chunked, SQLite-backed, resumable. [github.com/innovatorved/tts-app](https://github.com/innovatorved/tts-app)

## System packages

- **espeak-ng** (Kokoro): `brew install espeak-ng` (macOS) or `apt install espeak-ng` (Linux)
- **ffmpeg** (merge): `brew install ffmpeg` or `apt install ffmpeg`

## Install (use Conda — avoids spaCy/blis build failures on Python 3.14)

Chatterbox pulls **spaCy → thinc → blis**. On **Python 3.14**, `pip` often **builds blis from source** and the compile fails. Use **Python 3.11** via Conda:

```bash
conda env create -f environment.yml
conda activate tts_app
```

Dependencies are listed in **`requirements.txt`**; Conda installs them into this env.

**Optional:** `pip-audit` is included in `requirements.txt`. GPU: reinstall `torch` / `torchaudio` from [pytorch.org](https://pytorch.org/get-started/locally/) if you need CUDA wheels.

### Pip-only (Kokoro-focused, advanced)

If you insist on a venv, use **Python 3.11** (e.g. `pyenv install 3.11.9` then `python3.11 -m venv .venv`). Do **not** use 3.14 for the full `requirements.txt` with Chatterbox.

## Run

**Web UI:** `python webui.py` → http://localhost:7860  

**CLI:**

```bash
python main.py --pdf book.pdf --job-name myjob --num-workers 4 --merge_output --device cpu
python main.py --monitor --job-name myjob
python main.py --resume --job-name myjob --num-workers 4
```

**Engines:** `--engine kokoro` (default) or `--engine chatterbox` (one worker enforced).

## Tests

```bash
conda activate tts_app   # or your 3.11 venv
pytest tests/ -q
```

## CPU notes

Prefer **Kokoro** on CPU. Balance `--num-workers` and `--max-torch-threads` so you do not oversubscribe cores. **Chatterbox** is slow on CPU.

## Troubleshooting

- **`blis` / `spacy` compile errors:** you are on too new a Python (e.g. 3.14). Recreate the env with **`environment.yml`** (3.11).
- **`espeak-ng` / `ffmpeg`:** must be on `PATH`.
- **PDFs:** text-based only (`pypdf`).

`python main.py --help` for all flags.
