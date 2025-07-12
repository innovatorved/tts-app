# TTS: Text/PDF/Conversation to Audio Converter

**Source Code:** [https://github.com/innovatorved/tts-app](https://github.com/innovatorved/tts-app)

This application converts text or PDF documents into speech using the Kokoro-TTS engine.
It can process direct text input, extract text from PDF files for narration, or convert conversations with different voices for each speaker.

## Features

- Convert direct text to audio.
- Convert PDF document content to audio.
- Convert conversations with different voices for male and female speakers.
- Customizable language, voice, and speech speed.
- Outputs audio in WAV format, split into manageable segments.
- Command-line interface for easy operation.
- Supports various languages provided by Kokoro-TTS.
- Option to specify compute device (CPU, CUDA, MPS).

## Prerequisites

1.  **Python:** Version 3.8 or higher.
2.  **espeak-ng:** This is a critical dependency for Kokoro-TTS.
    *   **Linux (Debian/Ubuntu):**
        ```bash
        sudo apt-get update
        sudo apt-get install espeak-ng
        ```
    *   **Linux (Fedora):**
        ```bash
        sudo dnf install espeak-ng
        ```
    *   **MacOS (using Homebrew):**
        ```bash
        brew install espeak-ng
        ```
    *   **Windows:**
        1.  Go to [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases).
        2.  Click on **Latest release**.
        3.  Download the appropriate `*.msi` file (e.g., `espeak-ng-X.Y.Z-x64.msi`).
        4.  Run the downloaded installer.
        5.  Ensure the installation directory (e.g., `C:\Program Files\eSpeak NG`) is added to your system's PATH environment variable if the installer doesn't do it automatically.

3.  **PyTorch:** Kokoro-TTS relies on PyTorch. Installation instructions can be found at [pytorch.org](https://pytorch.org/get-started/locally/). The `requirements.txt` includes `torch`.

4.  **FFmpeg (for audio merging with `pydub`):**
    `pydub` relies on FFmpeg (or libav) for audio processing. Ensure it's installed and in your system's PATH.
    *   **Linux (Debian/Ubuntu):** `sudo apt-get install ffmpeg`
    *   **Linux (Fedora):** `sudo dnf install ffmpeg`
    *   **MacOS (using Homebrew):** `brew install ffmpeg`
    *   **Windows:** Download FFmpeg static builds from [ffmpeg.org]

## Setup

1.  **Clone the repository (or create the files as described):**
    ```bash
    git clone https://github.com/innovatorved/tts-app.git
    cd tts-app
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On MacOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    For specific non-English languages, you might need to install additional `misaki` components. For example, for Japanese:
    ```bash
    pip install misaki[ja]
    ```
    And for Mandarin Chinese:
    ```bash
    pip install misaki[zh]
    ```
    Update `requirements.txt` accordingly if you plan to use these languages frequently.

## Usage

The application is run from the command line from within the `kokoro_tts_app` directory (or the root `tts-app` directory if you `cd` into it first).

```bash
python main.py [OPTIONS]
```
(If your `main.py` is inside `kokoro_tts_app`, and you are in `tts-app`, you'd run `python kokoro_tts_app/main.py [OPTIONS]`)

**Basic Examples:**

1.  **Convert direct text to audio (American English, default voice):**
    ```bash
    python main.py --text "Hello world. This is a test of Kokoro TTS." --output_dir "my_audio"
    ```

2.  **Convert a PDF file to audio (American English, default voice, verbose output):**
    ```bash
    python main.py --pdf "path/to/your/document.pdf" --output_dir "story_audio" -v
    ```

**Command-line Options:**

*   **Input Source (choose one):**
    *   `--text "YOUR TEXT"`: Direct text to convert to speech.
        *   Example: `python main.py --text "The quick brown fox jumps over the lazy dog."`
    *   `--pdf "PATH_TO_PDF"`: Path to a PDF file to convert to speech.
        *   Example: `python main.py --pdf "report.pdf"`
    *   `--text_file "PATH_TO_TXT"`: Path to a text file to convert to speech.
        *   Example: `python main.py --text_file "notes.txt"`
    *   `--conversation "PATH_TO_CONVERSATION_TXT"`: Path to a text file containing conversation with speaker labels.
        *   Format: Text file with lines starting with "Man:" or "Woman:" to indicate speakers.
        *   Example: `python main.py --conversation "conversation.txt" --merge_output`
        *   Speaker lines should be formatted as "Man: Hello there!" or "Woman: Nice to meet you."
    *   **Note:** You can only specify one of `--text`, `--pdf`, `--text_file`, or `--conversation` at a time.

*   **Output Configuration:**
    *   `--output_dir "DIRECTORY_PATH"`: Directory to save the generated audio files.
        *   Default: `./output_audio`
        *   Example: `python main.py --text "..." --output_dir "project_audio_files"`
    *   `--output_filename_base "BASE_NAME"`: Base name for output audio files. If not provided, it's derived from the PDF name or a timestamp for text input.
        *   Example: `python main.py --pdf "chapter1.pdf" --output_filename_base "chapter1_audio"`
    *   `--merge_output`: A flag. If present, all generated audio segments will be merged into a single WAV file named `[output_filename_base]_merged.wav`.
        *   Requires `pydub` and FFmpeg.
        *   Example: `python main.py --pdf "story.pdf" --merge_output`

*   **TTS Engine Configuration:**
    *   `--lang "LANG_CODE"`: Language code for Kokoro-TTS.
        *   Default: `a` (American English)
        *   Supported codes (ensure corresponding `misaki` is installed if needed):
            *   `a`: American English
            *   `b`: British English
            *   `e`: Spanish (es)
            *   `f`: French (fr-fr)
            *   `h`: Hindi (hi)
            *   `i`: Italian (it)
            *   `j`: Japanese (requires `pip install misaki[ja]`)
            *   `p`: Brazilian Portuguese (pt-br)
            *   `z`: Mandarin Chinese (requires `pip install misaki[zh]`)
        *   Example (Japanese): `python main.py --text "こんにちは" --lang "j"`
    *   `--voice "VOICE_MODEL"`: Voice model to use for non-conversation input.
        *   Default: `af_heart` (an American English female voice)
        *   Refer to [Kokoro-82M Hugging Face SAMPLES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/SAMPLES.md) for available voices and their corresponding language codes.
        *   Example (British English male voice, *name is illustrative, check link for actual names*): `python main.py --text "Good day!" --lang "b" --voice "bm_somevoicename"`
    *   `--male_voice "VOICE_MODEL"`: Voice model to use for male speakers in conversation mode.
        *   Default: `am_adam` (an American English male voice - strong and confident)
        *   Other options: `am_michael` (warm and trustworthy), `am_echo` (resonant and clear), `am_eric` (professional and authoritative)
        *   Example: `python main.py --conversation "dialog.txt" --male_voice "am_michael"`
    *   `--female_voice "VOICE_MODEL"`: Voice model to use for female speakers in conversation mode.
        *   Default: `af_heart` (an American English female voice)
        *   Example: `python main.py --conversation "dialog.txt" --female_voice "af_soft"`
    *   `--speed SPEED_MULTIPLIER`: Speech speed. `1.0` is normal.
        *   Default: `1.0`
        *   Example (slower): `python main.py --text "..." --speed 0.8`
        *   Example (faster): `python main.py --text "..." --speed 1.2`
    *   `--split_pattern "REGEX_PATTERN"`: Regex pattern to split long text into manageable chunks for TTS processing.
        *   Default: `r'\n\n+|\r\n\r\n+|\n\s*\n+|[.!?]\s'` (splits by double newlines or sentence-ending punctuation followed by space)
        *   Example (split only by periods): `python main.py --text "Sentence one. Sentence two." --split_pattern r'\.\s'`

*   **System Configuration:**
    *   `--device "DEVICE_NAME"`: Specify the compute device for the model.
        *   Choices: `cpu`, `cuda`, `mps`
        *   Default: Kokoro auto-detects (usually CPU if no GPU is configured/available).
        *   Example (CPU): `python main.py --text "..." --device "cpu"`
        *   Example (NVIDIA GPU): `python main.py --text "..." --device "cuda"`
        *   Example (Apple Silicon GPU - **Important Note Below**):
            ```bash
            # For MPS, set this environment variable *before* running the script:
            PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py --text "..." --device "mps"
            ```
            The `--device "mps"` flag tells the script to attempt using MPS. The environment variable helps PyTorch manage operations on MPS.

*   **Threading & Chunking (for PDF/TXT input):**
    *   `--threads NUM_THREADS`: Number of worker threads for processing PDF/TXT chunks.
        *   Default: `1` (sequential processing).
        *   Recommended Max: `2-4` due to single TTS engine serialization.
        *   Example: `python main.py --txt "large_novel.txt" --threads 2`
    *   `--paragraphs_per_chunk NUM_PARAGRAPHS`: Number of paragraphs to group into a single chunk when processing PDFs/TXTs with threading.
        *   Default: `30`
        *   Example: `python main.py --pdf "large_doc.pdf" --threads 2 --paragraphs_per_chunk 50`

*   **Other:**
    *   `--verbose` or `-v`: Enable detailed debug logging for troubleshooting.
        *   Example: `python main.py --pdf "mydoc.pdf" -v`

**Advanced Example (Story Telling from PDF in British English):**

```bash
# Ensure you have a PDF named 'my_novel.pdf'
# This example uses a hypothetical British English voice 'bf_storyteller'.
# Check Kokoro-82M SAMPLES.md for actual available voices.
python main.py \
    --pdf "my_novel.pdf" \
    --lang "b" \
    --voice "bf_storyteller" \
    --speed 0.95 \
    --output_dir "audiobooks/my_novel" \
    --output_filename_base "my_novel_chapter" \
    --verbose
```

**Conversation Example (Dialog between Man and Woman):**

```bash
# Ensure you have a conversation text file formatted with "Man:" and "Woman:" prefixes
# This example uses different voices for each speaker and merges the output
python main.py \
    --conversation "example/conversation.txt" \
    --speed 1.0 \
    --output_dir "dialogs/conversation1" \
    --merge_output \
    --verbose
```

## Troubleshooting

*   **`espeak-ng` not found / `RuntimeError: espeak`:** Ensure `espeak-ng` is correctly installed and accessible in your system's PATH. Re-check the installation steps for your OS.
*   **Language/Voice Mismatches:** Ensure the `lang_code` and `voice` are compatible. Consult the [Kokoro-82M SAMPLES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/SAMPLES.md) on Hugging Face.
*   **PDF Parsing Issues:** If a PDF yields no text or garbled text, it might be an image-based PDF or have complex formatting. This tool uses PyPDF2, which is best for text-based PDFs. For scanned PDFs, you'll need an OCR (Optical Character Recognition) step before using this tool.
*   **Memory Issues for very long texts:** The `split_pattern` is crucial for breaking down long texts. If you encounter memory issues, try a more aggressive splitting pattern (e.g., splitting more frequently) or process the document in smaller sections manually.
*   **`ModuleNotFoundError` or import errors:** Ensure you have activated your virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`) and installed all packages from `requirements.txt`. If running from the root `tts-app` directory, use `python kokoro_tts_app/main.py ...`.
