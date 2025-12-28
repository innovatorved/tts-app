# TTS: Scalable and Reliable Text-to-Speech Converter

**Source Code:** [https://github.com/innovatorved/tts-app](https://github.com/innovatorved/tts-app)

This application converts text, PDFs, or conversations into speech using Kokoro-TTS or Chatterbox TTS engines. It is designed for scalability and reliability, capable of processing very large files like entire books, with progress tracking and job resumption capabilities.

## Key Features

- **Scalable & Reliable:** Built with a job-based architecture to handle large files. It can process an entire book without memory issues.
- **Parallel Processing:** Utilizes multiple CPU processes to convert text to speech, significantly speeding up large jobs.
- **Job-Based System:** Every conversion is a "job" tracked in a local SQLite database. This means the state is saved, and you can manage multiple conversions.
- **Progress Tracking:** Monitor the real-time progress of any running job using the command line.
- **Fault-Tolerant:** Resume failed or interrupted jobs exactly from where they left off. No more lost work!
- **Web UI with Job Dashboard:** A modern web interface for easily creating jobs and a dashboard to view the status of all past and present jobs.
- **Multiple Input Formats:** Convert direct text, PDF documents, or plain text files.
- **Choice of Engines:** Use Kokoro (multilingual) for versatility or Chatterbox (English) for high-quality, expressive speech.
- **Customizable Output:** Control language, voice, and speed. Final audio can be automatically merged into a single file.

## Architecture Overview

The application's resilience and scalability come from its core architecture, which revolves around a central database and parallel workers.

1.  **SQLite Database (`tts_jobs.db`):** This is the heart of the system. It contains two main tables:
    *   `jobs`: Stores high-level information about each conversion task (job name, input/output paths, TTS settings, overall status).
    *   `chunks`: The input text for each job is divided into smaller text "chunks," and each chunk gets a row in this table. This allows for fine-grained tracking and processing.

2.  **Main Script (`main.py` / `webui.py`):** This is the user's entry point. When a new job is created:
    *   It parses the input file (e.g., a PDF) to extract the raw text.
    *   It splits the raw text into larger chunks (based on paragraph count).
    *   It creates a new record in the `jobs` table and populates the `chunks` table with all the text chunks, marking them as `pending`.

3.  **Process Pool & Workers (`worker.py`):**
    *   The main script launches a pool of worker processes (`ProcessPoolExecutor`). The number of workers is configurable.
    *   Each worker runs the `process_chunk_worker` function. It connects to the database and enters a loop.
    *   In the loop, it "claims" a `pending` chunk, atomically updating its status to `processing`. This ensures no two workers grab the same chunk.
    *   The worker then performs the TTS conversion for that single chunk and saves the resulting audio file.
    *   Finally, it updates the chunk's status to `completed` or `failed`. The worker then loops to claim the next available chunk.

4.  **TTS Engines (`tts_engine/`):** The actual text-to-speech conversion is handled by processor classes that wrap the underlying TTS libraries (Kokoro or Chatterbox). These are initialized within each worker process.

This design means that even if the application is stopped, the state of the job is preserved in the database. You can simply resume the job, and the workers will pick up right where they left off, processing any `pending` or `failed` chunks.

## Prerequisites

1.  **Python:** Version 3.8 or higher.
2.  **espeak-ng:** A critical dependency for the Kokoro-TTS engine.
    *   **Linux (Debian/Ubuntu):** `sudo apt-get update && sudo apt-get install espeak-ng`
    *   **MacOS (using Homebrew):** `brew install espeak-ng`
    *   **Windows:** Download and install from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases).
3.  **FFmpeg (for audio merging):**
    *   **Linux (Debian/Ubuntu):** `sudo apt-get install ffmpeg`
    *   **MacOS (using Homebrew):** `brew install ffmpeg`
    *   **Windows:** Download static builds from [ffmpeg.org](https://ffmpeg.org/download.html).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/innovatorved/tts-app.git
    cd tts-app
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```

3.  **Install build tools and dependencies:**
    First, ensure your `pip`, `setuptools`, and `wheel` are up to date, which can prevent common installation issues:
    ```bash
    pip install --upgrade pip setuptools wheel
    ```
    Then, install the application's dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Web UI

The application includes a modern web interface built with Gradio.

### Running the Web UI

```bash
python webui.py
```

The web UI will be available at `http://localhost:7860` and includes:
- **Create Job Tab:** A simplified interface to create new TTS jobs from text or uploaded files.
- **Job Dashboard Tab:** View the status of all jobs (`completed`, `processing`, `failed`), see when they were created, and refresh the list.

## Usage (Command Line)

The job-based system is powerful and easy to use. Here’s how to work with it.

### 1. Creating a Job

To start a new conversion, you create a job. For example, to convert a PDF book:

```bash
python main.py --pdf "path/to/your/book.pdf" --job-name "my-book-job" --num-workers 4 --merge_output
```

- `--pdf "path/to/your/book.pdf"`: Specifies the input file.
- `--job-name "my-book-job"`: Gives the job a unique name for tracking. If you don't provide one, a name will be generated.
- `--num-workers 4`: This is the number of parallel processes to use. A good starting point is the number of cores in your CPU.
- `--merge_output`: This flag tells the app to automatically stitch the final audio chunks into a single WAV file.

The script will create the job, add it to the database, and start processing.

### 2. Monitoring a Job

While a job is running, you can open a **second terminal window** and monitor its progress:

```bash
python main.py --monitor --job-name "my-book-job"
```

You will see a live progress bar:
`Progress for job 'my-book-job': |█████████████------------------| 45.24% (19/42 Chunks)`

### 3. Resuming a Failed Job

If a job is interrupted or fails for any reason, you can easily resume it. The application will find all unprocessed or failed text chunks and restart the work.

```bash
python main.py --resume --job-name "my-book-job" --num-workers 4
```

## Command-Line Options

### Job Control
- `--job-name <name>`: Assign a unique name to a job for tracking and resuming.
- `--resume`: Resume a failed or interrupted job specified by `--job-name`.
- `--monitor`: Monitor the progress of a job specified by `--job-name`.
- `--num-workers <int>`: Number of parallel worker processes to use. Defaults to the number of CPU cores.

### Input Source (choose one for a new job)
- `--text "YOUR TEXT"`: A string of text to convert.
- `--pdf "PATH_TO_PDF"`: Path to a PDF file.
- `--text_file "PATH_TO_TXT"`: Path to a plain text file.
- `--conversation "PATH_TO_CONV_TXT"`: Path to a conversation file (Note: special conversation parsing is not yet fully integrated into the job system).

### Output & TTS Configuration
- `--output_dir "path"`: Directory to save audio files (default: `./output_audio`).
- `--merge_output`: If present, merges all audio chunks into a single file.
- `--engine {kokoro,chatterbox}`: Choose the TTS engine.
- `--device {cpu,cuda,mps}`: Specify the compute device for the model.
- `--paragraphs_per_chunk <int>`: Number of paragraphs to group into a single processing chunk (default: 10).

### Kokoro Engine Options
- `--lang "code"`: Language code (e.g., 'a' for American English).
- `--voice "name"`: Voice model name (e.g., 'af_heart').
- `--speed <float>`: Speech speed multiplier (e.g., 1.0 is normal).

### Chatterbox Engine Options
- `--cb_voice_cloning`: Enable voice cloning mode. When enabled, requires `--cb_audio_prompt`. When disabled (default), Chatterbox generates speech with its default voice.
- `--cb_audio_prompt <path.wav>`: Path to a reference audio file for voice cloning (requires `--cb_voice_cloning`).

- `--cb_temperature <float>`: Sampling temperature (default: 0.8).
- `--cb_top_p <float>`: Nucleus sampling `p` value (default: 1.0).
- `--cb_repetition_penalty <float>`: Penalty for repeating tokens (default: 1.2).

### Other
- `--verbose` or `-v`: Enable detailed debug logging.

## Troubleshooting

*   **Dependency Issues:** If you encounter errors during `pip install -r requirements.txt`, first ensure you have run `pip install --upgrade pip setuptools wheel`.
*   **`espeak-ng` not found:** Ensure `espeak-ng` is correctly installed and in your system's PATH.
*   **`ffmpeg` not found:** The `--merge_output` feature requires FFmpeg. Ensure it is installed and accessible in your system's PATH.
*   **PDF Parsing Issues:** The app uses `PyPDF2` and works best with text-based PDFs. Image-based (scanned) PDFs will not work.
*   **Chatterbox Issues:** The Chatterbox engine requires the `chatterbox-tts` package. If you only plan to use Kokoro, this is not needed. To use Chatterbox, run `pip install chatterbox-tts`.
