import gradio as gr
import os
import sys
import tempfile
import shutil
import threading
import time
from datetime import datetime

# Adjust path to ensure the app's top level directory (parent directory) is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from tts_engine.processor import KokoroTTSProcessor
from utils.pdf_parser import extract_text_from_pdf
from utils.text_file_parser import extract_text_from_txt
from utils.file_handler import ensure_dir_exists, get_safe_filename
from utils.audio_merger import merge_audio_files
import natsort



# --- TTS Processing Logic ---

def split_text_into_chunks(full_text: str, max_paragraphs_per_chunk: int = 30) -> list[str]:
    import re
    if not full_text or not full_text.strip():
        return []
    normalized_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n+", normalized_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        if normalized_text.strip():
            paragraphs = [normalized_text.strip()]
        else:
            return []
    chunks = []
    current_chunk_paragraphs = []
    for i, para in enumerate(paragraphs):
        current_chunk_paragraphs.append(para)
        if len(current_chunk_paragraphs) >= max_paragraphs_per_chunk or (i + 1) == len(paragraphs):
            chunks.append("\n\n".join(current_chunk_paragraphs))
            current_chunk_paragraphs = []
    if not chunks and full_text.strip():
        chunks = ["\n\n".join(paragraphs)]
    return chunks

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def process_input_file(file_obj, filetype):
    if filetype == "pdf":
        return extract_text_from_pdf(file_obj.name)
    elif filetype == "txt":
        return extract_text_from_txt(file_obj.name)
    else:
        return None

def get_base_filename(input_type, text=None, file_obj=None):
    if input_type == "text":
        return f"text_to_speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    elif input_type in ("pdf", "txt") and file_obj is not None:
        return os.path.splitext(os.path.basename(file_obj.name))[0]
    else:
        return f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def stream_tts_chunks(
    text_chunks,
    tts_processor,
    output_dir,
    base_filename,
    voice,
    speed,
    split_pattern,
):
    all_generated_files = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_base_filename = f"{base_filename}_chunk_{i:02d}"
        chunk_files = tts_processor.text_to_speech(
            text=chunk_text,
            output_dir=output_dir,
            base_filename=chunk_base_filename,
            voice=voice,
            speed=speed,
            split_pattern=split_pattern,
            use_lock=False,
        )
        if chunk_files:
            all_generated_files.extend(chunk_files)
            # Stream the latest audio segment
            for audio_file in chunk_files:
                yield audio_file, None  # None for merged audio (not ready yet)
        else:
            continue
    # After all chunks, merge if possible
    if all_generated_files:
        sorted_files = natsort.natsorted(all_generated_files)
        merged_filename = f"{base_filename}_merged.wav"
        merged_output_path = os.path.join(output_dir, merged_filename)
        success = merge_audio_files(sorted_files, merged_output_path)
        if success:
            yield None, merged_output_path
        else:
            yield None, None

def gradio_tts(
    input_text,
    input_file,
    lang,
    voice,
    speed,
    split_pattern,
    paragraphs_per_chunk,
):
    # Setup temp output dir for session
    temp_dir = tempfile.mkdtemp(prefix="tts_gradio_")
    ensure_dir_exists(temp_dir)
    try:
        # Determine input type and extract text
        if input_text and input_text.strip():
            text_to_process = input_text
            input_type = "text"
            base_filename = get_base_filename("text")
        elif input_file is not None:
            ext = get_file_extension(input_file.name)
            if ext == ".pdf":
                text_to_process = process_input_file(input_file, "pdf")
                input_type = "pdf"
            elif ext == ".txt":
                text_to_process = process_input_file(input_file, "txt")
                input_type = "txt"
            else:
                yield "Unsupported file type. Please upload a PDF or TXT file.", None, None
                shutil.rmtree(temp_dir)
                return
            base_filename = get_base_filename(input_type, file_obj=input_file)
        else:
            yield "No input provided.", None, None
            shutil.rmtree(temp_dir)
            return

        if not text_to_process or not text_to_process.strip():
            yield "No text found to process.", None, None
            shutil.rmtree(temp_dir)
            return

        # Initialize TTS processor
        tts_processor = KokoroTTSProcessor(lang_code=lang)
        tts_processor.set_generation_params(
            voice=voice, speed=speed, split_pattern=split_pattern
        )

        # Split text into chunks for streaming
        text_chunks = split_text_into_chunks(text_to_process, paragraphs_per_chunk)
        if not text_chunks:
            yield "Failed to split text into chunks.", None, None
            shutil.rmtree(temp_dir)
            return

        # Stream each chunk's audio as it's ready
        for audio_file, merged_audio in stream_tts_chunks(
            text_chunks,
            tts_processor,
            temp_dir,
            base_filename,
            voice,
            speed,
            split_pattern,
        ):
            if audio_file:
                yield None, audio_file, None  # (status, segment, merged)
            elif merged_audio:
                yield None, None, merged_audio
            else:
                continue

    finally:
        # Optionally, clean up temp_dir after session ends
        pass  # Comment this out if you want to keep files for debugging

# --- Gradio UI ---

with gr.Blocks(title="Kokoro-TTS: Text/PDF to Audio Converter") as demo:
    gr.Markdown(
        """
        # Kokoro-TTS: Text/PDF to Audio Converter
        Convert text, PDF, or TXT files to speech using Kokoro-TTS.<br>
        - Enter text or upload a PDF/TXT file.<br>
        - Audio segments are streamed as soon as they're ready.<br>
        - After all segments, a merged audio file is available for playback/download.<br>
        """
    )
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter Text",
                placeholder="Type or paste your text here...",
                lines=8,
            )
            input_file = gr.File(
                label="Or Upload PDF/TXT File",
                file_types=[".pdf", ".txt"],
            )
            lang = gr.Dropdown(
                label="Language",
                choices=[
                    ("American English", "a"),
                    ("British English", "b"),
                    ("Spanish", "e"),
                    ("French", "f"),
                    ("Hindi", "h"),
                    ("Italian", "i"),
                    ("Japanese", "j"),
                    ("Brazilian Portuguese", "p"),
                    ("Mandarin Chinese", "z"),
                ],
                value="a",
            )
            voice = gr.Textbox(
                label="Voice Model",
                value="af_heart",
                placeholder="e.g., af_heart (see Kokoro docs for more voices)",
            )
            speed = gr.Slider(
                label="Speech Speed",
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.05,
            )
            split_pattern = gr.Textbox(
                label="Split Pattern (Regex)",
                value=r"\n\n+|\r\n\r\n+|\n\s*\n+|[.!?]\s",
                placeholder="Regex for splitting text into segments",
            )
            paragraphs_per_chunk = gr.Slider(
                label="Paragraphs per Chunk",
                minimum=1,
                maximum=100,
                value=30,
                step=1,
            )
            submit_btn = gr.Button("Convert to Speech")

        with gr.Column():
            status = gr.Textbox(
                label="Status / Info",
                interactive=False,
                visible=False,
            )
            segment_audio = gr.Audio(
                label="Latest Audio Segment",
                interactive=False,
                visible=True,
                autoplay=True,
            )
            merged_audio = gr.Audio(
                label="Merged Audio (All Segments)",
                interactive=False,
                visible=True,
                autoplay=False,
            )

    def clear_outputs():
        return "", None, None

    submit_btn.click(
        gradio_tts,
        inputs=[
            input_text,
            input_file,
            lang,
            voice,
            speed,
            split_pattern,
            paragraphs_per_chunk,
        ],
        outputs=[status, segment_audio, merged_audio],
        concurrency_limit=1,
        show_progress=True,
        # stream=True,
    )

    gr.Button("Clear").click(
        clear_outputs,
        None,
        [status, segment_audio, merged_audio],
        show_progress=False,
    )

if __name__ == "__main__":
    demo.launch()
