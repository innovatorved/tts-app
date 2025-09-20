import gradio as gr
import os
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import natsort
import pandas as pd
import sys

# Local imports
import database as db
from utils.logger import setup_logging
from worker import process_chunk_worker
from utils.pdf_parser import extract_text_from_pdf
from utils.text_file_parser import extract_text_from_txt
from utils.split_text import split_text_into_chunks
from utils.audio_merger import merge_audio_files
from utils.file_handler import ensure_dir_exists

# Adjust path to import from sibling directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# --- Logging Setup ---
setup_logging(main_process=True)
logger = logging.getLogger(__name__)
# ---

def run_job_processing(job_name, num_workers):
    """Synchronously runs the ProcessPoolExecutor for a given job."""
    db_conn = db.create_connection()
    if not db_conn:
        return False
    
    try:
        logger.info(f"Starting ProcessPoolExecutor with {num_workers} workers for job '{job_name}'.")
        job_id = db.get_job_by_name(db_conn, job_name)['id']
        db.update_job_status(db_conn, job_id, 'processing')
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_chunk_worker, job_name) for _ in range(num_workers)]
            for future in as_completed(futures):
                future.result() # Wait for all workers to complete
        
        logger.info(f"All workers have finished for job '{job_name}'.")

        stats = db.get_job_stats(db_conn, job_id)

        if stats.get('total', 0) == stats.get('completed', 0):
            db.update_job_status(db_conn, job_id, 'completed')
            return True
        else:
            db.update_job_status(db_conn, job_id, 'failed')
            return False
    finally:
        db_conn.close()

def create_and_run_job(
    file_obj, text_input, num_workers, paragraphs_per_chunk,
    output_dir, engine, lang, voice, speed, device, merge_output,
    cb_audio_prompt, cb_exaggeration, cb_cfg_weight
):
    """Handles job creation and execution for the Web UI."""
    db_conn = db.create_connection()
    if not db_conn:
        yield "Error: Could not connect to the database.", None, gr.update(interactive=True), gr.update(interactive=True)
        return
    db.create_tables(db_conn)

    try:
        # --- Determine Job Name and Extract Text ---
        text_to_process = ""
        input_source_name = "direct_text"
        input_file_path = "direct_text"
        if file_obj is not None:
            input_file_path = file_obj.name
            input_source_name = Path(input_file_path).stem
            file_ext = Path(input_file_path).suffix.lower()
            if file_ext == '.pdf':
                text_to_process = extract_text_from_pdf(input_file_path)
            elif file_ext in ['.txt', '.md']:
                text_to_process = extract_text_from_txt(input_file_path)
        elif text_input:
            text_to_process = text_input
        
        if not text_to_process.strip():
            yield "Error: No text to process.", None, gr.update(interactive=True), gr.update(interactive=True)
            return

        job_name = f"{input_source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # --- Create Job in DB ---
        cb_prompt_path = cb_audio_prompt.name if cb_audio_prompt else None
        job_id = db.create_job(
            conn=db_conn, job_name=job_name,
            input_file=input_file_path,
            output_dir=output_dir, engine=engine, lang=lang,
            voice=voice, speed=speed, device=device, merge_output=merge_output,
            cb_audio_prompt=cb_prompt_path,
            cb_exaggeration=cb_exaggeration,
            cb_cfg_weight=cb_cfg_weight
        )
        if not job_id:
            yield f"Error: Job '{job_name}' already exists or could not be created.", None, gr.update(interactive=True), gr.update(interactive=True)
            return

        text_chunks = split_text_into_chunks(text_to_process, paragraphs_per_chunk)
        db.create_chunks(db_conn, job_id, text_chunks)

        # --- Run Processing ---
        status_message = f"Job '{job_name}' created with {len(text_chunks)} chunks. Processing..."
        yield status_message, None, gr.update(interactive=False), gr.update(interactive=False)

        job_successful = run_job_processing(job_name, num_workers)

        # --- Finalize and Return Result ---
        if job_successful:
            job_data = db.get_job_by_name(db_conn, job_name)
            if job_data['merge_output']:
                all_chunks = db.get_chunks_for_job(db_conn, job_id)
                audio_files = [c['audio_file_path'] for c in all_chunks if c['status'] == 'completed' and c['audio_file_path']]
                if audio_files:
                    sorted_files = natsort.natsorted(audio_files)
                    merged_filename = f"{job_name}_merged.wav"
                    merged_path = os.path.join(job_data['output_dir'], merged_filename)
                    ensure_dir_exists(job_data['output_dir'])
                    merge_audio_files(sorted_files, merged_path)
                    yield f"Job '{job_name}' completed and merged successfully!", merged_path, gr.update(interactive=True), gr.update(interactive=True)
                else:
                    yield f"Job '{job_name}' completed, but no audio files found to merge.", None, gr.update(interactive=True), gr.update(interactive=True)
            else:
                yield f"Job '{job_name}' completed successfully (no merging).", None, gr.update(interactive=True), gr.update(interactive=True)
        else:
            yield f"Error: Job '{job_name}' failed or completed with errors.", None, gr.update(interactive=True), gr.update(interactive=True)

    finally:
        db_conn.close()

def get_jobs_df():
    """Fetches all jobs from the database for display."""
    db_conn = db.create_connection()
    if not db_conn:
        return pd.DataFrame()
    try:
        jobs = db.get_all_jobs(db_conn)
        if not jobs:
            return pd.DataFrame(columns=['ID', 'Job Name', 'Status', 'Created At'])
        df = pd.DataFrame(jobs)
        df = df.rename(columns={'id': 'ID', 'job_name': 'Job Name', 'status': 'Status', 'created_at': 'Created At'})
        return df
    finally:
        db_conn.close()

def create_ui():
    with gr.Blocks(title="TTS App - Advanced", theme=gr.themes.Monochrome()) as interface:
        gr.Markdown("# 🎵 TTS: Scalable Text-to-Speech")
        
        with gr.Tabs():
            with gr.TabItem("Create Job"):
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(label="Upload File (PDF/TXT)", file_types=[".pdf", ".txt", ".md"])
                        text_input = gr.Textbox(label="Or Enter Text", lines=5)
                        output_dir = gr.Textbox(label="Output Directory", value="./output_audio")
                        
                        with gr.Accordion("Advanced Settings", open=False):
                            engine = gr.Radio(["kokoro", "chatterbox"], label="TTS Engine", value="kokoro")

                            with gr.Group(visible=True) as kokoro_settings:
                                lang = gr.Textbox(label="Language Code (Kokoro)", value="a")
                                voice = gr.Textbox(label="Voice (Kokoro)", value="af_heart")
                                speed = gr.Slider(label="Speed", minimum=0.5, maximum=2.0, value=1.0)

                            with gr.Group(visible=False) as chatterbox_settings:
                                cb_audio_prompt = gr.File(label="Reference Audio (Chatterbox)", file_types=[".wav"])
                                cb_exaggeration = gr.Slider(label="Exaggeration", minimum=0, maximum=2, value=0.5)
                                cb_cfg_weight = gr.Slider(label="CFG Weight", minimum=0, maximum=2, value=0.5)

                            device = gr.Radio(["cpu", "cuda", "mps"], label="Device", value="cpu")
                            num_workers = gr.Slider(label="Number of Workers", minimum=1, maximum=os.cpu_count(), step=1, value=2)
                            paragraphs_per_chunk = gr.Slider(label="Paragraphs per Chunk", minimum=1, maximum=50, step=1, value=10)
                            merge_output = gr.Checkbox(label="Merge Output Audio", value=True)

                    with gr.Column(scale=1):
                        status_box = gr.Textbox(label="Status", interactive=False)
                        audio_output = gr.Audio(label="Generated Audio")
                        submit_btn = gr.Button("Start Job", variant="primary")

            with gr.TabItem("Job Dashboard"):
                jobs_df = gr.DataFrame(label="All Jobs", interactive=False)
                refresh_btn = gr.Button("Refresh Jobs List")

        # --- Event Handlers ---
        def toggle_engine_settings(engine_choice):
            if engine_choice == "kokoro":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        engine.change(
            toggle_engine_settings,
            inputs=engine,
            outputs=[kokoro_settings, chatterbox_settings]
        )

        submit_btn.click(
            create_and_run_job,
            inputs=[
                file_input, text_input, num_workers, paragraphs_per_chunk,
                output_dir, engine, lang, voice, speed, device, merge_output,
                cb_audio_prompt, cb_exaggeration, cb_cfg_weight
            ],
            outputs=[status_box, audio_output, submit_btn, refresh_btn]
        )
        
        refresh_btn.click(
            get_jobs_df,
            inputs=[],
            outputs=[jobs_df]
        )
        
        interface.load(get_jobs_df, outputs=jobs_df) # Load jobs on startup

    return interface

if __name__ == "__main__":
    # Initialize the database and tables on startup
    init_db_conn = db.create_connection()
    if init_db_conn:
        db.create_tables(init_db_conn)
        init_db_conn.close()

    ui = create_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
