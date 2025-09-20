import argparse
import logging
import os
import sys
import glob
from datetime import datetime
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import natsort

import database as db
from utils.logger import setup_logging
from utils.pdf_parser import extract_text_from_pdf
from utils.file_handler import ensure_dir_exists
from utils.text_file_parser import extract_text_from_txt
from utils.conversation_parser import extract_conversation_from_text
from utils.split_text import split_text_into_chunks
from utils.audio_merger import merge_audio_files
from worker import process_chunk_worker

# Adjust path to import from sibling directories
# Adjust path to ensure the app's root directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# --- Logging Setup ---
setup_logging(main_process=True)
logger = logging.getLogger(__name__)
# ---

def main():
    parser = argparse.ArgumentParser(
        description="TTS: A scalable and reliable Text-to-Speech application."
    )

    # --- Job control arguments ---
    parser.add_argument("--job-name", type=str, help="Unique name for the conversion job. If not provided, one will be generated.")
    parser.add_argument("--resume", action="store_true", help="Resume a failed or interrupted job by its --job-name.")
    parser.add_argument("--monitor", action="store_true", help="Monitor the progress of a job by its --job-name.")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="Number of worker processes to use.")

    # --- Input source group (optional if resuming or monitoring) ---
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", type=str, help="Direct text to convert.")
    input_group.add_argument("--pdf", type=str, help="Path to a PDF file.")
    input_group.add_argument("--text_file", type=str, help="Path to a text file.")
    input_group.add_argument("--conversation", type=str, help="Path to a conversation file.")

    # --- Standard TTS and output arguments ---
    parser.add_argument("--output_dir", type=str, default="./output_audio", help="Directory to save audio files.")
    parser.add_argument("--engine", type=str, default="kokoro", choices=["kokoro", "chatterbox"], help="TTS engine.")
    parser.add_argument("--lang", type=str, default="a", help="Language code for Kokoro.")
    parser.add_argument("--voice", type=str, default="af_heart", help="Voice model for Kokoro.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Device to use for TTS.")
    parser.add_argument("--merge_output", action="store_true", help="Merge final audio segments.")
    parser.add_argument("--paragraphs_per_chunk", type=int, default=10, help="Number of paragraphs per processing chunk.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")

    # --- Chatterbox-specific arguments ---
    cb_group = parser.add_argument_group("Chatterbox Options")
    cb_group.add_argument("--cb_audio_prompt", type=str, help="Path to reference audio for Chatterbox.")
    cb_group.add_argument("--cb_exaggeration", type=float, default=0.5, help="Chatterbox emotion exaggeration.")
    cb_group.add_argument("--cb_cfg_weight", type=float, default=0.5, help="Chatterbox CFG weight.")
    cb_group.add_argument("--cb_temperature", type=float, default=0.8, help="Chatterbox temperature.")
    cb_group.add_argument("--cb_top_p", type=float, default=1.0, help="Chatterbox top_p.")
    cb_group.add_argument("--cb_min_p", type=float, default=0.05, help="Chatterbox min_p.")
    cb_group.add_argument("--cb_repetition_penalty", type=float, default=1.2, help="Chatterbox repetition penalty.")

    args = parser.parse_args()

    if args.verbose:
        setup_logging(level=logging.DEBUG, main_process=True)
        logger.info("Verbose logging enabled.")

    db_conn = db.create_connection()
    if not db_conn:
        return
    db.create_tables(db_conn)

    # --- Mode selection: Monitor, Resume, or Create/Process ---
    if args.monitor:
        if not args.job_name:
            logger.error("--job-name is required for monitoring.")
            return
        monitor_job(db_conn, args.job_name)
        db_conn.close()
        return

    job_to_process = None
    if args.resume:
        if not args.job_name:
            logger.error("--job-name is required for resuming.")
            db_conn.close()
            return
        job = db.get_job_by_name(db_conn, args.job_name)
        if not job:
            logger.error(f"No job found with name: {args.job_name}")
            db_conn.close()
            return
        db.reset_failed_chunks(db_conn, job['id'])
        logger.info(f"Job '{args.job_name}' is ready to be resumed.")
        job_to_process = args.job_name

    # --- Job Creation (if input is provided) ---
    input_source = args.text or args.pdf or args.text_file or args.conversation
    if not input_source and not args.resume:
        parser.error("An input source (--text, --pdf, etc.) is required to start a new job, or use --resume or --monitor with --job-name.")

    if input_source:
        job_name = args.job_name
        if not job_name:
            if args.pdf or args.text_file or args.conversation:
                base_name = Path(input_source).stem
            else:
                base_name = "direct_text"
            job_name = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Creating new job: {job_name}")

        text_to_process = ""
        if args.text: text_to_process = args.text
        elif args.pdf: text_to_process = extract_text_from_pdf(args.pdf)
        elif args.text_file: text_to_process = extract_text_from_txt(args.text_file)
        elif args.conversation: text_to_process = extract_text_from_txt(args.conversation)

        if not text_to_process or not text_to_process.strip():
            logger.error("Input source is empty or could not be read. Exiting.")
            db_conn.close()
            return

        job_id = db.create_job(
            conn=db_conn, job_name=job_name,
            input_file=input_source if not args.text else "direct_text",
            output_dir=args.output_dir, engine=args.engine, lang=args.lang,
            voice=args.voice, speed=args.speed, device=args.device,
            merge_output=args.merge_output,
            cb_audio_prompt=args.cb_audio_prompt,
            cb_exaggeration=args.cb_exaggeration,
            cb_cfg_weight=args.cb_cfg_weight,
            cb_temperature=args.cb_temperature,
            cb_top_p=args.cb_top_p,
            cb_min_p=args.cb_min_p,
            cb_repetition_penalty=args.cb_repetition_penalty
        )

        if not job_id:
            logger.error("Failed to create job in the database. Exiting.")
            db_conn.close()
            return

        text_chunks = split_text_into_chunks(text_to_process, args.paragraphs_per_chunk)
        db.create_chunks(db_conn, job_id, text_chunks)

        logger.info(f"Job '{job_name}' created with {len(text_chunks)} chunks. Starting processing...")
        job_to_process = job_name

    # --- Processing Logic ---
    if job_to_process:
        logger.info(f"Starting ProcessPoolExecutor with {args.num_workers} workers for job '{job_to_process}'.")
        db.update_job_status(db_conn, db.get_job_by_name(db_conn, job_to_process)['id'], 'processing')

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_chunk_worker, job_to_process) for _ in range(args.num_workers)]

            total_processed = 0
            for future in as_completed(futures):
                total_processed += future.result()

        logger.info(f"All workers have finished. Total chunks processed in this run: {total_processed}.")

        # --- Finalization and Merging ---
        job_id = db.get_job_by_name(db_conn, job_to_process)['id']
        stats = db.get_job_stats(db_conn, job_id)

        if stats.get('total') == stats.get('completed', 0):
            logger.info(f"Job '{job_to_process}' completed successfully.")
            db.update_job_status(db_conn, job_id, 'completed')

            job_data = db.get_job_by_name(db_conn, job_to_process)
            if job_data and job_data['merge_output']:
                logger.info("Merging audio files...")
                # Collect ALL segment files generated for this job across all chunks.
                # Each chunk may produce multiple segment WAV files with pattern: {job_name}_chunk_XXXX_segment_YYY.wav
                pattern = os.path.join(job_data['output_dir'], f"{job_to_process}_chunk_*_segment_*.wav")
                segment_files = glob.glob(pattern)
                if not segment_files:
                    logger.warning("No segment audio files found for merging (pattern: %s).", pattern)
                else:
                    # Natural sort to ensure correct chronological ordering
                    sorted_files = natsort.natsorted(segment_files)
                    print(sorted_files)
                    merged_filename = f"{job_to_process}_merged.wav"
                    merged_output_path = os.path.join(job_data['output_dir'], merged_filename)
                    logger.info(f"Merging {len(sorted_files)} segment files into {merged_output_path}")
                    success = merge_audio_files(sorted_files, merged_output_path)
                    if success:
                        logger.info(f"Successfully merged {len(sorted_files)} segments into {merged_output_path}")
                    else:
                        logger.error("Failed to merge audio files (see previous errors).")

        else:
            logger.warning(f"Job '{job_to_process}' finished with incomplete or failed chunks.")
            db.update_job_status(db_conn, job_id, 'failed')

    db_conn.close()


def monitor_job(conn, job_name):
    """Monitors the progress of a job."""
    job = db.get_job_by_name(conn, job_name)
    if not job:
        logger.error(f"No job found with name: {job_name}")
        return

    job_id = job['id']

    try:
        while True:
            stats = db.get_job_stats(conn, job_id)
            total = stats.get('total', 0)
            if total == 0:
                logger.info("Waiting for chunks to be created...")
                time.sleep(5)
                continue

            completed = stats.get('completed', 0)
            pending = stats.get('pending', 0)
            processing = stats.get('processing', 0)
            failed = stats.get('failed', 0)

            progress = (completed / total) * 100

            bar_length = 50
            filled_length = int(bar_length * completed // total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)

            print(f"\rProgress for job '{job_name}': |{bar}| {progress:.2f}% ({completed}/{total} Chunks)  ", end="")

            if completed == total:
                print("\nJob completed successfully!")
                break
            if failed > 0 and pending == 0 and processing == 0:
                print(f"\nJob finished with {failed} failed chunks. You can resume it with the --resume flag.")
                break

            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
