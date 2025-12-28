import logging
import os

# Set environment limits BEFORE importing torch/heavy libs
# This is critical as PyTorch reads these at import time
from utils.resource_limiter import ResourceConfig, apply_resource_limits, set_environment_limits

# Apply environment-level thread limits early (before torch import)
set_environment_limits(max_threads=4)  # Default conservative limit

import database as db
from utils.file_handler import ensure_dir_exists
from utils.split_text import smart_split_text
from utils.logger import setup_logging

def process_chunk_worker(job_name: str) -> int:
    """The main worker function that runs in a separate process to handle TTS.

    This function is designed to be executed by a process pool. It connects to
    the database, retrieves the job details, and enters a loop to continuously
    claim and process text chunks associated with the job.

    Inside the loop, it performs the following steps for each chunk:
    1. Claims a 'pending' chunk from the database, atomically setting its status
       to 'processing'.
    2. Initializes the appropriate TTS engine (Kokoro or Chatterbox).
    3. Splits the chunk's text into smaller, manageable segments.
    4. Calls the TTS engine to convert each segment into an audio file.
    5. Updates the chunk's status to 'completed' or 'failed' in the database.

    The function exits when no more 'pending' chunks are available for the job.

    Args:
        job_name: The unique name of the job this worker should process.

    Returns:
        The total number of chunks successfully processed by this worker instance.
    """
    # Re-initialize logging for the worker process
    setup_logging(main_process=False)
    worker_logger = logging.getLogger(__name__)

    db_conn = db.create_connection()
    if not db_conn:
        worker_logger.error(f"Worker for job '{job_name}': Could not connect to database. Exiting.")
        return 0

    job_data = db.get_job_by_name(db_conn, job_name)
    if not job_data:
        worker_logger.error(f"Worker for job '{job_name}': Could not find job data. Exiting.")
        db_conn.close()
        return 0

    # Apply resource limits to prevent system overload
    max_threads = job_data.get('max_torch_threads', 4)
    # For Chatterbox, use more restrictive defaults
    if job_data.get('engine') == 'chatterbox':
        max_threads = min(max_threads, 2)  # Chatterbox needs fewer threads
    
    resource_config = ResourceConfig(
        max_cpu_cores=job_data.get('max_cpu_cores'),
        max_torch_threads=max_threads,
        max_gpu_memory_fraction=job_data.get('max_gpu_memory', 0.75),
        low_priority=job_data.get('low_priority', True),
    )
    apply_resource_limits(resource_config, device=job_data.get('device'))

    # This import needs to be inside the worker function for ProcessPoolExecutor
    from tts_engine.processor import KokoroTTSProcessor
    from tts_engine.chatterbox_processor import ChatterboxTTSProcessor

    try:
        if job_data['engine'] == 'kokoro':
            tts_processor = KokoroTTSProcessor(lang_code=job_data['lang'], device=job_data['device'])
            tts_processor.set_generation_params(voice=job_data['voice'], speed=job_data['speed'])
        elif job_data['engine'] == 'chatterbox':
            tts_processor = ChatterboxTTSProcessor(
                device=job_data['device'],
                enable_voice_cloning=job_data.get('cb_voice_cloning', False)
            )
            tts_processor.set_generation_params(
                audio_prompt_path=job_data['cb_audio_prompt'],

                temperature=job_data['cb_temperature'],
                top_p=job_data['cb_top_p'],
                repetition_penalty=job_data['cb_repetition_penalty'],
            )
        else:
            worker_logger.warning(f"Worker for job '{job_name}': Engine '{job_data['engine']}' is not supported.")
            tts_processor = None
    except Exception as e:
        worker_logger.error(f"Worker for job '{job_name}': Failed to initialize TTS processor: {e}. Exiting.", exc_info=True)
        db_conn.close()
        return 0

    worker_logger.info(f"Worker process {os.getpid()} started for job '{job_name}'.")
    processed_count = 0

    while True:
        chunk = db.claim_chunk(db_conn, job_data['id'])
        if not chunk:
            worker_logger.info(f"Worker {os.getpid()}: No more pending chunks for job '{job_name}'. Exiting.")
            break

        try:
            worker_logger.info(f"Worker {os.getpid()}: Processing chunk {chunk['chunk_index']} for job '{job_name}'.")
            ensure_dir_exists(job_data['output_dir'])
            base_filename = f"{job_name}_chunk_{chunk['chunk_index']:04d}"

            # External segmentation to avoid double splitting inside processors
            # We keep a minimal split pattern to rely on smart_split_text defaults
            segments = smart_split_text(chunk['text'])
            if not segments:
                worker_logger.warning(f"Worker {os.getpid()}: Chunk {chunk['chunk_index']} produced no segments after splitting.")
                db.update_chunk_status(db_conn, chunk['id'], 'failed')
                continue

            generated_files = []
            for seg_idx, seg_text in enumerate(segments):
                seg_base = f"{base_filename}_segment_{seg_idx:03d}"  # maintain compatibility with merger glob
                # Pass pre_split=True so processor treats whole seg_text as single unit
                audio_files = tts_processor.text_to_speech(
                    text=seg_text,
                    output_dir=job_data['output_dir'],
                    base_filename=seg_base,
                    use_lock=False,
                )
                if audio_files:
                    generated_files.extend(audio_files)
                else:
                    worker_logger.warning(f"Worker {os.getpid()}: No audio returned for segment {seg_idx} of chunk {chunk['chunk_index']}.")

            if generated_files:
                # For database we record first file (others share naming pattern)
                db.update_chunk_status(db_conn, chunk['id'], 'completed', generated_files[0])
                worker_logger.info(f"Worker {os.getpid()}: Successfully processed chunk {chunk['chunk_index']} into {len(generated_files)} segment file(s).")
                processed_count += 1
            else:
                db.update_chunk_status(db_conn, chunk['id'], 'failed')
                worker_logger.warning(f"Worker {os.getpid()}: All segments failed for chunk {chunk['chunk_index']}.")

        except Exception as e:
            worker_logger.error(f"Worker {os.getpid()}: Error processing chunk {chunk['chunk_index']}: {e}", exc_info=True)
            db.update_chunk_status(db_conn, chunk['id'], 'failed')

    db_conn.close()
    return processed_count
