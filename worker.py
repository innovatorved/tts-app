import logging
import os
import database as db
from utils.file_handler import ensure_dir_exists

def process_chunk_worker(job_name: str) -> int:
    """
    The main worker function that runs in a separate process.
    It continuously fetches and processes chunks for a given job.
    """
    # Re-initialize logging for the worker process
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s")
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

    # This import needs to be inside the worker function for ProcessPoolExecutor
    from tts_engine.processor import KokoroTTSProcessor
    from tts_engine.chatterbox_processor import ChatterboxTTSProcessor

    try:
        if job_data['engine'] == 'kokoro':
            tts_processor = KokoroTTSProcessor(lang_code=job_data['lang'], device=job_data['device'])
            tts_processor.set_generation_params(voice=job_data['voice'], speed=job_data['speed'], split_pattern=r"(?!.*)")
        elif job_data['engine'] == 'chatterbox':
            tts_processor = ChatterboxTTSProcessor(device=job_data['device'])
            tts_processor.set_generation_params(
                audio_prompt_path=job_data['cb_audio_prompt'],
                exaggeration=job_data['cb_exaggeration'],
                cfg_weight=job_data['cb_cfg_weight'],
                temperature=job_data['cb_temperature'],
                top_p=job_data['cb_top_p'],
                min_p=job_data['cb_min_p'],
                repetition_penalty=job_data['cb_repetition_penalty'],
                split_pattern=r"(?!.*)"
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

            audio_files = tts_processor.text_to_speech(
                text=chunk['text'],
                output_dir=job_data['output_dir'],
                base_filename=base_filename,
                use_lock=False
            )

            if audio_files:
                audio_file_path = audio_files[0]
                db.update_chunk_status(db_conn, chunk['id'], 'completed', audio_file_path)
                worker_logger.info(f"Worker {os.getpid()}: Successfully processed chunk {chunk['chunk_index']}.")
                processed_count += 1
            else:
                worker_logger.warning(f"Worker {os.getpid()}: TTS for chunk {chunk['chunk_index']} produced no audio.")
                db.update_chunk_status(db_conn, chunk['id'], 'failed')

        except Exception as e:
            worker_logger.error(f"Worker {os.getpid()}: Error processing chunk {chunk['chunk_index']}: {e}", exc_info=True)
            db.update_chunk_status(db_conn, chunk['id'], 'failed')

    db_conn.close()
    return processed_count
