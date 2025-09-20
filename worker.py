import logging
import os
import database as db
from utils.file_handler import ensure_dir_exists
from utils.split_text import smart_split_text
from utils.logger import setup_logging

def process_chunk_worker(job_name: str) -> int:
    """
    The main worker function that runs in a separate process.
    It continuously fetches and processes chunks for a given job.
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

    # This import needs to be inside the worker function for ProcessPoolExecutor
    from tts_engine.processor import KokoroTTSProcessor
    from tts_engine.chatterbox_processor import ChatterboxTTSProcessor

    try:
        if job_data['engine'] == 'kokoro':
            tts_processor = KokoroTTSProcessor(lang_code=job_data['lang'], device=job_data['device'])
            tts_processor.set_generation_params(voice=job_data['voice'], speed=job_data['speed'])
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
