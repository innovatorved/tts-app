import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def create_connection(db_file="tts_jobs.db"):
    """Creates and returns a connection to a SQLite database.

    This function establishes a connection to the SQLite database file specified.
    It is configured to be thread-safe by setting `check_same_thread=False`.

    Args:
        db_file: The path to the SQLite database file. Defaults to
            "tts_jobs.db".

    Returns:
        A sqlite3.Connection object or None if the connection fails.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        logger.info(f"Successfully connected to SQLite database: {db_file}")
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
    return conn

def create_tables(conn):
    """Creates the 'jobs' and 'chunks' tables in the database if they don't exist.

    Args:
        conn: An active sqlite3.Connection object.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_name TEXT NOT NULL UNIQUE,
                input_file TEXT,
                output_dir TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                engine TEXT,
                lang TEXT,
                voice TEXT,
                speed REAL,
                device TEXT,
                merge_output BOOLEAN,
                cb_audio_prompt TEXT,
                cb_voice_cloning BOOLEAN DEFAULT 0,
                cb_exaggeration REAL,
                cb_cfg_weight REAL,
                cb_temperature REAL,
                cb_top_p REAL,
                cb_min_p REAL,
                cb_repetition_penalty REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                audio_file_path TEXT,
                retries INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs (id),
                UNIQUE (job_id, chunk_index)
            );
        """)
        conn.commit()
        logger.info("Tables 'jobs' and 'chunks' are ready.")
    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}")

def create_job(conn, job_name, input_file, output_dir, engine, lang, voice, speed, device, merge_output,
               cb_audio_prompt=None, cb_voice_cloning=False, cb_exaggeration=None, cb_cfg_weight=None, cb_temperature=None,
               cb_top_p=None, cb_min_p=None, cb_repetition_penalty=None):
    """Creates a new job record in the 'jobs' table.

    If a job with the same `job_name` already exists, it does not create a
    new one but returns the ID of the existing job.

    Args:
        conn: An active sqlite3.Connection object.
        job_name: A unique name for the job.
        input_file: Path to the source file (PDF, TXT, etc.).
        output_dir: Directory to save the final audio.
        engine: The TTS engine to use ('kokoro' or 'chatterbox').
        lang: Language code for the TTS engine.
        voice: Voice model name.
        speed: Speech speed multiplier.
        device: The compute device to use.
        merge_output: Boolean flag to merge audio chunks.
        cb_audio_prompt: Path to reference audio for Chatterbox voice cloning.
        cb_voice_cloning: Boolean flag to enable voice cloning mode.
        cb_exaggeration: Exaggeration parameter for Chatterbox.
        cb_cfg_weight: CFG weight for Chatterbox.
        cb_temperature: Temperature for Chatterbox.
        cb_top_p: Top-p sampling for Chatterbox.
        cb_min_p: Min-p sampling for Chatterbox.
        cb_repetition_penalty: Repetition penalty for Chatterbox.

    Returns:
        The integer ID of the newly created or existing job, or None on error.
    """
    sql = ''' INSERT INTO jobs(job_name, input_file, output_dir, engine, lang, voice, speed, device, merge_output,
                               cb_audio_prompt, cb_voice_cloning, cb_exaggeration, cb_cfg_weight, cb_temperature,
                               cb_top_p, cb_min_p, cb_repetition_penalty)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
    try:
        params = (job_name, input_file, output_dir, engine, lang, voice, speed, device, merge_output,
                  cb_audio_prompt, cb_voice_cloning, cb_exaggeration, cb_cfg_weight, cb_temperature,
                  cb_top_p, cb_min_p, cb_repetition_penalty)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        logger.warning(f"Job with name '{job_name}' already exists. Returning existing job ID.")
        job = get_job_by_name(conn, job_name)
        return job['id'] if job else None
    except sqlite3.Error as e:
        logger.error(f"Error creating job: {e}")
        return None

def get_job_by_name(conn, job_name):
    """Retrieves a single job record by its unique name.

    Args:
        conn: An active sqlite3.Connection object.
        job_name: The name of the job to retrieve.

    Returns:
        A dictionary representing the job record, or None if not found or on error.
    """
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE job_name = ?", (job_name,))
        job = cursor.fetchone()
        return dict(job) if job else None
    except sqlite3.Error as e:
        logger.error(f"Error getting job by name: {e}")
        return None

def create_chunks(conn, job_id, text_chunks):
    """Creates multiple chunk records for a given job in a single transaction.

    Empty or whitespace-only chunks in the input list are automatically skipped.

    Args:
        conn: An active sqlite3.Connection object.
        job_id: The ID of the parent job.
        text_chunks: A list of strings, where each string is the text for a chunk.
    """
    sql = ''' INSERT INTO chunks(job_id, chunk_index, text)
              VALUES(?,?,?) '''
    try:
        if not text_chunks:
            logger.warning(f"No chunks supplied for job ID {job_id}; nothing to insert.")
            return

        # Filter out empty / whitespace-only chunks proactively
        filtered = [c.strip() for c in text_chunks if c and c.strip()]
        skipped = len(text_chunks) - len(filtered)
        if skipped:
            logger.info(f"Skipped {skipped} empty/blank chunk(s) for job ID {job_id}.")

        if not filtered:
            logger.warning(f"All provided chunks were empty for job ID {job_id}; nothing inserted.")
            return

        cursor = conn.cursor()
        # Ensure contiguous chunk_index (0..n-1) after filtering
        chunk_data = [(job_id, i, chunk) for i, chunk in enumerate(filtered)]
        cursor.executemany(sql, chunk_data)
        conn.commit()
        logger.info(f"Successfully created {len(filtered)} chunks for job ID {job_id} (skipped {skipped}).")
    except sqlite3.Error as e:
        logger.error(f"Error creating chunks: {e}")

def get_pending_chunk(conn, job_id):
    """Retrieves the next available chunk with 'pending' status for a job.

    Args:
        conn: An active sqlite3.Connection object.
        job_id: The ID of the job to fetch a chunk from.

    Returns:
        A dictionary representing the chunk record, or None if no pending chunks
        are found or an error occurs.
    """
    sql = "SELECT * FROM chunks WHERE job_id = ? AND status = 'pending' ORDER BY chunk_index ASC LIMIT 1"
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql, (job_id,))
        chunk = cursor.fetchone()
        return dict(chunk) if chunk else None
    except sqlite3.Error as e:
        logger.error(f"Error fetching pending chunk: {e}")
        return None

def claim_chunk(conn, job_id):
    """Atomically retrieves a pending chunk and updates its status to 'processing'.

    This function ensures that in a multi-worker setup, a single chunk is
    claimed by only one worker.

    Args:
        conn: An active sqlite3.Connection object.
        job_id: The ID of the job from which to claim a chunk.

    Returns:
        A dictionary representing the claimed chunk, or None if no pending
        chunks are available or an error occurs.
    """
    with conn: # Using 'with conn' ensures the transaction is handled correctly
        try:
            cursor = conn.cursor()
            # Find a pending chunk
            cursor.execute("SELECT id FROM chunks WHERE job_id = ? AND status = 'pending' ORDER BY chunk_index ASC LIMIT 1", (job_id,))
            chunk_id_row = cursor.fetchone()

            if chunk_id_row:
                chunk_id = chunk_id_row[0]
                # Update its status to 'processing'
                cursor.execute("UPDATE chunks SET status = 'processing' WHERE id = ?", (chunk_id,))

                # Retrieve the full chunk data
                conn.row_factory = sqlite3.Row
                cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
                chunk_row = cursor.fetchone()
                return dict(chunk_row) if chunk_row else None
            else:
                return None # No pending chunks left
        except sqlite3.Error as e:
            logger.error(f"Error claiming chunk: {e}")
            return None

def update_chunk_status(conn, chunk_id, status, audio_file_path=None):
    """Updates the status and audio file path of a specific chunk.

    Args:
        conn: An active sqlite3.Connection object.
        chunk_id: The ID of the chunk to update.
        status: The new status string (e.g., 'completed', 'failed').
        audio_file_path: The path to the generated audio file. Defaults to None.
    """
    sql = "UPDATE chunks SET status = ?, audio_file_path = ? WHERE id = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (status, audio_file_path, chunk_id))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error updating chunk status: {e}")

def update_job_status(conn, job_id, status):
    """Updates the status of a specific job.

    Args:
        conn: An active sqlite3.Connection object.
        job_id: The ID of the job to update.
        status: The new status string (e.g., 'processing', 'completed').
    """
    sql = "UPDATE jobs SET status = ? WHERE id = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (status, job_id))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error updating job status: {e}")

def get_chunks_for_job(conn, job_id):
    """Retrieves all chunks associated with a given job, ordered by index.

    Args:
        conn: An active sqlite3.Connection object.
        job_id: The ID of the job.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk.
        Returns an empty list on error.
    """
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE job_id = ? ORDER BY chunk_index ASC", (job_id,))
        chunks = cursor.fetchall()
        return [dict(chunk) for chunk in chunks]
    except sqlite3.Error as e:
        logger.error(f"Error getting chunks for job: {e}")
        return []

def get_job_stats(conn, job_id):
    """Calculates statistics for a given job based on its chunk statuses.

    Args:
        conn: An active sqlite3.Connection object.
        job_id: The ID of the job.

    Returns:
        A dictionary with counts for each status (e.g., 'completed', 'pending')
        and a 'total' count. Returns an empty dictionary on error.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT status, COUNT(*) FROM chunks WHERE job_id = ? GROUP BY status", (job_id,))
        stats = dict(cursor.fetchall())
        total_chunks = sum(stats.values())
        stats['total'] = total_chunks
        return stats
    except sqlite3.Error as e:
        logger.error(f"Error getting job stats: {e}")
        return {}

def get_all_jobs(conn):
    """Retrieves a summary of all jobs from the database.

    Args:
        conn: An active sqlite3.Connection object.

    Returns:
        A list of dictionaries, each representing a job summary, ordered by
        creation date descending. Returns an empty list on error.
    """
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, job_name, status, created_at FROM jobs ORDER BY created_at DESC")
        jobs = cursor.fetchall()
        return [dict(job) for job in jobs]
    except sqlite3.Error as e:
        logger.error(f"Error getting all jobs: {e}")
        return []

def reset_failed_chunks(conn, job_id):
    """Resets chunks with 'failed' or 'processing' status back to 'pending'.

    This is used for resuming jobs, allowing workers to retry chunks that
    previously failed or were stuck in a 'processing' state (e.g., if a
    worker crashed).

    Args:
        conn: An active sqlite3.Connection object.
        job_id: The ID of the job whose chunks need resetting.

    Returns:
        The number of chunks that were updated.
    """
    sql = "UPDATE chunks SET status = 'pending', retries = retries + 1 WHERE job_id = ? AND status IN ('failed', 'processing')"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (job_id,))
        updated_count = cursor.rowcount
        conn.commit()
        logger.info(f"Reset {updated_count} failed/stuck chunks to 'pending' for job ID {job_id}.")
        return updated_count
    except sqlite3.Error as e:
        logger.error(f"Error resetting failed chunks: {e}")
        return 0

if __name__ == '__main__':
    # Example usage
    db_conn = create_connection("tts_jobs_test.db")
    if db_conn:
        create_tables(db_conn)
        job_id = create_job(db_conn, "test_job_1", "/path/to/file.txt", "/path/to/output", "kokoro", "a", "af_heart", 1.0, "cpu", True)
        if job_id:
            sample_chunks = ["This is the first sentence.", "This is the second.", "And a third."]
            create_chunks(db_conn, job_id, sample_chunks)

            print(f"Created job with ID: {job_id}")

            pending_chunk = get_pending_chunk(db_conn, job_id)
            if pending_chunk:
                print(f"Processing chunk: {pending_chunk['id']}")
                update_chunk_status(db_conn, pending_chunk['id'], "completed", "/path/to/audio_0.wav")
                print("Chunk status updated.")

            stats = get_job_stats(db_conn, job_id)
            print(f"Job stats: {stats}")

        db_conn.close()
