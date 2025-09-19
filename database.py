import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def create_connection(db_file="tts_jobs.db"):
    """Create a database connection to a SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        logger.info(f"Successfully connected to SQLite database: {db_file}")
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
    return conn

def create_tables(conn):
    """Create the necessary tables if they don't exist."""
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
               cb_audio_prompt=None, cb_exaggeration=None, cb_cfg_weight=None, cb_temperature=None,
               cb_top_p=None, cb_min_p=None, cb_repetition_penalty=None):
    """Create a new job in the jobs table."""
    sql = ''' INSERT INTO jobs(job_name, input_file, output_dir, engine, lang, voice, speed, device, merge_output,
                               cb_audio_prompt, cb_exaggeration, cb_cfg_weight, cb_temperature,
                               cb_top_p, cb_min_p, cb_repetition_penalty)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
    try:
        params = (job_name, input_file, output_dir, engine, lang, voice, speed, device, merge_output,
                  cb_audio_prompt, cb_exaggeration, cb_cfg_weight, cb_temperature,
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
    """Retrieve a job by its name."""
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
    """Create chunks for a given job."""
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
    """Get a single pending chunk to process."""
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
    """Atomically claim a pending chunk for processing."""
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
                cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
                conn.row_factory = sqlite3.Row
                chunk_row = cursor.fetchone()
                return dict(chunk_row) if chunk_row else None
            else:
                return None # No pending chunks left
        except sqlite3.Error as e:
            logger.error(f"Error claiming chunk: {e}")
            return None

def update_chunk_status(conn, chunk_id, status, audio_file_path=None):
    """Update the status of a chunk."""
    sql = "UPDATE chunks SET status = ?, audio_file_path = ? WHERE id = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (status, audio_file_path, chunk_id))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error updating chunk status: {e}")

def update_job_status(conn, job_id, status):
    """Update the status of a job."""
    sql = "UPDATE jobs SET status = ? WHERE id = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (status, job_id))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error updating job status: {e}")

def get_chunks_for_job(conn, job_id):
    """Retrieve all chunks for a given job, ordered by index."""
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
    """Get statistics for a given job."""
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
    """Retrieve all jobs from the database."""
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
    """Reset all 'failed' or 'processing' chunks to 'pending' for a job."""
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
        job_id = create_job(db_conn, "test_job_1", "/path/to/file.txt", "/path/to/output", "kokoro", "a", "af_heart", 1.0, True)
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
