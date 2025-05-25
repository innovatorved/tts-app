# kokoro_tts_app/utils/file_handler.py
import os
import logging

logger = logging.getLogger(__name__)

def ensure_dir_exists(dir_path: str):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")
        except OSError as e:
            logger.error(f"Error creating directory {dir_path}: {e}")
            raise
    else:
        logger.debug(f"Directory already exists: {dir_path}")

def get_safe_filename(name: str) -> str:
    """Converts a string to a safe filename."""
    # Remove or replace characters not allowed in filenames
    # This is a basic version, more robust solutions might be needed for edge cases
    name = "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in name])
    name = name.replace(' ', '_')
    return name