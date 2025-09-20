import os
import logging

logger = logging.getLogger(__name__)


def ensure_dir_exists(dir_path: str):
    """Checks if a directory exists at the given path and creates it if not.

    Args:
        dir_path: The path to the directory to check.

    Raises:
        OSError: If the directory could not be created due to an OS-level error.
    """
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
    """Sanitizes a string to create a safe filename.

    This function replaces spaces with underscores and removes or replaces
    characters that are not alphanumeric, a period, an underscore, or a hyphen.

    Args:
        name: The string to convert into a safe filename.

    Returns:
        A sanitized string suitable for use as a filename.
    """
    # Remove or replace characters not allowed in filenames
    # This is a basic version, more robust solutions might be needed for edge cases
    name = "".join(
        [c if c.isalnum() or c in (" ", ".", "_", "-") else "_" for c in name]
    )
    name = name.replace(" ", "_")
    return name
