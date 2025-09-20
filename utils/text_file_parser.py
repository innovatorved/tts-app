import logging

logger = logging.getLogger(__name__)


def extract_text_from_txt(txt_path: str) -> str | None:
    """Reads and returns the content of a plain text file.

    This function attempts to read a text file using UTF-8 encoding. If a
    `UnicodeDecodeError` occurs, it falls back and tries to read the file
    again using 'latin-1' encoding, which is more permissive.

    Args:
        txt_path: The local filesystem path to the .txt file.

    Returns:
        A string containing the entire content of the file.
        Returns None if the file cannot be found or if an error occurs
        during reading, even with the fallback encoding.
    """
    try:
        logger.info(f"Attempting to open text file: {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as file:
            content = file.read()
        logger.info(f"Successfully extracted text from {txt_path}")
        return content
    except FileNotFoundError:
        logger.error(f"Text file not found: {txt_path}")
        return None
    except UnicodeDecodeError:
        logger.warning(f"Could not decode {txt_path} as UTF-8. Trying with 'latin-1'.")
        try:
            with open(txt_path, "r", encoding="latin-1") as file:  # Fallback encoding
                content = file.read()
            logger.info(
                f"Successfully extracted text from {txt_path} using latin-1 encoding."
            )
            return content
        except Exception as e:
            logger.error(f"Failed to read {txt_path} even with latin-1: {e}")
            return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while processing text file {txt_path}: {e}"
        )
        return None
