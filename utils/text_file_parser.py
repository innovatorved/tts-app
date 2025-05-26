import logging

logger = logging.getLogger(__name__)


def extract_text_from_txt(txt_path: str) -> str | None:
    """
    Reads text content from a plain text file.

    Args:
        txt_path: Path to the .txt file.

    Returns:
        A string containing all text from the file, or None if an error occurs.
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
