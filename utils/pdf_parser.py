import PyPDF2
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str | None:
    """
    Extracts text from all pages of a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A string containing all extracted text, or None if an error occurs.
    """
    try:
        logger.info(f"Attempting to open PDF: {pdf_path}")
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = []
            num_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {num_pages} pages.")
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text_content.append(page.extract_text())
                logger.debug(f"Extracted text from page {page_num + 1}")

            full_text = "\n".join(
                filter(None, text_content)
            )  # Filter out None if a page has no text
            logger.info(f"Successfully extracted text from {pdf_path}")
            return full_text
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        return None
    except PyPDF2.errors.PdfReadError:
        logger.error(
            f"Could not read PDF (possibly corrupted or password-protected without password): {pdf_path}"
        )
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while processing PDF {pdf_path}: {e}"
        )
        return None
