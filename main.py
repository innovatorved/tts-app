# kokoro_tts_app/main.py
import argparse
import logging
import os
import sys
from datetime import datetime

# Adjust path to import from sibling directories
# Adjust path to ensure the app's root directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR) # Insert at the beginning for higher precedence
    
from tts_engine.processor import KokoroTTSProcessor
from utils.pdf_parser import extract_text_from_pdf
from utils.file_handler import ensure_dir_exists

# --- Logging Setup ---
# Basic configuration for logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__) # Logger for this main module
# ---

def main():
    parser = argparse.ArgumentParser(description="Kokoro-TTS Text/PDF to Audio Converter")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Direct text to convert to speech.")
    input_group.add_argument("--pdf", type=str, help="Path to a PDF file to convert to speech.")

    parser.add_argument("--output_dir", type=str, default="./output_audio",
                        help="Directory to save the generated audio files (default: ./output_audio).")
    parser.add_argument("--lang", type=str, default="a",
                        help="Language code for Kokoro-TTS (e.g., 'a' for American English, 'b' for British, 'j' for Japanese). Default: 'a'.")
    parser.add_argument("--voice", type=str, default="af_heart",
                        help="Voice model to use (e.g., 'af_heart'). Default: 'af_heart'. Check Kokoro documentation for available voices for your chosen language.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed (e.g., 0.8 for slower, 1.2 for faster). Default: 1.0.")
    parser.add_argument("--output_filename_base", type=str, default=None,
                        help="Base name for output audio files. If not provided, it's derived from PDF name or a timestamp for text input.")
    parser.add_argument("--device", type=str, default=None, choices=['cpu', 'cuda', 'mps'],
                        help="Device to run the model on ('cpu', 'cuda', 'mps'). Default: Kokoro auto-detects.")
    parser.add_argument("--split_pattern", type=str, default=r'\n\n+|\r\n\r\n+|\n\s*\n+|[.!?]\s',
                        help=r"Regex pattern to split long text. Default: r'\n\n+|\r\n\r\n+|\n\s*\n+|[.!?]\s'")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")


    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger level
        for handler in logging.getLogger().handlers: # Apply to all handlers
            handler.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled.")


    # Determine input text and base filename
    text_to_process = ""
    base_filename = args.output_filename_base

    if args.text:
        logger.info("Processing direct text input.")
        text_to_process = args.text
        if not base_filename:
            base_filename = f"text_to_speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    elif args.pdf:
        logger.info(f"Processing PDF file: {args.pdf}")
        pdf_text = extract_text_from_pdf(args.pdf)
        if pdf_text:
            text_to_process = pdf_text
            if not base_filename:
                base_filename = os.path.splitext(os.path.basename(args.pdf))[0]
        else:
            logger.error(f"Could not extract text from PDF: {args.pdf}. Exiting.")
            return

    if not text_to_process.strip():
        logger.warning("No text content to process. Exiting.")
        return

    # Initialize TTS Processor
    try:
        # For MacOS Apple Silicon GPU Acceleration, you might need to set this environment variable
        # *before* importing torch or kokoro. This is best done when launching the script:
        # PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py ...
        # We'll log a reminder if 'mps' is chosen.
        if args.device == "mps":
            logger.info("If using MPS on Apple Silicon, ensure 'PYTORCH_ENABLE_MPS_FALLBACK=1' environment variable is set for optimal performance.")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # Try setting it, but might be too late if torch already imported.

        tts_processor = KokoroTTSProcessor(lang_code=args.lang, device=args.device)
    except Exception as e:
        logger.error(f"Failed to initialize TTS processor: {e}")
        logger.error("Please check your Kokoro-TTS and espeak-ng installation.")
        return

    # Perform TTS
    logger.info(f"Generating audio. Output directory: {args.output_dir}, Base filename: {base_filename}")
    
    ensure_dir_exists(args.output_dir) # Ensure output directory exists before TTS
    
    generated_files = tts_processor.text_to_speech(
        text=text_to_process,
        output_dir=args.output_dir,
        base_filename=base_filename,
        voice=args.voice,
        speed=args.speed,
        split_pattern=args.split_pattern
    )

    if generated_files:
        logger.info(f"Successfully generated {len(generated_files)} audio file(s):")
        for f_path in generated_files:
            logger.info(f"- {f_path}")
    else:
        logger.warning("No audio files were generated. Check logs for details.")

if __name__ == "__main__":
    main()