import argparse
import logging
import os
import sys
from datetime import datetime
import re
import concurrent.futures

import natsort

# Adjust path to import from sibling directories
# Adjust path to ensure the app's root directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tts_engine.processor import KokoroTTSProcessor
from tts_engine.chatterbox_processor import ChatterboxTTSProcessor
from utils.pdf_parser import extract_text_from_pdf
from utils.file_handler import ensure_dir_exists
from utils.text_file_parser import extract_text_from_txt
from utils.audio_merger import merge_audio_files
from utils.conversation_parser import (
    extract_conversation_from_text,
    get_voice_for_speaker,
)
from utils.split_text import split_text_into_chunks

# --- Logging Setup ---
# Basic configuration for logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
# ---


def main():
    parser = argparse.ArgumentParser(
        description="TTS: Text/PDF/Conversation to Audio (Kokoro or Chatterbox)"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", type=str, help="Direct text to convert to speech."
    )
    input_group.add_argument(
        "--pdf", type=str, help="Path to a PDF file to convert to speech."
    )
    input_group.add_argument(
        "--text_file",
        type=str,
        help="Path to a plain text file (e.g., .txt) to convert to speech.",
    )
    input_group.add_argument(
        "--conversation",
        type=str,
        help="Path to a text file containing a conversation between Man: and Woman: to convert with different voices.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_audio",
        help="Directory to save the generated audio files (default: ./output_audio).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="kokoro",
        choices=["kokoro", "chatterbox"],
        help="TTS engine to use: 'kokoro' or 'chatterbox' (default: kokoro).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="a",
        help="Language code (e.g., 'a' for American English). Default: 'a'.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="af_heart",
        help="Voice model (e.g., 'af_heart'). Default: 'af_heart'.",
    )
    parser.add_argument(
        "--male_voice",
        type=str,
        default="am_adam",
        help="Voice model for male speakers in conversation mode (e.g., 'am_adam'). Default: 'am_adam'.",
    )
    parser.add_argument(
        "--female_voice",
        type=str,
        default="af_heart",
        help="Voice model for female speakers in conversation mode (e.g., 'af_heart'). Default: 'af_heart'.",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed. Default: 1.0."
    )
    # Chatterbox-specific generation options (ignored by Kokoro)
    parser.add_argument(
        "--cb_audio_prompt",
        type=str,
        default=None,
        help="Chatterbox: path to a reference voice WAV file (audio prompt).",
    )
    parser.add_argument(
        "--cb_audio_prompt_male",
        type=str,
        default=None,
        help="Chatterbox: reference voice WAV for male speaker in conversation mode.",
    )
    parser.add_argument(
        "--cb_audio_prompt_female",
        type=str,
        default=None,
        help="Chatterbox: reference voice WAV for female speaker in conversation mode.",
    )
    parser.add_argument(
        "--cb_exaggeration",
        type=float,
        default=0.5,
        help="Chatterbox: emotion exaggeration (0.0-? typical 0.5).",
    )
    parser.add_argument(
        "--cb_cfg_weight",
        type=float,
        default=0.5,
        help="Chatterbox: classifier-free guidance weight (try 0.3-0.7).",
    )
    parser.add_argument(
        "--cb_temperature",
        type=float,
        default=0.8,
        help="Chatterbox: temperature for sampling.",
    )
    parser.add_argument(
        "--cb_top_p",
        type=float,
        default=1.0,
        help="Chatterbox: nucleus sampling top_p.",
    )
    parser.add_argument(
        "--cb_min_p",
        type=float,
        default=0.05,
        help="Chatterbox: minimum p sampler parameter.",
    )
    parser.add_argument(
        "--cb_repetition_penalty",
        type=float,
        default=1.2,
        help="Chatterbox: repetition penalty.",
    )
    parser.add_argument(
        "--output_filename_base",
        type=str,
        default=None,
        help="Base name for output audio files. Derived from input if not set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device ('cpu', 'cuda', 'mps'). Default: Kokoro auto-detects.",
    )
    parser.add_argument(
        "--split_pattern",
        type=str,
        default=r"\n\n+|\r\n\r\n+|\n\s*\n+|[.!?]\s",
        help=r"Regex for splitting text into smaller segments for TTS. Default splits by double newlines or sentence ends.",
    )
    parser.add_argument(
        "--merge_output",
        action="store_true",
        help="Merge all generated audio segments into a single WAV file.",
    )

    # --- Threading/Chunking Arguments ---
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of worker threads for PDF processing (default: 1, sequential). Recommended max: 2-4.",
    )
    parser.add_argument(
        "--paragraphs_per_chunk",
        type=int,
        default=30,
        help="Number of paragraphs to group into one chunk for threaded PDF processing (default: 30).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled.")

    text_to_process = ""
    base_filename = args.output_filename_base
    input_type = "direct_text"
    is_conversation = False
    conversation_parts = []

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
            input_type = "pdf_file"
        else:
            logger.error(f"Could not extract text from PDF: {args.pdf}. Exiting.")
            return
    elif args.text_file:
        logger.info(f"Processing TXT file: {args.text_file}")
        txt_content = extract_text_from_txt(args.text_file)
        if txt_content:
            text_to_process = txt_content
            if not base_filename:
                base_filename = os.path.splitext(os.path.basename(args.text_file))[0]
            input_type = "text_file"
        else:
            logger.error(
                f"Could not extract text from TXT file: {args.text_file}. Exiting."
            )
            return
    elif args.conversation:
        logger.info(f"Processing conversation file: {args.conversation}")
        conversation_text = extract_text_from_txt(args.conversation)
        if conversation_text:
            # Extract conversation parts but don't set text_to_process
            # as we'll handle each part separately
            conversation_parts = extract_conversation_from_text(conversation_text)
            if not conversation_parts:
                logger.error(f"No conversation parts found in: {args.conversation}")
                return

            if not base_filename:
                base_filename = os.path.splitext(os.path.basename(args.conversation))[0]
            input_type = "conversation"
            is_conversation = True
            logger.info(
                f"Found {len(conversation_parts)} conversation parts to process"
            )
        else:
            logger.error(
                f"Could not extract text from conversation file: {args.conversation}. Exiting."
            )
            return

    if not is_conversation and not text_to_process.strip():
        logger.warning("No text content to process. Exiting.")
        return

    ensure_dir_exists(args.output_dir)

    try:
        if args.device == "mps":
            logger.info(
                "For MPS on Apple Silicon, ensure 'PYTORCH_ENABLE_MPS_FALLBACK=1' env var is set."
            )
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Initialize engine
        if args.engine == "kokoro":
            tts_processor = KokoroTTSProcessor(lang_code=args.lang, device=args.device)
            if not is_conversation:
                tts_processor.set_generation_params(
                    voice=args.voice, speed=args.speed, split_pattern=args.split_pattern
                )
        else:  # chatterbox
            tts_processor = ChatterboxTTSProcessor(device=args.device)
            # Seed Chatterbox defaults
            tts_processor.set_generation_params(
                audio_prompt_path=args.cb_audio_prompt,
                exaggeration=args.cb_exaggeration,
                cfg_weight=args.cb_cfg_weight,
                temperature=args.cb_temperature,
                top_p=args.cb_top_p,
                min_p=args.cb_min_p,
                repetition_penalty=args.cb_repetition_penalty,
                split_pattern=args.split_pattern,
            )
    except Exception as e:
        logger.error(f"Failed to initialize TTS processor: {e}")
        return

    all_generated_files = []

    # Define voice configuration for conversation mode
    voice_config = {"Man": args.male_voice, "Woman": args.female_voice}

    # Initialize dictionaries for managing futures
    future_to_chunk_info = {}
    future_to_conv_info = {}

    if input_type == "conversation":
        # Process conversation with different voices for each speaker
        logger.info("Processing conversation mode with different voices per speaker")

        future_to_conv_info = {}

        # Process each conversation part with appropriate voice
        if (
            args.threads > 1 and len(conversation_parts) > 3
        ):  # Use threading only if there are enough parts
            logger.info(
                f"Processing {len(conversation_parts)} conversation parts with up to {args.threads} threads."
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.threads
            ) as executor:
                for i, (speaker, text) in enumerate(conversation_parts):
                    if not text.strip():
                        continue

                    # For Kokoro: choose voice by speaker; for Chatterbox: keep audio_prompt as provided
                    voice = get_voice_for_speaker(speaker, voice_config)

                    # Set base filename to include speaker and part number
                    part_base_filename = f"{base_filename}_{speaker.lower()}_{i:02d}"

                    logger.info(
                        f"Processing conversation part {i} from {speaker} with voice {voice}"
                    )

                    # Dispatch per engine
                    if args.engine == "kokoro":
                        tts_processor.set_generation_params(
                            voice=voice, speed=args.speed, split_pattern=args.split_pattern
                        )
                        future = executor.submit(
                            tts_processor.text_to_speech,
                            text=text,
                            output_dir=args.output_dir,
                            base_filename=part_base_filename,
                            voice=voice,
                            speed=args.speed,
                            split_pattern=args.split_pattern,
                            use_lock=True,
                        )
                    else:
                        # Chatterbox ignores Kokoro voice; rely on cb params
                        # Choose per-speaker prompt if provided
                        cb_prompt = (
                            args.cb_audio_prompt_male if speaker == "Man" and args.cb_audio_prompt_male
                            else args.cb_audio_prompt_female if speaker == "Woman" and args.cb_audio_prompt_female
                            else args.cb_audio_prompt
                        )
                        future = executor.submit(
                            tts_processor.text_to_speech,
                            text=text,
                            output_dir=args.output_dir,
                            base_filename=part_base_filename,
                            audio_prompt_path=cb_prompt,
                            exaggeration=args.cb_exaggeration,
                            cfg_weight=args.cb_cfg_weight,
                            temperature=args.cb_temperature,
                            top_p=args.cb_top_p,
                            min_p=args.cb_min_p,
                            repetition_penalty=args.cb_repetition_penalty,
                            split_pattern=args.split_pattern,
                            use_lock=True,
                        )
                    future_to_conv_info[future] = {
                        "id": i,
                        "speaker": speaker,
                        "name": part_base_filename,
                    }

                for future in concurrent.futures.as_completed(future_to_conv_info):
                    info = future_to_conv_info[future]
                    try:
                        generated_part_files = future.result()
                        if generated_part_files:
                            logger.info(
                                f"Conversation part {info['id']} ({info['speaker']}: {info['name']}) processed, "
                                f"{len(generated_part_files)} audio files generated."
                            )
                            all_generated_files.extend(generated_part_files)
                        else:
                            logger.warning(
                                f"Conversation part {info['id']} ({info['speaker']}) resulted in no audio files."
                            )
                    except Exception as exc:
                        logger.error(
                            f"Conversation part {info['id']} generated an exception: {exc}",
                            exc_info=True,
                        )
        else:  # Sequential processing
            # Process each conversation part in sequence
            for i, (speaker, text) in enumerate(conversation_parts):
                if not text.strip():
                    continue

                voice = get_voice_for_speaker(speaker, voice_config)

                # Set base filename to include speaker and part number
                part_base_filename = f"{base_filename}_{speaker.lower()}_{i:02d}"

                logger.info(
                    f"Processing conversation part {i} from {speaker} with voice {voice}"
                )
                if args.engine == "kokoro":
                    tts_processor.set_generation_params(
                        voice=voice, speed=args.speed, split_pattern=args.split_pattern
                    )
                    part_files = tts_processor.text_to_speech(
                        text=text,
                        output_dir=args.output_dir,
                        base_filename=part_base_filename,
                        voice=voice,
                        speed=args.speed,
                        split_pattern=args.split_pattern,
                        use_lock=False,
                    )
                else:
                    part_files = tts_processor.text_to_speech(
                        text=text,
                        output_dir=args.output_dir,
                        base_filename=part_base_filename,
                        audio_prompt_path=(
                            args.cb_audio_prompt_male if speaker == "Man" and args.cb_audio_prompt_male
                            else args.cb_audio_prompt_female if speaker == "Woman" and args.cb_audio_prompt_female
                            else args.cb_audio_prompt
                        ),
                        exaggeration=args.cb_exaggeration,
                        cfg_weight=args.cb_cfg_weight,
                        temperature=args.cb_temperature,
                        top_p=args.cb_top_p,
                        min_p=args.cb_min_p,
                        repetition_penalty=args.cb_repetition_penalty,
                        split_pattern=args.split_pattern,
                        use_lock=False,
                    )

                if part_files:
                    all_generated_files.extend(part_files)

    elif input_type in ["pdf_file", "text_file"]:
        # PDF processing can use chunking and threading
        text_chunks = split_text_into_chunks(text_to_process, args.paragraphs_per_chunk)
        if not text_chunks:
            logger.warning("PDF text was split into zero chunks. Nothing to process.")
            return

        # Store futures in a list to preserve order for merging, if needed
        futures_list = []
        future_to_chunk_info = {}

        if args.threads > 1 and len(text_chunks) > 1:
            logger.info(
                f"Processing {len(text_chunks)} PDF chunks with up to {args.threads} threads."
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.threads
            ) as executor:
                for i, chunk_text in enumerate(text_chunks):
                    # base_filename for PDF is e.g. "mydocument"
                    # chunk_base_filename becomes "mydocument_chunk_00"
                    chunk_base_filename = f"{base_filename}_chunk_{i:02d}"
                    logger.debug(
                        f"Submitting chunk {i} (base: {chunk_base_filename}) for TTS."
                    )
                    if args.engine == "kokoro":
                        future = executor.submit(
                            tts_processor.text_to_speech,
                            text=chunk_text,
                            output_dir=args.output_dir,
                            base_filename=chunk_base_filename,
                            use_lock=True,
                        )
                    else:
                        future = executor.submit(
                            tts_processor.text_to_speech,
                            text=chunk_text,
                            output_dir=args.output_dir,
                            base_filename=chunk_base_filename,
                            audio_prompt_path=args.cb_audio_prompt,
                            exaggeration=args.cb_exaggeration,
                            cfg_weight=args.cb_cfg_weight,
                            temperature=args.cb_temperature,
                            top_p=args.cb_top_p,
                            min_p=args.cb_min_p,
                            repetition_penalty=args.cb_repetition_penalty,
                            split_pattern=args.split_pattern,
                            use_lock=True,
                        )
                    future_to_chunk_info[future] = {
                        "id": i,
                        "name": chunk_base_filename,
                    }

                for future in concurrent.futures.as_completed(future_to_chunk_info):
                    info = future_to_chunk_info[future]
                    try:
                        generated_chunk_files = future.result()
                        if generated_chunk_files:
                            logger.info(
                                f"Chunk {info['id']} ({info['name']}) processed, {len(generated_chunk_files)} audio files generated."
                            )
                            all_generated_files.extend(generated_chunk_files)
                        else:
                            logger.warning(
                                f"Chunk {info['id']} ({info['name']}) resulted in no audio files."
                            )
                    except Exception as exc:
                        logger.error(
                            f"Chunk {info['id']} ({info['name']}) generated an exception: {exc}",
                            exc_info=True,
                        )
        else:  # Sequential processing for PDF (threads=1 or only one chunk)
            logger.info(f"Processing {len(text_chunks)} PDF chunks sequentially.")
            for i, chunk_text in enumerate(text_chunks):
                chunk_base_filename = f"{base_filename}_chunk_{i:02d}"
                logger.info(
                    f"Processing chunk {i} ({chunk_base_filename}) sequentially."
                )
                if args.engine == "kokoro":
                    chunk_files = tts_processor.text_to_speech(
                        text=chunk_text,
                        output_dir=args.output_dir,
                        base_filename=chunk_base_filename,
                        use_lock=False,
                    )
                else:
                    chunk_files = tts_processor.text_to_speech(
                        text=chunk_text,
                        output_dir=args.output_dir,
                        base_filename=chunk_base_filename,
                        audio_prompt_path=args.cb_audio_prompt,
                        exaggeration=args.cb_exaggeration,
                        cfg_weight=args.cb_cfg_weight,
                        temperature=args.cb_temperature,
                        top_p=args.cb_top_p,
                        min_p=args.cb_min_p,
                        repetition_penalty=args.cb_repetition_penalty,
                        split_pattern=args.split_pattern,
                        use_lock=False,
                    )
                if chunk_files:
                    all_generated_files.extend(chunk_files)

    elif input_type == "direct_text":
        # Direct text input is processed as a single block, no chunking/threading here by default
        logger.info("Processing direct text input sequentially.")
        if args.engine == "kokoro":
            generated_text_files = tts_processor.text_to_speech(
                text=text_to_process,
                output_dir=args.output_dir,
                base_filename=base_filename,
                use_lock=False,
            )
        else:
            generated_text_files = tts_processor.text_to_speech(
                text=text_to_process,
                output_dir=args.output_dir,
                base_filename=base_filename,
                audio_prompt_path=args.cb_audio_prompt,
                exaggeration=args.cb_exaggeration,
                cfg_weight=args.cb_cfg_weight,
                temperature=args.cb_temperature,
                top_p=args.cb_top_p,
                min_p=args.cb_min_p,
                repetition_penalty=args.cb_repetition_penalty,
                split_pattern=args.split_pattern,
                use_lock=False,
            )
        if generated_text_files:
            all_generated_files.extend(generated_text_files)

    if all_generated_files:
        logger.info(
            f"Successfully generated {len(all_generated_files)} audio segment(s) in total for '{base_filename}'."
        )  # Updated log

        if args.merge_output:  # Check if merging is requested
            logger.info("Preparing to merge audio segments...")

            # Sort files naturally to ensure correct order for concatenation.
            # This is crucial because segment generation order might not be sequential
            # when using threads. natsort handles "name_1.wav", "name_10.wav" correctly.
            logger.debug(
                f"Collected {len(all_generated_files)} segment files. Sorting them for merge."
            )

            # For conversation mode, we need to sort based on the conversation sequence number
            # rather than just natural sorting which would group speakers together
            if input_type == "conversation":
                # Extract conversation part number from filename (e.g., "base_man_04_segment_000.wav" -> 4)
                # Then sort by that number to maintain conversation order
                def get_conversation_order(filename):
                    # Extract the conversation part number from the filename
                    # The pattern is base_speaker_XX_segment_YYY.wav where XX is the part number
                    match = re.search(
                        r"_(?:man|woman)_(\d+)_segment", os.path.basename(filename)
                    )
                    if match:
                        return int(match.group(1))
                    return 0  # Fallback

                sorted_files_for_merging = sorted(
                    all_generated_files, key=get_conversation_order
                )
                logger.info(
                    "Conversation mode: Sorting audio segments by conversation sequence"
                )
            else:
                # For non-conversation modes, use standard natural sorting
                sorted_files_for_merging = natsort.natsorted(all_generated_files)

            if not sorted_files_for_merging:
                logger.warning("No files available to merge after attempting to sort.")
            else:
                logger.debug(
                    f"Files to be merged (in order): {sorted_files_for_merging}"
                )
                merged_filename = (
                    f"{base_filename}_merged.wav"  # Define merged file name
                )
                merged_output_path = os.path.join(args.output_dir, merged_filename)

                logger.info(
                    f"Calling audio merger for {len(sorted_files_for_merging)} files to create {merged_output_path}."
                )
                success = merge_audio_files(
                    sorted_files_for_merging, merged_output_path
                )  # Call the merge function

                if success:
                    logger.info(
                        f"All segments successfully merged into: {merged_output_path}"
                    )
                    # Add special message for conversation mode
                    if input_type == "conversation":
                        logger.info(
                            "Conversation audio has been merged with alternating voices for different speakers."
                        )
                else:
                    logger.error("Failed to merge audio segments.")
        # If not merging, individual segments are already saved.
    else:
        logger.warning("No audio files were generated overall. Check logs for details.")


if __name__ == "__main__":
    main()
