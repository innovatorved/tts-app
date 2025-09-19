import logging
from pydub import AudioSegment
import os

logger = logging.getLogger(__name__)

# Explicitly specify the path to ffmpeg if pydub has trouble finding it.
AudioSegment.converter = "/usr/bin/ffmpeg"

def merge_audio_files(audio_file_paths: list[str], output_merged_path: str) -> bool:
    """
    Merges a list of audio files (WAV) into a single audio file.

    Args:
        audio_file_paths: A list of paths to the audio files to merge.
                          The files are assumed to be in WAV format and
                          will be merged in the order they appear in the list.
        output_merged_path: The path where the merged audio file will be saved.

    Returns:
        True if merging was successful, False otherwise.
    """
    if not audio_file_paths:
        logger.warning("No audio files provided for merging.")
        return False

    logger.info(
        f"Attempting to merge {len(audio_file_paths)} audio files into {output_merged_path}."
    )

    try:
        # Ensure all files exist before starting
        for f_path in audio_file_paths:
            if not os.path.exists(f_path):
                logger.error(f"Audio file for merging not found: {f_path}")
                return False

        # Load the first audio file as the base
        combined_audio = AudioSegment.from_wav(audio_file_paths[0])
        logger.debug(f"Loaded initial segment: {audio_file_paths[0]}")

        # Append the rest of the audio files
        for i in range(1, len(audio_file_paths)):
            segment = AudioSegment.from_wav(audio_file_paths[i])
            combined_audio += segment
            logger.debug(f"Appended segment: {audio_file_paths[i]}")

        # Export the combined audio
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_merged_path)
        if output_dir and not os.path.exists(
            output_dir
        ):  # Check if output_dir is not empty string
            os.makedirs(output_dir, exist_ok=True)

        combined_audio.export(output_merged_path, format="wav")
        logger.info(f"Successfully merged audio files into: {output_merged_path}")
        return True
    except FileNotFoundError as e:  # Should be caught by pre-check, but good to have
        logger.error(f"A file was not found during merging: {e}")
        return False
    except Exception as e:
        logger.error(f"An error occurred during audio merging: {e}")
        logger.error(
            "Please ensure FFmpeg or libav is installed and accessible in your system's PATH if pydub requires it."
        )
        return False
