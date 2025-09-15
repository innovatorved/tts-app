import torch
import soundfile as sf
from kokoro import KPipeline
import logging
import os
import threading

from utils.file_handler import ensure_dir_exists, get_safe_filename
from utils.split_text import smart_split_text

logger = logging.getLogger(__name__)


class KokoroTTSProcessor:
    def __init__(self, lang_code: str = "a", device: str | None = None):
        """
        Initializes the Kokoro TTS pipeline.
        """
        self.lang_code = lang_code
        self.device = device
        self.pipeline = None
        self._initialize_pipeline()
        self.tts_lock = (
            threading.Lock()
        )  # <--- Added: Lock for thread-safe TTS operations

        # Default generation parameters, can be updated via set_generation_params
        self.default_voice: str = "af_heart"
        self.default_speed: float = 1.0
        self.default_split_pattern: str = r"\n\n+|\r\n\r\n+|\n\s*\n+|[.!?]\s"

    def _initialize_pipeline(self):
        """Initializes or re-initializes the TTS pipeline."""
        logger.info(
            f"Initializing Kokoro TTS pipeline for lang_code='{self.lang_code}' on device='{self.device or 'auto'}'..."
        )
        try:
            if self.device:
                if self.device == "mps" and torch.backends.mps.is_available():
                    logger.info("MPS device requested and available.")
                elif self.device == "cuda" and torch.cuda.is_available():
                    logger.info("CUDA device requested and available.")
                    if (
                        torch.cuda.is_available()
                    ):  # Ensure cuda is truly available before trying to set
                        try:
                            torch.cuda.set_device(self.device)
                        except Exception as e:
                            logger.warning(
                                f"Could not explicitly set CUDA device {self.device}, PyTorch will manage: {e}"
                            )
                else:
                    logger.info(
                        f"Device '{self.device}' requested. Model will run on CPU if not available/supported or auto-detected."
                    )

            self.pipeline = KPipeline(lang_code=self.lang_code)
            logger.info("Kokoro TTS pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS pipeline: {e}")
            logger.error(
                "Please ensure 'espeak-ng' is installed correctly on your system."
            )
            if self.lang_code not in ["a", "b"]:
                logger.error(
                    f"For lang_code '{self.lang_code}', ensure you have 'misaki[{self.lang_code_to_misaki_ext()}]' installed."
                )
            raise

    def lang_code_to_misaki_ext(self) -> str:
        mapping = {"j": "ja", "z": "zh"}
        return mapping.get(self.lang_code, "en")

    def set_generation_params(self, voice: str, speed: float, split_pattern: str):
        """Sets default generation parameters for the processor instance."""
        self.default_voice = voice
        self.default_speed = speed
        self.default_split_pattern = split_pattern
        logger.debug(
            f"Processor generation params set: voice={voice}, speed={speed}, split_pattern='{split_pattern}'"
        )

    def _generate_audio_core(
        self,
        text: str,
        output_dir: str,
        base_filename: str,
        voice: str,
        speed: float,
        split_pattern: str,
    ) -> list[str]:
        """Core TTS generation logic, called internally."""
        if not self.pipeline:
            logger.error("TTS Pipeline not initialized. Cannot generate audio.")
            return []

        if not text or not text.strip():
            logger.warning(
                f"Input text for '{base_filename}' is empty. No audio will be generated."
            )
            return []

        # Apply smart splitting
        segments = smart_split_text(text, split_pattern)
        if not segments:
            logger.warning(f"No segments to process for '{base_filename}'.")
            return []

        ensure_dir_exists(output_dir)
        safe_base_filename = get_safe_filename(base_filename)

        generated_files = []
        logger.info(
            f"Thread {threading.get_ident()}: Starting TTS for '{safe_base_filename}', voice='{voice}', speed={speed}."
        )

        segment_index = 0
        for segment in segments:
            try:
                generator = self.pipeline(
                    segment, voice=voice, speed=speed, split_pattern=r"(?!.*)"  # Disable internal splitting
                )

                for i, (graphemes, phonemes, audio_data) in enumerate(generator):
                    # Segment filename now includes chunk info (from base_filename) and segment index
                    output_path = os.path.join(
                        output_dir, f"{safe_base_filename}_segment_{segment_index:03d}.wav"
                    )
                    try:
                        sf.write(output_path, audio_data, 24000)
                        generated_files.append(output_path)
                        logger.info(
                            f"Thread {threading.get_ident()}: Saved audio segment: {output_path}"
                        )
                        segment_index += 1
                    except Exception as e:
                        logger.error(
                            f"Thread {threading.get_ident()}: Error writing audio file {output_path}: {e}"
                        )

            except RuntimeError as e:
                if "espeak" in str(e).lower():
                    logger.error(
                        f"Thread {threading.get_ident()}: RuntimeError related to espeak for '{safe_base_filename}': {e}"
                    )
                else:
                    logger.error(
                        f"Thread {threading.get_ident()}: An unexpected RuntimeError occurred during TTS for '{safe_base_filename}': {e}"
                    )
            except Exception as e:
                logger.error(
                    f"Thread {threading.get_ident()}: An unexpected error occurred during TTS for '{safe_base_filename}': {e}"
                )

        if not generated_files:
            logger.warning(
                f"Thread {threading.get_ident()}: No audio segments were generated for '{safe_base_filename}'. The input text might be too short or result in no processable chunks with the current split_pattern."
            )

        return generated_files

    def text_to_speech(
        self,
        text: str,
        output_dir: str,
        base_filename: str = "audio_segment",
        voice: str | None = None,
        speed: float | None = None,
        split_pattern: str | None = None,
        use_lock: bool = True,
    ) -> list[str]:
        """
        Converts text to speech and saves audio files. Uses a lock for thread-safety if specified.
        """
        current_voice = voice if voice is not None else self.default_voice
        current_speed = speed if speed is not None else self.default_speed
        current_split_pattern = (
            split_pattern if split_pattern is not None else self.default_split_pattern
        )

        if use_lock:
            with self.tts_lock:
                # logger.debug(f"Thread {threading.get_ident()} acquired TTS lock for base_filename: {base_filename}")
                result = self._generate_audio_core(
                    text,
                    output_dir,
                    base_filename,
                    current_voice,
                    current_speed,
                    current_split_pattern,
                )
                # logger.debug(f"Thread {threading.get_ident()} released TTS lock for base_filename: {base_filename}")
                return result
        else:
            # Direct call without locking, useful for single-threaded scenarios
            return self._generate_audio_core(
                text,
                output_dir,
                base_filename,
                current_voice,
                current_speed,
                current_split_pattern,
            )
