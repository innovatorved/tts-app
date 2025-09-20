import torch
import soundfile as sf
from kokoro import KPipeline
import logging
import os
import threading

from utils.file_handler import ensure_dir_exists, get_safe_filename

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

    def set_generation_params(self, voice: str, speed: float, split_pattern: str | None = None):
        """Sets default generation parameters. split_pattern retained for backward compatibility (ignored)."""
        self.default_voice = voice
        self.default_speed = speed
        logger.debug(
            f"Processor generation params set: voice={voice}, speed={speed} (external splitting)"
        )

    def _generate_audio_core(
        self,
        text: str,
        output_dir: str,
        base_filename: str,
        voice: str,
        speed: float,
    ) -> list[str]:
        """Core TTS generation logic for a SINGLE already-split text segment."""
        if not self.pipeline:
            logger.error("TTS Pipeline not initialized. Cannot generate audio.")
            return []

        if not text or not text.strip():
            logger.warning(
                f"Input text for '{base_filename}' is empty. No audio will be generated."
            )
            return []

        ensure_dir_exists(output_dir)
        safe_base_filename = get_safe_filename(base_filename)
        logger.info(
            f"Thread {threading.get_ident()}: Generating audio for '{safe_base_filename}', voice='{voice}', speed={speed}."
        )
        output_path = os.path.join(output_dir, f"{safe_base_filename}.wav")
        try:
            generator = self.pipeline(
                text.strip(), voice=voice, speed=speed, split_pattern=r"(?!.*)"
            )
            for i, (graphemes, phonemes, audio_data) in enumerate(generator):
                sf.write(output_path, audio_data, 24000)
                break
            if os.path.exists(output_path):
                return [output_path]
            else:
                logger.warning(f"No audio generated for '{safe_base_filename}'.")
                return []
        except Exception as e:
            logger.error(
                f"Thread {threading.get_ident()}: Error generating audio for '{safe_base_filename}': {e}"
            )
            return []

    def text_to_speech(
        self,
        text: str,
        output_dir: str,
        base_filename: str = "audio_segment",
        voice: str | None = None,
        speed: float | None = None,
        use_lock: bool = True,
    ) -> list[str]:
        """
        Converts text to speech and saves audio files. Uses a lock for thread-safety if specified.
    Input text is treated as a single pre-split segment; external code handles segmentation.
        """
        current_voice = voice if voice is not None else self.default_voice
        current_speed = speed if speed is not None else self.default_speed
        if use_lock:
            with self.tts_lock:
                return self._generate_audio_core(
                    text, output_dir, base_filename, current_voice, current_speed
                )
        else:
            return self._generate_audio_core(
                text, output_dir, base_filename, current_voice, current_speed
            )
