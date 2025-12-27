import warnings

# Suppress torch deprecation warnings
warnings.filterwarnings("ignore", message=".*weight_norm.*")
warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")

import torch
import soundfile as sf
from kokoro import KPipeline
import logging
import os
import threading

from utils.file_handler import ensure_dir_exists, get_safe_filename

logger = logging.getLogger(__name__)


class KokoroTTSProcessor:
    """A thread-safe wrapper for the Kokoro Text-to-Speech (TTS) pipeline.

    This class initializes and manages the Kokoro TTS engine, providing a
    simplified interface for generating audio from text. It includes features
    for device selection (CPU, CUDA, MPS), thread-safe audio generation, and
    customizable voice and speed parameters.

    Attributes:
        lang_code (str): The language code used to initialize the pipeline.
        device (str | None): The target compute device (e.g., 'cuda', 'mps').
        pipeline (KPipeline | None): The underlying Kokoro pipeline instance.
        tts_lock (threading.Lock): A lock to ensure thread-safe TTS operations.
        default_voice (str): The default voice model to use for generation.
        default_speed (float): The default speech speed multiplier.
    """

    def __init__(self, lang_code: str = "a", device: str | None = None):
        """Initializes the KokoroTTSProcessor.

        Sets up the configuration for the TTS pipeline and initializes it.

        Args:
            lang_code: The language code for Kokoro (e.g., 'a' for American
                English). Defaults to "a".
            device: The compute device to use ('cuda', 'mps', or None for auto).
                Defaults to None.
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
        """Initializes or re-initializes the TTS pipeline.

        Loads the Kokoro TTS model onto the specified device. It includes checks
        for device availability and provides informative error messages if the
        pipeline fails to load, which can happen if dependencies like 'espeak-ng'
        are missing.

        Raises:
            Exception: If the pipeline fails to initialize for any reason.
        """
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

            self.pipeline = KPipeline(lang_code=self.lang_code, repo_id='hexgrad/Kokoro-82M')
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
        """Maps a Kokoro language code to its Misaki model extension.

        Returns:
            The Misaki model extension string (e.g., 'ja', 'zh').
        """
        mapping = {"j": "ja", "z": "zh"}
        return mapping.get(self.lang_code, "en")

    def set_generation_params(self, voice: str, speed: float, split_pattern: str | None = None):
        """Sets the default parameters for audio generation.

        Args:
            voice: The name of the voice model to use.
            speed: The speech speed multiplier (e.g., 1.0 is normal).
            split_pattern: This argument is ignored and retained only for
                backward compatibility. Text splitting is handled externally.
        """
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
        """Core TTS generation logic for a single text segment.

        This internal method handles the actual audio generation for a given
        piece of text. It ensures the output directory exists and saves the
        generated audio as a WAV file.

        Args:
            text: The text segment to convert to speech.
            output_dir: The directory where the audio file will be saved.
            base_filename: The desired name for the output file, without the
                .wav extension.
            voice: The voice model to use for this specific generation.
            speed: The speech speed to use for this specific generation.

        Returns:
            A list containing the full path to the generated audio file if
            successful, or an empty list if generation fails or the input
            text is empty.
        """
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
        """Converts a text segment to speech and saves it as a WAV file.

        This method serves as the main public interface for generating audio.
        It handles parameter selection (using defaults if none are provided)
        and ensures thread-safety by using a lock during the core generation
        process. Text splitting is expected to be handled before calling this.

        Args:
            text: The pre-split text segment to convert.
            output_dir: The directory to save the output audio file.
            base_filename: The base name for the output file. Defaults to
                "audio_segment".
            voice: The specific voice to use. If None, the processor's default
                voice is used.
            speed: The specific speed to use. If None, the processor's default
                speed is used.
            use_lock: Whether to acquire the thread lock during generation.
                Should be True when calling from multiple threads.

        Returns:
            A list containing the path to the generated audio file, or an
            empty list on failure.
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
