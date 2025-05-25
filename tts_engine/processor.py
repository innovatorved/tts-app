# kokoro_tts_app/tts_engine/processor.py
import torch
import soundfile as sf
from kokoro import KPipeline
import logging
import os
from utils.file_handler import ensure_dir_exists, get_safe_filename

logger = logging.getLogger(__name__)

class KokoroTTSProcessor:
    def __init__(self, lang_code: str = 'a', device: str | None = None):
        """
        Initializes the Kokoro TTS pipeline.

        Args:
            lang_code: Language code for TTS (e.g., 'a' for American English).
                       Supported codes: 'a' (American English), 'b' (British English),
                                        'e' (Spanish), 'f' (French), 'h' (Hindi),
                                        'i' (Italian), 'j' (Japanese), 'p' (Brazilian Portuguese),
                                        'z' (Mandarin Chinese).
                       Ensure 'misaki[lang_shortcode]' is installed for non-English languages
                       (e.g., misaki[ja] for Japanese).
            device: The device to run the model on (e.g., "cuda", "cpu", "mps").
                    If None, Kokoro will attempt to auto-detect.
        """
        self.lang_code = lang_code
        self.device = device
        self.pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initializes or re-initializes the TTS pipeline."""
        logger.info(f"Initializing Kokoro TTS pipeline for lang_code='{self.lang_code}' on device='{self.device or 'auto'}'...")
        try:
            if self.device:
                if self.device == "mps" and torch.backends.mps.is_available():
                    logger.info("MPS device requested and available.")
                elif self.device == "cuda" and torch.cuda.is_available():
                    logger.info("CUDA device requested and available.")
                    # Note: For PyTorch, device selection is often handled by moving models/tensors to the device.
                    # KPipeline might not have a direct global device setter, but individual components might.
                    # Forcing torch.cuda.set_device(self.device) can sometimes influence this.
                else:
                    logger.info(f"Device '{self.device}' requested. Model will run on CPU if not available or not fully supported by Kokoro's components.")
            
            self.pipeline = KPipeline(lang_code=self.lang_code) # KPipeline itself doesn't take a device arg
            logger.info("Kokoro TTS pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS pipeline: {e}")
            logger.error("Please ensure 'espeak-ng' is installed correctly on your system.")
            logger.error("For Linux: sudo apt-get install espeak-ng")
            logger.error("For Windows: Download and run the installer from espeak-ng releases on GitHub.")
            logger.error("For MacOS: brew install espeak-ng / sudo port install espeak-ng") # Corrected brew command
            if self.lang_code not in ['a', 'b']: 
                logger.error(f"For lang_code '{self.lang_code}', ensure you have 'misaki[{self.lang_code_to_misaki_ext()}]' installed (e.g., pip install misaki[ja]).")
            raise

    def lang_code_to_misaki_ext(self) -> str:
        """Maps Kokoro lang_code to misaki extension if applicable."""
        mapping = {'j': 'ja', 'z': 'zh'} 
        return mapping.get(self.lang_code, 'en')


    def text_to_speech(self, text: str, output_dir: str,
                       base_filename: str = "audio_segment",
                       voice: str = 'af_heart', speed: float = 1.0,
                       split_pattern: str = r'\n\n+|\r\n\r\n+|\n\s*\n+|[.!?]\s') -> list[str]:
        """
        Converts text to speech and saves audio files.

        Args:
            text: The text to convert.
            output_dir: Directory to save the audio files.
            base_filename: Base name for the output WAV files.
            voice: The voice to use (e.g., 'af_heart').
            speed: Speech speed (1.0 is normal).
            split_pattern: Regex pattern to split long text into manageable chunks.
                           Default splits by double newlines or sentence-ending punctuation.

        Returns:
            A list of paths to the generated audio files.
        """
        if not self.pipeline:
            logger.error("TTS Pipeline not initialized. Cannot generate audio.")
            return []

        if not text or not text.strip():
            logger.warning("Input text is empty. No audio will be generated.")
            return []

        ensure_dir_exists(output_dir)
        safe_base_filename = get_safe_filename(base_filename)
        
        generated_files = []
        logger.info(f"Starting TTS generation for voice='{voice}', speed={speed}.")
        logger.debug(f"Input text snippet: '{text[:100]}...'")
        logger.debug(f"Using split pattern: '{split_pattern}'")

        try:
            generator = self.pipeline(
                text,
                voice=voice,
                speed=speed,
                split_pattern=split_pattern
            )

            for i, (graphemes, phonemes, audio_data) in enumerate(generator):
                output_path = os.path.join(output_dir, f"{safe_base_filename}_{i:03d}.wav")
                try:
                    sf.write(output_path, audio_data, 24000)
                    generated_files.append(output_path)
                    logger.info(f"Saved audio segment: {output_path}")
                    logger.debug(f"Segment {i}: Graphemes: '{graphemes[:50]}...', Phonemes: '{phonemes[:50]}...'")
                except Exception as e:
                    logger.error(f"Error writing audio file {output_path}: {e}")
            
            if not generated_files:
                logger.warning("No audio segments were generated. The input text might be too short or result in no processable chunks.")

        except RuntimeError as e:
            if "espeak" in str(e).lower():
                logger.error(f"RuntimeError related to espeak: {e}")
                logger.error("This often indicates 'espeak-ng' is not installed or not found in PATH.")
            else:
                logger.error(f"An unexpected RuntimeError occurred during TTS generation: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during TTS generation: {e}")

        return generated_files