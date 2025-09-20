import logging
import os
import threading
from typing import List, Optional

import soundfile as sf

# Import lazily so the project can still run in Kokoro mode without chatterbox installed
try:
    import torch
    from chatterbox.tts import ChatterboxTTS as _ChatterboxTTS
except Exception as _e:  # noqa: N816
    _ChatterboxTTS = None
    torch = None

from utils.file_handler import ensure_dir_exists, get_safe_filename

logger = logging.getLogger(__name__)


class ChatterboxTTSProcessor:
    """A thread-safe wrapper for the Chatterbox Text-to-Speech (TTS) engine.

    This class manages the Chatterbox TTS model, providing an interface for
    generating high-quality, expressive speech. It handles model initialization,
    device auto-detection, and thread-safe audio generation. Voice cloning is
    controlled via a reference audio prompt.

    Attributes:
        device (str): The compute device ('cuda', 'mps', or 'cpu').
        model (_ChatterboxTTS | None): The loaded Chatterbox model instance.
        tts_lock (threading.Lock): A lock for thread-safe TTS operations.
        default_audio_prompt_path (str | None): Default path to the reference
            audio file for voice cloning.
        default_exaggeration (float): Default emotional intensity control.
        default_cfg_weight (float): Default guidance weight.
        default_temperature (float): Default sampling temperature.
        default_top_p (float): Default nucleus sampling probability.
        default_min_p (float): Default minimum nucleus sampling probability.
        default_repetition_penalty (float): Default repetition penalty.
    """

    def __init__(self, device: Optional[str] = None):
        """Initializes the ChatterboxTTSProcessor.

        Checks for Chatterbox installation, auto-detects the best available
        compute device, and loads the TTS model.

        Args:
            device: The compute device to use ('cuda', 'mps', or 'cpu'). If
                None, it will be auto-detected.

        Raises:
            ImportError: If the 'chatterbox-tts' library is not installed.
        """
        if _ChatterboxTTS is None:
            raise ImportError(
                "chatterbox-tts is not installed. Please `pip install chatterbox-tts` to use this engine."
            )

        self.device = device or self._autodetect_device()
        self.model = None
        self.tts_lock = threading.Lock()

    # Defaults; can be overridden by set_generation_params or per-call
        self.default_audio_prompt_path: Optional[str] = None
        self.default_exaggeration: float = 0.5
        self.default_cfg_weight: float = 0.5
        self.default_temperature: float = 0.8
        self.default_top_p: float = 1.0
        self.default_min_p: float = 0.05
        self.default_repetition_penalty: float = 1.2

        self._initialize_model()

    def _autodetect_device(self) -> str:
        """Auto-detects the best available torch device.

        Returns:
            A string ('cuda', 'mps', or 'cpu') representing the best
            available device.
        """
        try:
            if torch and torch.cuda.is_available():
                return "cuda"
            if torch and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _initialize_model(self):
        """Loads the Chatterbox TTS model from pretrained weights.

        Raises:
            Exception: If the model fails to load.
        """
        logger.info(f"Initializing Chatterbox TTS on device='{self.device}'.")
        try:
            self.model = _ChatterboxTTS.from_pretrained(device=self.device)
            logger.info("Chatterbox TTS model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load Chatterbox model: {e}")
            raise

    def set_generation_params(
        self,
        *,
        audio_prompt_path: Optional[str] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> None:
        """Sets the default parameters for audio generation.

        Args:
            audio_prompt_path: Path to a reference audio file (WAV/MP3/FLAC)
                to guide the voice.
            exaggeration: Emotion/intensity control (e.g., 0.5).
            cfg_weight: Guidance weight (e.g., 0.5).
            temperature: Sampling temperature for generation.
            top_p: Nucleus sampling `p` value.
            min_p: Minimum nucleus sampling `p` value.
            repetition_penalty: Penalty for repeating tokens.
        """
        if audio_prompt_path is not None:
            self.default_audio_prompt_path = audio_prompt_path
        if exaggeration is not None:
            self.default_exaggeration = float(exaggeration)
        if cfg_weight is not None:
            self.default_cfg_weight = float(cfg_weight)
        if temperature is not None:
            self.default_temperature = float(temperature)
        if top_p is not None:
            self.default_top_p = float(top_p)
        if min_p is not None:
            self.default_min_p = float(min_p)
        if repetition_penalty is not None:
            self.default_repetition_penalty = float(repetition_penalty)

    def _generate_single(
        self,
        *,
        text: str,
        output_dir: str,
        base_filename: str,
        audio_prompt_path: str,
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
    ) -> Optional[str]:
        """Generates audio for a single text segment and returns the file path.

        This internal method performs the core TTS generation, saving the
        resulting audio to a WAV file.

        Args:
            text: The text segment to convert to speech.
            output_dir: The directory to save the audio file.
            base_filename: The base name for the output file.
            audio_prompt_path: Path to the reference audio for voice cloning.
            exaggeration: Emotion/intensity control.
            cfg_weight: Guidance weight.
            temperature: Sampling temperature.
            top_p: Nucleus sampling `p` value.
            min_p: Minimum nucleus sampling `p` value.
            repetition_penalty: Repetition penalty.

        Returns:
            The full path to the generated WAV file, or None if generation fails.
        """
        if not text or not text.strip():
            logger.warning(f"Empty text for base '{base_filename}'. Skipping.")
            return None
        try:
            wav = self.model.generate(
                text.strip(),
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )
            wav_np = wav.squeeze(0).detach().cpu().numpy()
            ensure_dir_exists(output_dir)
            safe_base = get_safe_filename(base_filename)
            fpath = os.path.join(output_dir, f"{safe_base}.wav")
            sf.write(fpath, wav_np, self.model.sr)
            logger.info(f"Saved audio file: {fpath}")
            return fpath
        except Exception as e:
            logger.error(f"Error generating audio for '{base_filename}': {e}")
            return None

    def text_to_speech(
        self,
        *,
        text: str,
        output_dir: str,
        base_filename: str = "audio_segment",
        audio_prompt_path: Optional[str] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        use_lock: bool = True,
        pre_split: bool = True,  # ignored; always single segment now
    ) -> List[str]:
        """Converts a text segment to speech using the Chatterbox engine.

        This is the main public method for audio generation. It resolves all
        generation parameters (using defaults if not provided), ensures a
        reference audio prompt is available, and performs the generation in a
        thread-safe manner.

        Args:
            text: The pre-split text segment to convert.
            output_dir: The directory to save the output audio file.
            base_filename: The base name for the output file.
            audio_prompt_path: Specific audio prompt to use for this call.
            exaggeration: Specific exaggeration value for this call.
            cfg_weight: Specific CFG weight for this call.
            temperature: Specific temperature for this call.
            top_p: Specific top_p for this call.
            min_p: Specific min_p for this call.
            repetition_penalty: Specific repetition penalty for this call.
            use_lock: If True, uses a lock to ensure thread-safety.
            pre_split: Ignored; retained for compatibility. Assumes text is
                a single segment.

        Returns:
            A list containing the path to the generated audio file, or an
            empty list on failure (e.g., model not initialized, no audio prompt).
        """
        if not self.model:
            logger.error("Chatterbox model not initialized.")
            return []

        # Resolve current params
        curr_prompt = audio_prompt_path if audio_prompt_path is not None else self.default_audio_prompt_path
        curr_exaggeration = self.default_exaggeration if exaggeration is None else float(exaggeration)
        curr_cfg_weight = self.default_cfg_weight if cfg_weight is None else float(cfg_weight)
        curr_temperature = self.default_temperature if temperature is None else float(temperature)
        curr_top_p = self.default_top_p if top_p is None else float(top_p)
        curr_min_p = self.default_min_p if min_p is None else float(min_p)
        curr_rep = self.default_repetition_penalty if repetition_penalty is None else float(repetition_penalty)
        # Enforce audio prompt requirement for Chatterbox
        if curr_prompt is None:
            logger.error(
                "Chatterbox requires a reference audio_prompt_path (WAV/MP3/FLAC). Provide one to proceed."
            )
            return []
        def _run_single() -> List[str]:
            result = self._generate_single(
                text=text,
                output_dir=output_dir,
                base_filename=base_filename,
                audio_prompt_path=curr_prompt,
                exaggeration=curr_exaggeration,
                cfg_weight=curr_cfg_weight,
                temperature=curr_temperature,
                top_p=curr_top_p,
                min_p=curr_min_p,
                repetition_penalty=curr_rep,
            )
            return [result] if result else []

        if use_lock:
            with self.tts_lock:
                return _run_single()
        else:
            return _run_single()
