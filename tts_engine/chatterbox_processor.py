import logging
import os
import re
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
    """
    Processor that wraps ResembleAI Chatterbox TTS.

    Notes:
    - Chatterbox supports English currently.
    - Voices are controlled via an optional audio_prompt_path and parameters like exaggeration/cfg_weight.
    - We segment input text ourselves using a regex pattern before calling model.generate per segment.
    """

    def __init__(self, device: Optional[str] = None):
        if _ChatterboxTTS is None:
            raise ImportError(
                "chatterbox-tts is not installed. Please `pip install chatterbox-tts` to use this engine."
            )

        self.device = device or self._autodetect_device()
        self.model = None
        self.tts_lock = threading.Lock()

        # Defaults; can be overridden by set_generation_params or per-call
        self.default_split_pattern = r"\n\n+|\r\n\r\n+|\n\s*\n+|[.!?]\s"
        self.default_audio_prompt_path: Optional[str] = None
        self.default_exaggeration: float = 0.5
        self.default_cfg_weight: float = 0.5
        self.default_temperature: float = 0.8
        self.default_top_p: float = 1.0
        self.default_min_p: float = 0.05
        self.default_repetition_penalty: float = 1.2

        self._initialize_model()

    def _autodetect_device(self) -> str:
        try:
            if torch and torch.cuda.is_available():
                return "cuda"
            if torch and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _initialize_model(self):
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
        split_pattern: Optional[str] = None,
    ) -> None:
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
        if split_pattern is not None:
            self.default_split_pattern = split_pattern

    def _split_text(self, text: str, split_pattern: str) -> List[str]:
        if not text or not text.strip():
            return []
        # Ensure we retain sentence-ending punctuation where possible by splitting on lookbehind
        # Use the provided split pattern as-is for compatibility with existing CLI
        try:
            parts = re.split(split_pattern, text)
            # re.split drops the delimiter; we don't re-attach punctuation here as the model normalizes
        except re.error:
            # Fallback: split on double newlines or sentence enders followed by space
            parts = re.split(r"\n\n+|[.!?]\s", text)
        # Clean and filter
        cleaned = [p.strip() for p in parts if p and p.strip()]
        return cleaned if cleaned else ([text.strip()] if text.strip() else [])

    def _generate_segment(
        self,
        text: str,
        output_dir: str,
        base_filename: str,
        segment_idx: int,
        *,
        audio_prompt_path: Optional[str],
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
    ) -> Optional[str]:
        try:
            wav = self.model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )
            # Convert torch tensor to numpy for soundfile
            wav_np = wav.squeeze(0).detach().cpu().numpy()
            ensure_dir_exists(output_dir)
            safe_base = get_safe_filename(base_filename)
            fpath = os.path.join(output_dir, f"{safe_base}_segment_{segment_idx:03d}.wav")
            sf.write(fpath, wav_np, self.model.sr)
            logger.info(f"Saved audio segment: {fpath}")
            return fpath
        except Exception as e:
            logger.error(f"Error generating/saving segment {segment_idx} for '{base_filename}': {e}")
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
        split_pattern: Optional[str] = None,
        use_lock: bool = True,
    ) -> List[str]:
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
        curr_split_pattern = self.default_split_pattern if split_pattern is None else split_pattern

        segments = self._split_text(text, curr_split_pattern)
        if not segments:
            logger.warning(f"No text to synthesize for '{base_filename}'.")
            return []

        out_files: List[str] = []

        def _run() -> List[str]:
            local_out: List[str] = []
            for i, seg in enumerate(segments):
                fpath = self._generate_segment(
                    seg,
                    output_dir,
                    base_filename,
                    i,
                    audio_prompt_path=curr_prompt,
                    exaggeration=curr_exaggeration,
                    cfg_weight=curr_cfg_weight,
                    temperature=curr_temperature,
                    top_p=curr_top_p,
                    min_p=curr_min_p,
                    repetition_penalty=curr_rep,
                )
                if fpath:
                    local_out.append(fpath)
            return local_out

        if use_lock:
            with self.tts_lock:
                out_files = _run()
        else:
            out_files = _run()

        return out_files
