import logging
import os
import re
import threading
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import soundfile as sf

# Import lazily so the project can still run in Kokoro mode without chatterbox installed
try:
    import torch
    from chatterbox.tts import ChatterboxTTS as _ChatterboxTTS
except Exception as _e:  # noqa: N816
    _ChatterboxTTS = None
    torch = None

from utils.file_handler import ensure_dir_exists, get_safe_filename
from utils.split_text import smart_split_text

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
        # Use smart splitting instead of regex splitting
        return smart_split_text(text, split_pattern)

    def _generate_segment(
        self,
        segment_data: tuple,
    ) -> Optional[str]:
        """
        Generate audio for a single segment. Designed for parallel execution.
        segment_data: (text, output_dir, base_filename, segment_idx, audio_prompt_path, exaggeration, cfg_weight, temperature, top_p, min_p, repetition_penalty)
        """
        text, output_dir, base_filename, segment_idx, audio_prompt_path, exaggeration, cfg_weight, temperature, top_p, min_p, repetition_penalty = segment_data
        
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

        # Enforce audio prompt requirement for Chatterbox
        if curr_prompt is None:
            logger.error(
                "Chatterbox requires a reference audio_prompt_path (WAV/MP3/FLAC). Provide one to proceed."
            )
            return []

        segments = self._split_text(text, curr_split_pattern)
        if not segments:
            logger.warning(f"No text to synthesize for '{base_filename}'.")
            return []

        out_files: List[str] = []

        def _run() -> List[str]:
            local_out: List[str] = []
            
            # For parallel processing, we'll use ThreadPoolExecutor instead of ProcessPoolExecutor
            # since the model object can't be pickled for process-based parallelism
            from concurrent.futures import ThreadPoolExecutor
            
            # Prepare segment data for parallel processing
            segment_data_list = []
            for i, seg in enumerate(segments):
                segment_data = (
                    seg, output_dir, base_filename, i,
                    curr_prompt, curr_exaggeration, curr_cfg_weight,
                    curr_temperature, curr_top_p, curr_min_p, curr_rep
                )
                segment_data_list.append(segment_data)
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(len(segment_data_list), 4)) as executor:
                futures = [executor.submit(self._generate_segment, data) for data in segment_data_list]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        local_out.append(result)
            
            # Sort by segment index to maintain order
            local_out.sort(key=lambda x: int(re.search(r'_segment_(\d+)\.wav', x).group(1)))
            return local_out

        if use_lock:
            with self.tts_lock:
                out_files = _run()
        else:
            out_files = _run()

        return out_files
