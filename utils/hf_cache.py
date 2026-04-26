"""HuggingFace cache + offline helpers.

The Kokoro pipeline calls into `huggingface_hub` for every `KPipeline(...)` init.
By default the hub does an online HEAD revalidation per file even when the
weights are already cached, which spams logs and wastes time on every worker
start (we have N workers x 2 files = 2N HTTP HEADs per job).

Strategy:
1. Disable telemetry + the gradio/HF analytics endpoints up front.
2. On first run, prefetch the Kokoro repo into the local HF cache.
3. Once the cache contains the required files, flip the process into
   ``HF_HUB_OFFLINE=1`` so future ``KPipeline`` inits skip network calls.

This is safe: if a required file is missing offline, we fall back to online
mode automatically by clearing the env var before raising.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
KOKORO_REQUIRED_FILES = ("config.json", "kokoro-v1_0.pth")


def disable_hf_telemetry() -> None:
    """Silence HuggingFace + Gradio analytics chatter (no-op if already set)."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _voice_filename(voice: str) -> str:
    return f"voices/{voice}.pt"


def prefetch_kokoro(voice: str = "af_heart") -> bool:
    """Ensure Kokoro weights + the requested voice are cached locally.

    Returns True if the cache is complete (or was successfully populated),
    False if a network error prevented prefetch. On True, callers may safely
    enable offline mode.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - hub always present in our env
        logger.warning("huggingface_hub unavailable, skipping prefetch: %s", exc)
        return False

    targets = list(KOKORO_REQUIRED_FILES) + [_voice_filename(voice)]
    for fname in targets:
        try:
            hf_hub_download(repo_id=KOKORO_REPO_ID, filename=fname)
        except Exception as exc:
            logger.warning("Kokoro prefetch failed for %s: %s", fname, exc)
            return False
    logger.info("Kokoro cache ready (repo=%s, voice=%s).", KOKORO_REPO_ID, voice)
    return True


def enable_hf_offline() -> None:
    """Switch the current process to HF offline mode."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def configure_hf_for_kokoro(voice: str = "af_heart") -> None:
    """Convenience wrapper: silence telemetry, prefetch, then go offline.

    Safe to call from the parent process before spawning workers; child
    processes inherit the resulting env vars.
    """
    disable_hf_telemetry()
    if prefetch_kokoro(voice=voice):
        enable_hf_offline()
