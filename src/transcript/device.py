"""Device and compute-type auto-detection.

Important platform note: WhisperX runs ASR through CTranslate2 (faster-whisper),
which supports **CPU and CUDA only** — there is no Metal/MPS backend. So on Apple
Silicon we deliberately select "cpu" for the ASR stage rather than "mps", which
would raise at model-load time. CUDA (e.g. your NVIDIA 5090) is the fast path.
"""

from __future__ import annotations

import logging

log = logging.getLogger("transcript.device")


def detect_device(prefer: str | None = None) -> str:
    """Return the device string for the ASR stage: "cuda" or "cpu".

    Pass ``prefer`` to force a choice (e.g. "cuda", "cpu"). We never return
    "mps" because CTranslate2 cannot use it.
    """
    if prefer:
        if prefer == "mps":
            log.warning("MPS is not supported by WhisperX/CTranslate2; falling back to CPU.")
            return "cpu"
        return prefer

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            log.warning(
                "Detected Apple MPS, but WhisperX ASR cannot use it; using CPU instead. "
                "This works but is slower than a CUDA GPU."
            )
    except Exception as exc:  # torch not importable yet, etc.
        log.debug("Could not probe torch for devices (%s); defaulting to CPU.", exc)

    return "cpu"


def default_compute_type(device: str) -> str:
    """Pick a sensible CTranslate2 compute type for the device.

    - CUDA  -> float16 (fast, accurate on modern GPUs)
    - CPU   -> int8    (only practical option for reasonable speed)
    """
    return "float16" if device == "cuda" else "int8"
