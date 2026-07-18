"""Async speech-to-text over OpenAI's audio transcription API.

Bot-agnostic: the caller hands over already-downloaded audio bytes (or a path)
and gets back the transcript text. No Telegram code here — the original n8n
bots downloaded the voice OGG/Opus first, then transcribed via Whisper; this
module is just that second step.

Auth uses ``OPENAI_API_KEY`` from the master env file (via :mod:`core.config`),
so a missing key fails loud with :class:`~core.errors.ConfigError`. Tests inject
a fake ``client`` so no real API call is ever made.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import MASTER_ENV_PATH, load
from .errors import SttError

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "whisper-1"
DEFAULT_FILENAME = "audio.ogg"


async def transcribe(
    audio: bytes | str | Path,
    *,
    filename: str = DEFAULT_FILENAME,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    api_key: str | None = None,
    client: AsyncOpenAI | None = None,
    env_path: Path = MASTER_ENV_PATH,
) -> str:
    """Transcribe audio to text via OpenAI's audio transcription API.

    Args:
        audio: Raw audio bytes (Telegram voice OGG/Opus, already downloaded) or
            a filesystem path to an audio file.
        filename: Filename hint the API uses to detect the format. Ignored when
            ``audio`` is a path (the path's own name is used instead).
        model: Transcription model id (default ``whisper-1``).
        language: Optional ISO-639-1 language code (e.g. ``"ru"``). Omitted from
            the request when ``None`` so the model auto-detects.
        api_key: Override the key. Defaults to ``OPENAI_API_KEY`` loaded from the
            master env file via :mod:`core.config`.
        client: Inject an ``AsyncOpenAI`` (or fake) for tests; when given,
            ``api_key`` resolution is skipped.
        env_path: Override for the master env file (tests pass a tmp path).

    Returns:
        The transcribed text.

    Raises:
        ConfigError: if no ``client`` is given and ``OPENAI_API_KEY`` is missing.
        SttError: if the transcription request fails.
    """
    payload, hint = _read_audio(audio, filename)
    api_client = client or _build_client(api_key, env_path)

    kwargs: dict[str, Any] = {"file": (hint, payload), "model": model}
    if language is not None:
        kwargs["language"] = language

    try:
        response = await api_client.audio.transcriptions.create(**kwargs)
    except Exception as exc:  # SDK/network/API errors -> domain error for the caller.
        raise SttError(f"transcription failed: {exc}") from exc

    return response.text


def _read_audio(audio: bytes | str | Path, filename: str) -> tuple[bytes, str]:
    """Normalize the audio input to ``(bytes, filename_hint)``."""
    if isinstance(audio, bytes):
        return audio, filename
    path = Path(audio)
    return path.read_bytes(), path.name


def _build_client(api_key: str | None, env_path: Path) -> AsyncOpenAI:
    """Construct an AsyncOpenAI client, resolving the key from config if needed."""
    from openai import AsyncOpenAI

    key = api_key or load(["OPENAI_API_KEY"], env_path=env_path).require("OPENAI_API_KEY")
    return AsyncOpenAI(api_key=key)
