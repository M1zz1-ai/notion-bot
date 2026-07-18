"""STT transcription: argument pass-through, key resolution, error wrapping —
all against a fake AsyncOpenAI client (no network)."""

from pathlib import Path

import pytest

from core.errors import ConfigError, SttError
from core.stt import transcribe


class _FakeTranscriptions:
    def __init__(self, recorder, *, text="hello world", raises=None):
        self._recorder = recorder
        self._text = text
        self._raises = raises

    async def create(self, **kwargs):
        self._recorder["kwargs"] = kwargs
        if self._raises is not None:
            raise self._raises
        return type("Resp", (), {"text": self._text})()


class _FakeAudio:
    def __init__(self, transcriptions):
        self.transcriptions = transcriptions


class _FakeClient:
    """Stand-in for AsyncOpenAI exposing .audio.transcriptions.create()."""

    def __init__(self, recorder, *, text="hello world", raises=None):
        self.audio = _FakeAudio(_FakeTranscriptions(recorder, text=text, raises=raises))


async def test_success_returns_text():
    rec = {}
    out = await transcribe(b"oggbytes", client=_FakeClient(rec, text="transcribed"))
    assert out == "transcribed"


async def test_missing_key_fails_loud(tmp_path):
    empty_env = tmp_path / ".env"
    empty_env.write_text("")
    # No client and no api_key -> must resolve via core.config and fail loud.
    with pytest.raises(ConfigError):
        await transcribe(b"oggbytes", env_path=empty_env)


async def test_filename_and_model_passed_through():
    rec = {}
    await transcribe(
        b"oggbytes",
        filename="voice.ogg",
        model="whisper-1",
        client=_FakeClient(rec),
    )
    file_arg = rec["kwargs"]["file"]
    assert file_arg[0] == "voice.ogg"  # (filename, bytes) tuple
    assert file_arg[1] == b"oggbytes"
    assert rec["kwargs"]["model"] == "whisper-1"


async def test_language_passed_when_given():
    rec = {}
    await transcribe(b"x", language="ru", client=_FakeClient(rec))
    assert rec["kwargs"]["language"] == "ru"


async def test_language_omitted_when_none():
    rec = {}
    await transcribe(b"x", client=_FakeClient(rec))
    assert "language" not in rec["kwargs"]


async def test_path_input_read_from_disk(tmp_path: Path):
    audio_file = tmp_path / "clip.ogg"
    audio_file.write_bytes(b"disk-audio")
    rec = {}
    await transcribe(audio_file, client=_FakeClient(rec))
    file_arg = rec["kwargs"]["file"]
    assert file_arg[1] == b"disk-audio"
    assert file_arg[0] == "clip.ogg"  # filename defaults to the path's name


async def test_sdk_error_wrapped_in_stt_error():
    rec = {}
    client = _FakeClient(rec, raises=RuntimeError("boom"))
    with pytest.raises(SttError):
        await transcribe(b"x", client=client)
