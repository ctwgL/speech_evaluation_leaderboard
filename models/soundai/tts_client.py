"""SoundAI AzeroTTS speech-synthesis client.

The API contract follows the official AzeroTTS documentation:
https://azero-doc.soundai.com/apis/docs/azerotts.html

The service returns a WAV stream (24 kHz, 16-bit, mono). Configure the
credential through ``SOUNDAI_API_KEY``; both a raw key and a value already
prefixed with ``SaiApi`` are accepted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests


SOUNDAI_TTS_BASE_URL = "https://openapi-gateway-azero.soundai.com"
SOUNDAI_TTS_CLONE_PATH = "/ntts-clone/v3/clone"
SOUNDAI_TTS_QUERY_PATH = "/ntts-clone/v3/query"
SOUNDAI_TTS_PATH = "/ntts-clone/v3/speech"
SOUNDAI_TTS_CLONE_URL = f"{SOUNDAI_TTS_BASE_URL}{SOUNDAI_TTS_CLONE_PATH}"
SOUNDAI_TTS_QUERY_URL = f"{SOUNDAI_TTS_BASE_URL}{SOUNDAI_TTS_QUERY_PATH}"
SOUNDAI_TTS_URL = f"{SOUNDAI_TTS_BASE_URL}{SOUNDAI_TTS_PATH}"
SOUNDAI_API_KEY_ENV = "SOUNDAI_API_KEY"

# Kept for compatibility with the other SoundAI clients. Prefer setting the
# SOUNDAI_API_KEY environment variable instead of storing a credential here.
API_KEY = ""


class SoundAITTSError(RuntimeError):
    """Raised when the SoundAI TTS request or response is invalid."""


def _authorization_value(api_key: str | None) -> str:
    value = (api_key or os.getenv(SOUNDAI_API_KEY_ENV) or API_KEY).strip()
    if not value:
        raise ValueError(
            f"未配置 SoundAI API Key，请设置环境变量 {SOUNDAI_API_KEY_ENV}"
        )
    if value.lower().startswith("saiapi "):
        return value
    return f"SaiApi {value}"


def _error_body(response: requests.Response, limit: int = 2000) -> str:
    """Return a bounded response body suitable for an exception message."""
    try:
        body = response.text.strip()
    except (AttributeError, UnicodeError):
        body = ""
    if not body:
        return "<empty response body>"
    return body if len(body) <= limit else f"{body[:limit]}... <truncated>"


def _sse_events(response: requests.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse the ``event`` and JSON ``data`` fields from an SSE response."""
    event_name = "message"
    data_lines: list[str] = []

    def emit() -> tuple[str, dict[str, Any]] | None:
        if not data_lines:
            return None
        raw_data = "\n".join(data_lines)
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise SoundAITTSError(f"SoundAI TTS 返回了无效 SSE JSON: {raw_data}") from exc
        if not isinstance(payload, dict):
            raise SoundAITTSError(f"SoundAI TTS SSE data 不是 JSON 对象: {raw_data}")
        return event_name, payload

    for raw_line in response.iter_lines(decode_unicode=True):
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, bytes)
            else raw_line
        )
        if line == "":
            item = emit()
            if item is not None:
                yield item
            event_name = "message"
            data_lines = []
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    item = emit()
    if item is not None:
        yield item


def _sse_error(payload: dict[str, Any]) -> str:
    return str(payload.get("message") or payload.get("msg") or payload)


def clone_speaker(
    audio_file_path: str | Path,
    speaker_name: str,
    *,
    emotion: str = "default",
    enable_asr: bool = True,
    language: str = "auto",
    api_key: str | None = None,
    timeout: float = 300,
    url: str = SOUNDAI_TTS_CLONE_URL,
    session: Any = requests,
) -> str:
    """Clone a reference WAV and return its SoundAI speaker identifier."""
    audio_path = Path(audio_file_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"参考音频不存在: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError(f"SoundAI 音色克隆仅支持 WAV: {audio_path}")
    if not isinstance(speaker_name, str) or not speaker_name.strip():
        raise ValueError("speaker_name 不能为空")
    if not isinstance(emotion, str) or not emotion.strip():
        raise ValueError("emotion 不能为空")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language 不能为空")
    if not isinstance(enable_asr, bool):
        raise ValueError("enable_asr 必须是布尔值")
    if timeout <= 0:
        raise ValueError("timeout 必须大于 0")

    headers = {
        "Authorization": _authorization_value(api_key),
        "Accept": "text/event-stream",
    }
    data = {
        "speaker_name": speaker_name.strip(),
        "emotion": emotion.strip(),
        "enable_asr": str(bool(enable_asr)),
        "language": language.strip(),
    }
    try:
        with audio_path.open("rb") as audio_file:
            response = session.post(
                url,
                headers=headers,
                files={"wav_file": (audio_path.name, audio_file, "audio/wav")},
                data=data,
                stream=True,
                timeout=timeout,
            )
    except requests.RequestException as exc:
        raise SoundAITTSError(f"请求 SoundAI 音色克隆失败: {exc}") from exc

    try:
        if not response.ok:
            raise SoundAITTSError(
                f"SoundAI 音色克隆返回 HTTP {response.status_code}: "
                f"{_error_body(response)}"
            )
        response.encoding = "utf-8"
        for event, payload in _sse_events(response):
            if event == "error":
                raise SoundAITTSError(f"SoundAI 音色克隆失败: {_sse_error(payload)}")
            if event == "result":
                if payload.get("status") not in (None, 200, "200"):
                    raise SoundAITTSError(
                        f"SoundAI 音色克隆失败: {_sse_error(payload)}"
                    )
                result = payload.get("data")
                if not isinstance(result, str) or not result.strip():
                    raise SoundAITTSError("SoundAI 音色克隆结果缺少 speaker_name")
                return result.strip()
        raise SoundAITTSError("SoundAI 音色克隆响应中缺少 result 事件")
    except requests.RequestException as exc:
        raise SoundAITTSError(f"读取 SoundAI 音色克隆响应失败: {exc}") from exc
    finally:
        response.close()


def speaker_exists(
    speaker_name: str,
    *,
    api_key: str | None = None,
    timeout: float = 60,
    url: str = SOUNDAI_TTS_QUERY_URL,
    session: Any = requests,
) -> bool:
    """Return whether a cloned SoundAI speaker identifier already exists."""
    if not isinstance(speaker_name, str) or not speaker_name.strip():
        raise ValueError("speaker_name 不能为空")
    if timeout <= 0:
        raise ValueError("timeout 必须大于 0")
    headers = {
        "Authorization": _authorization_value(api_key),
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/event-stream",
    }
    try:
        response = session.post(
            url,
            headers=headers,
            data={"speaker_name": speaker_name.strip()},
            stream=True,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise SoundAITTSError(f"查询 SoundAI 音色失败: {exc}") from exc

    try:
        if not response.ok:
            raise SoundAITTSError(
                f"SoundAI 音色查询返回 HTTP {response.status_code}: "
                f"{_error_body(response)}"
            )
        response.encoding = "utf-8"
        for event, payload in _sse_events(response):
            if event == "error":
                if payload.get("success") is False:
                    return False
                raise SoundAITTSError(f"SoundAI 音色查询失败: {_sse_error(payload)}")
            if event == "result":
                return payload.get("success") is True
        raise SoundAITTSError("SoundAI 音色查询响应中缺少 result/error 事件")
    except requests.RequestException as exc:
        raise SoundAITTSError(f"读取 SoundAI 音色查询响应失败: {exc}") from exc
    finally:
        response.close()


def synthesize_speech(
    speaker_name: str,
    text: str,
    output_path: str | Path,
    *,
    speed: float = 1.0,
    api_key: str | None = None,
    timeout: float = 120,
    url: str = SOUNDAI_TTS_URL,
    session: Any = requests,
    chunk_size: int = 64 * 1024,
) -> Path:
    """Synthesize ``text`` with a cloned speaker and save the returned WAV.

    ``speaker_name`` is the identifier returned by SoundAI's voice-cloning
    endpoint. The destination directory is created automatically. Audio is
    first streamed to a temporary file and atomically moved into place only
    after the whole response has been received.

    Returns the resolved output path. Network, HTTP, or invalid-response
    failures raise :class:`SoundAITTSError`; invalid local arguments raise
    :class:`ValueError`.
    """
    if not isinstance(speaker_name, str) or not speaker_name.strip():
        raise ValueError("speaker_name 不能为空")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text 不能为空")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ValueError("speed 必须是 0.5 到 2.0 之间的数字")
    if not 0.5 <= float(speed) <= 2.0:
        raise ValueError("speed 必须在 0.5 到 2.0 之间")
    if timeout <= 0:
        raise ValueError("timeout 必须大于 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".wav":
        raise ValueError(f"SoundAI 当前仅支持 WAV 输出: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": _authorization_value(api_key),
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "audio/wav, application/octet-stream",
    }
    payload = {
        "speakerName": speaker_name.strip(),
        "text": text.strip(),
        "speed": float(speed),
        "format": "wav",
    }

    try:
        response = session.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise SoundAITTSError(f"请求 SoundAI TTS 失败: {exc}") from exc

    temporary_path: Path | None = None
    try:
        if not response.ok:
            raise SoundAITTSError(
                f"SoundAI TTS 返回 HTTP {response.status_code}: "
                f"{_error_body(response)}"
            )

        content_type = response.headers.get("Content-Type", "").lower()
        if "json" in content_type or content_type.startswith("text/"):
            raise SoundAITTSError(
                "SoundAI TTS 未返回音频流: " + _error_body(response)
            )

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            size = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    output.write(chunk)
                    size += len(chunk)

        if size == 0:
            raise SoundAITTSError("SoundAI TTS 返回了空音频流")
        os.replace(temporary_path, destination)
        temporary_path = None
    except requests.RequestException as exc:
        raise SoundAITTSError(f"读取 SoundAI TTS 音频流失败: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        close = getattr(response, "close", None)
        if callable(close):
            close()

    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="调用 SoundAI AzeroTTS 合成语音")
    parser.add_argument("--speaker-name", required=True, help="音色克隆接口返回的音色名称")
    parser.add_argument("--text", required=True, help="待合成文本")
    parser.add_argument("--output-path", required=True, type=Path, help="输出 WAV 路径")
    parser.add_argument("--speed", type=float, default=1.0, help="语速，范围 0.5-2.0")
    parser.add_argument("--timeout", type=float, default=120, help="请求超时秒数")
    args = parser.parse_args()

    try:
        output_path = synthesize_speech(
            args.speaker_name,
            args.text,
            args.output_path,
            speed=args.speed,
            timeout=args.timeout,
        )
    except (OSError, ValueError, SoundAITTSError) as exc:
        logging.error("语音合成失败: %s", exc)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
