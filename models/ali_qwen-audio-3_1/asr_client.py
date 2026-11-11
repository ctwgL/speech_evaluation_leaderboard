"""Qwen Audio 3.0 ASR client compatible with ASRResultConsumer.

Usage with the consumer::

    python scripts/asr_result_provider.py \
      --client models.ali_qwen-audio-3_0.asr_client:recognize \
      --dataset MINI_ZH --output-root /path/to/results
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import wave
from pathlib import Path
from typing import Any

import requests


URL = "https://maas.qianwenaiapi.com/api/v1/services/aigc/multimodal-generation/generation"
DEFAULT_MODEL = "qwen-audio-3.1-asr-flash"


def _transcript(payload: dict[str, Any]) -> str:
    """Extract recognized text from a DashScope multimodal response."""
    output = payload.get("output")
    candidates: list[Any] = [payload.get("text")]
    if isinstance(output, dict):
        candidates.append(output.get("text"))
        choices = output.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict):
                        candidates.append(message.get("content"))
    for content in candidates:
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            texts = [part["text"].strip() for part in content
                     if isinstance(part, dict) and isinstance(part.get("text"), str)
                     and part["text"].strip()]
            if texts:
                return " ".join(texts)
    raise ValueError("Qwen ASR 响应缺少识别文本")


def recognize(
    file_path: str | Path,
    language: str = "auto",
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = 120,
    api_key: str | None = None,
) -> dict[str, str]:
    """Recognize a WAV file and return ``{'text': transcript}``.

    ``language`` is accepted for compatibility with the shared consumer.
    Qwen Audio detects the language automatically for this request format.
    """
    audio_path = Path(file_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError(f"Qwen Audio 当前请求格式仅支持 WAV 音频: {audio_path}")
    key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("缺少 DASHSCOPE_API_KEY 环境变量")

    with wave.open(str(audio_path), "rb") as stream:
        sample_rate = stream.getframerate()
    data_uri = "data:audio/wav;base64," + base64.b64encode(audio_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [{
            "type": "input_audio", "input_audio": {"data": data_uri},
        }]}]},
        "parameters": {"format": "wav", "sample_rate": str(sample_rate)},
    }
    response = requests.post(
        URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "X-DashScope-SSE": "disable"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        body = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise ValueError("Qwen ASR 返回非 JSON 响应") from exc
    if not isinstance(body, dict):
        raise ValueError("Qwen ASR 响应格式无效")
    if body.get("code") or body.get("success") is False:
        raise RuntimeError(f"Qwen ASR 请求失败: {body.get('code')}: {body.get('message')}")
    return {"text": _transcript(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description="调用 Qwen Audio 3.0 ASR")
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    result = recognize(args.file_path, args.language, model=args.model, timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
