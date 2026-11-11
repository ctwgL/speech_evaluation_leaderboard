import argparse
import json
import logging
import mimetypes
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = "https://openapi-gateway-azero.soundai.com"
ASR_PATH = "/api/asr/recognize"
DEFAULT_MODEL = "AzeroAsr-V1-Gasr"
API_KEY = ""
save_path = ""

def setup_logger():
    """设置日志记录器。"""
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gasr_result_{current_time}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def _response_summary(response, limit=2000):
    """返回适合写入日志的响应摘要。"""
    body = response.text.strip()
    if not body:
        return "<empty response body>"
    if len(body) > limit:
        return f"{body[:limit]}... <truncated>"
    return body


def get_gasr_result(
    file_path,
    language,
    denoise="0",
    *,
    api_key=None,
    model=DEFAULT_MODEL,
    sample_rate=16000,
    timeout=60,
    logger=None,
):
    """调用 SoundAI GASR 接口并返回与评测脚本兼容的结果。"""
    logger = logger or logging.getLogger(__name__)
    api_key = api_key or API_KEY
    if not api_key:
        raise ValueError("未配置 API Key，请设置代码中的 API_KEY")

    audio_path = Path(file_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    audio_format = audio_path.suffix.lstrip(".").lower() or "mp3"
    mime_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    request_payload = {
        "provider": "soundai",
        "model": model,
        "audioType": "file",
        "audioFormat": audio_format,
        "sampleRate": sample_rate,
        "language": language,
        "enablePunctuation": True,
        "enableITN": True,
        "extraParams": {"denoise": str(denoise)},
    }

    url = f"{BASE_URL}{ASR_PATH}"
    headers = {"Authorization": f"SaiApi {api_key}"}

    try:
        with audio_path.open("rb") as audio_file:
            response = requests.post(
                url,
                headers=headers,
                files={"file": (audio_path.name, audio_file, mime_type)},
                data={"request": json.dumps(request_payload, ensure_ascii=False)},
                timeout=timeout,
            )
    except requests.RequestException as exc:
        logger.error("请求 SoundAI ASR 失败: %s", exc)
        return None

    if not response.ok:
        logger.error(
            "SoundAI ASR 返回 HTTP %s: %s",
            response.status_code,
            _response_summary(response),
        )
        return None

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError:
        logger.error("SoundAI ASR 返回了非 JSON 响应: %s", _response_summary(response))
        return None

    if payload.get("success") is False:
        logger.error(
            "SoundAI ASR 业务请求失败，code=%s, message=%s",
            payload.get("code"),
            payload.get("message"),
        )
        return None

    result = payload.get("result")
    text = result.get("text") if isinstance(result, dict) else None
    text = text or payload.get("text")
    if not isinstance(text, str) or not text.strip():
        logger.error("SoundAI ASR 响应中缺少识别文本: %s", _response_summary(response))
        return None

    return {"result": text.strip(), "language": language}


def main():
    parser = argparse.ArgumentParser(description="语音识别转写")
    parser.add_argument("--file_path", required=True, help="音频文件路径")
    parser.add_argument("--language", required=True, help="语言类型：auto, zh, en, de...")
    parser.add_argument("--denoise", default="0", help="是否启用降噪，默认 1")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SoundAI ASR 模型名称")
    parser.add_argument("--sample_rate", type=int, default=16000, help="采样率，默认 16000")
    parser.add_argument("--timeout", type=float, default=60, help="请求超时秒数，默认 60")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("开始处理文件: %s, 语言: %s", args.file_path, args.language)

    try:
        result = get_gasr_result(
            args.file_path,
            args.language,
            args.denoise,
            model=args.model,
            sample_rate=args.sample_rate,
            timeout=args.timeout,
            logger=logger,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("转写失败: %s", exc)
        return 1

    if not result:
        logger.error("转写失败")
        return 1

    logger.info("转写结果: %s", json.dumps(result, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
