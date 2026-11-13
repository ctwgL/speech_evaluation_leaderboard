"""Consume TTS datasets, run synthesis clients, and persist JSON results."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.tts_dataset_provider import (  # noqa: E402
    DATASETS,
    DEFAULT_CV3_EVAL_ROOT,
    DEFAULT_SEED_TTS_ROOT,
    TTSDatasetProvider,
    TTSSample,
)


Synthesizer = Callable[[TTSSample], Any]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class TTSResultConsumer:
    """Synthesize every sample and save one JSON file per benchmark task."""

    def __init__(
        self,
        synthesizer: Synthesizer,
        output_root: Path | str,
        *,
        overwrite: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.synthesizer = synthesizer
        self.output_root = Path(output_root).expanduser().resolve()
        self.overwrite = overwrite
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _validate_dataset(dataset: str) -> tuple[str, str]:
        if dataset not in DATASETS:
            raise ValueError(
                f"无效语音合成数据集: {dataset!r}；可选值: {', '.join(DATASETS)}"
            )
        return tuple(dataset.split("/", 1))  # type: ignore[return-value]

    def result_path(self, dataset: str | TTSSample) -> Path:
        name = dataset.dataset if isinstance(dataset, TTSSample) else dataset
        family, subset = self._validate_dataset(name)
        return self.output_root / family / f"{subset}.json"

    def checkpoint_path(self, dataset: str | TTSSample) -> Path:
        return self.result_path(dataset).with_suffix(".checkpoint.jsonl")

    def _load_result(self, dataset: str) -> dict[str, dict[str, Any]]:
        path = self.result_path(dataset)
        if not path.is_file() or self.overwrite:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset") != dataset or not isinstance(payload.get("results"), list):
            raise ValueError(f"已有结果文件格式不正确: {path}")
        records: dict[str, dict[str, Any]] = {}
        for record in payload["results"]:
            if not isinstance(record, dict) or not record.get("sample_id"):
                raise ValueError(f"已有结果文件包含无效记录: {path}")
            records[str(record["sample_id"])] = record
        return records

    def _load_checkpoint(self, dataset: str) -> dict[str, dict[str, Any]]:
        path = self.checkpoint_path(dataset)
        if not path.is_file():
            return {}
        records: dict[str, dict[str, Any]] = {}
        valid_lines: list[str] = []
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if line_number == len(lines):
                    self.logger.warning("忽略未完整写入的最后一条 checkpoint: %s", path)
                    path.write_text(
                        ("\n".join(valid_lines) + "\n") if valid_lines else "",
                        encoding="utf-8",
                    )
                    break
                raise ValueError(
                    f"checkpoint 第 {line_number} 行不是有效 JSON: {path}"
                )
            if (
                not isinstance(record, dict)
                or record.get("dataset") != dataset
                or not record.get("sample_id")
            ):
                raise ValueError(f"checkpoint 第 {line_number} 行格式不正确: {path}")
            records[str(record["sample_id"])] = record
            valid_lines.append(line)
        return records

    def _append_checkpoint(self, dataset: str, record: dict[str, Any]) -> None:
        path = self.checkpoint_path(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _base_record(sample: TTSSample) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "family": sample.family,
            "subset": sample.subset,
            "language": sample.language,
            "target_text": sample.target_text,
            "prompt_text": sample.prompt_text,
            "prompt_audio_path": str(sample.prompt_audio_path),
            "ground_truth_audio_path": (
                str(sample.ground_truth_audio_path)
                if sample.ground_truth_audio_path is not None
                else None
            ),
            "generated_audio_path": (
                str(sample.generated_audio_path)
                if sample.generated_audio_path is not None
                else None
            ),
            "metadata": dict(sample.metadata),
            "result": None,
            "status": "failed",
        }

    @staticmethod
    def _completed(record: dict[str, Any] | None, sample: TTSSample) -> bool:
        if not record or record.get("status") != "success":
            return False
        output = sample.generated_audio_path
        if output is None or record.get("generated_audio_path") != str(output):
            return False
        return output.is_file() and output.stat().st_size > 0

    def _synthesize(self, sample: TTSSample) -> dict[str, Any]:
        record = self._base_record(sample)
        if sample.generated_audio_path is None:
            record["error"] = "数据集 provider 未配置 generated_root"
            return record
        if not sample.prompt_audio_path.is_file():
            record["error"] = f"参考音频不存在: {sample.prompt_audio_path}"
            return record

        try:
            result = self.synthesizer(sample)
            output = sample.generated_audio_path
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"客户端未生成有效的合成音频: {output}")
            record["result"] = result
            record["status"] = "success"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.logger.exception("处理 TTS 样本失败: %s", sample.sample_id)
        return record

    def consume_dataset(
        self,
        provider: TTSDatasetProvider,
        dataset: str,
        *,
        limit: int | None = None,
    ) -> dict[str, int]:
        spec = provider.get_dataset_spec(dataset)
        path = self.result_path(dataset)
        checkpoint = self.checkpoint_path(dataset)
        previous = self._load_result(dataset)
        if self.overwrite:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("", encoding="utf-8")
        recovered = self._load_checkpoint(dataset)
        previous.update(recovered)

        stats = {"success": 0, "failed": 0, "skipped": 0}
        results: list[dict[str, Any]] = []
        processed_ids: set[str] = set()
        self.logger.info(
            "开始处理 TTS 数据集 %s，已有结果 %d 条（checkpoint 恢复 %d 条），输出: %s",
            dataset,
            len(previous),
            len(recovered),
            path,
        )

        for index, sample in enumerate(provider.iter_samples(dataset, limit=limit), 1):
            processed_ids.add(sample.sample_id)
            old = previous.get(sample.sample_id)
            if self._completed(old, sample):
                results.append(old)
                stats["skipped"] += 1
                self.logger.info(
                    "[%s] 第 %d 条，样本 %s：跳过已有成功结果",
                    dataset,
                    index,
                    sample.sample_id,
                )
                continue

            self.logger.info(
                "[%s] 第 %d 条，样本 %s：开始合成，prompt=%s",
                dataset,
                index,
                sample.sample_id,
                sample.prompt_audio_path,
            )
            record = self._synthesize(sample)
            self._append_checkpoint(dataset, record)
            results.append(record)
            stats["success" if record["status"] == "success" else "failed"] += 1
            self.logger.info(
                "[%s] 第 %d 条，样本 %s：%s%s",
                dataset,
                index,
                sample.sample_id,
                record["status"],
                f"，原因: {record['error']}" if record.get("error") else "",
            )

        for sample_id, old in previous.items():
            if sample_id not in processed_ids:
                results.append(old)
                stats["skipped" if old.get("status") == "success" else "failed"] += 1

        _atomic_json(
            path,
            {
                "dataset": dataset,
                "family": spec.family,
                "subset": spec.subset,
                "generated_root": str(provider.generated_root),
                "results": results,
                "stats": stats,
            },
        )
        checkpoint.unlink(missing_ok=True)
        self.logger.info(
            "TTS 数据集 %s 完成：成功 %d，失败 %d，跳过 %d，结果: %s",
            dataset,
            stats["success"],
            stats["failed"],
            stats["skipped"],
            path,
        )
        return stats

    def consume(
        self,
        provider: TTSDatasetProvider,
        datasets: Iterable[str],
        *,
        limit: int | None = None,
    ) -> dict[str, dict[str, int]]:
        return {
            dataset: self.consume_dataset(provider, dataset, limit=limit)
            for dataset in datasets
        }


def _audio_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _speaker_name(prefix: str, digest: str) -> str:
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", prefix).strip("_-")
    if not safe_prefix:
        raise ValueError("clone_prefix 必须至少包含一个字母或数字")
    return f"{safe_prefix[:32]}_{digest[:24]}"


def load_synthesizer(
    module_name: str = "models.soundai.tts_client",
    *,
    speaker_name: str | None = None,
    clone_prefix: str = "speech_eval",
    emotion: str = "default",
    clone_language: str = "auto",
    enable_asr: bool = True,
    speed: float = 1.0,
    clone_timeout: float = 300,
    synthesis_timeout: float = 120,
) -> Synthesizer:
    """Adapt SoundAI's clone/query/synthesis functions to ``TTSSample``.

    Supplying ``speaker_name`` uses an existing voice for every sample.
    Otherwise a deterministic name derived from the reference WAV is queried,
    cloned when absent, cached, and then used for synthesis.
    """
    if speaker_name is not None:
        speaker_name = speaker_name.strip()
        if not speaker_name:
            raise ValueError("speaker_name 不能为空")
    try:
        module = importlib.import_module(module_name)
        synthesize_speech = getattr(module, "synthesize_speech")
        clone_speaker = getattr(module, "clone_speaker", None)
        speaker_exists = getattr(module, "speaker_exists", None)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"无法加载 TTS 客户端: {module_name}") from exc
    if not callable(synthesize_speech):
        raise ValueError(f"TTS 合成函数不可调用: {module_name}:synthesize_speech")
    if speaker_name is None and (
        not callable(clone_speaker) or not callable(speaker_exists)
    ):
        raise ValueError(
            "未指定已有 speaker_name 时，客户端必须提供 clone_speaker 和 speaker_exists"
        )

    try:
        inspect.signature(synthesize_speech).bind(
            "speaker", "text", Path("output.wav"), speed=speed, timeout=synthesis_timeout
        )
        if speaker_name is None:
            inspect.signature(clone_speaker).bind(
                Path("prompt.wav"),
                "speaker",
                emotion=emotion,
                enable_asr=enable_asr,
                language=clone_language,
                timeout=clone_timeout,
            )
            inspect.signature(speaker_exists).bind("speaker", timeout=clone_timeout)
    except TypeError as exc:
        raise ValueError(f"TTS 客户端参数不兼容: {module_name}: {exc}") from exc

    voice_cache: dict[str, str] = {}

    def synthesize(sample: TTSSample) -> dict[str, Any]:
        output = sample.generated_audio_path
        if output is None:
            raise ValueError("TTSSample.generated_audio_path 不能为空")

        mode = "existing"
        voice = speaker_name.strip() if speaker_name is not None else ""
        if not voice:
            digest = _audio_digest(sample.prompt_audio_path)
            voice = voice_cache.get(digest, "")
            if not voice:
                voice = _speaker_name(clone_prefix, digest)
                if not speaker_exists(voice, timeout=clone_timeout):
                    cloned = clone_speaker(
                        sample.prompt_audio_path,
                        voice,
                        emotion=emotion,
                        enable_asr=enable_asr,
                        language=clone_language,
                        timeout=clone_timeout,
                    )
                    if not isinstance(cloned, str) or not cloned.strip():
                        raise RuntimeError("音色克隆接口未返回 speaker_name")
                    voice = cloned.strip()
                    mode = "cloned"
                else:
                    mode = "reused"
                voice_cache[digest] = voice
            else:
                mode = "cached"

        generated = synthesize_speech(
            voice,
            sample.target_text,
            output,
            speed=speed,
            timeout=synthesis_timeout,
        )
        return {
            "speaker_name": voice,
            "speaker_mode": mode,
            "audio_path": str(generated),
        }

    return synthesize


def main() -> int:
    parser = argparse.ArgumentParser(description="批量调用 TTS 接口生成评测音频")
    parser.add_argument(
        "--client",
        default="models.soundai.tts_client",
        help="提供 synthesize_speech/clone_speaker/speaker_exists 的客户端模块",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        help="数据集名称，可重复；默认处理全部数据集",
    )
    parser.add_argument("--seed-tts-root", default=str(DEFAULT_SEED_TTS_ROOT))
    parser.add_argument("--cv3-eval-root", default=str(DEFAULT_CV3_EVAL_ROOT))
    parser.add_argument("--output-root", required=True, help="JSON 结果输出根目录")
    parser.add_argument(
        "--generated-root",
        help="合成音频根目录；默认 <output-root>/generated",
    )
    parser.add_argument(
        "--speaker-name",
        help="使用已有音色并跳过克隆；默认按每条参考音频自动查询/克隆",
    )
    parser.add_argument("--clone-prefix", default="speech_eval")
    parser.add_argument("--emotion", default="default")
    parser.add_argument("--clone-language", default="auto")
    parser.add_argument("--disable-clone-asr", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--clone-timeout", type=float, default=300)
    parser.add_argument("--synthesis-timeout", type=float, default=120)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.5 <= args.speed <= 2.0:
        parser.error("--speed 必须在 0.5 到 2.0 之间")
    if args.clone_timeout <= 0 or args.synthesis_timeout <= 0:
        parser.error("timeout 必须大于 0")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit 不能为负数")
    if args.speaker_name is not None and not args.speaker_name.strip():
        parser.error("--speaker-name 不能为空")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    output_root = Path(args.output_root).expanduser().resolve()
    generated_root = (
        Path(args.generated_root).expanduser().resolve()
        if args.generated_root
        else output_root / "generated"
    )
    provider = TTSDatasetProvider(
        generated_root,
        seed_tts_root=args.seed_tts_root,
        cv3_eval_root=args.cv3_eval_root,
        generated_suffix=".wav",
    )
    synthesizer = load_synthesizer(
        args.client,
        speaker_name=args.speaker_name,
        clone_prefix=args.clone_prefix,
        emotion=args.emotion,
        clone_language=args.clone_language,
        enable_asr=not args.disable_clone_asr,
        speed=args.speed,
        clone_timeout=args.clone_timeout,
        synthesis_timeout=args.synthesis_timeout,
    )
    consumer = TTSResultConsumer(
        synthesizer,
        output_root,
        overwrite=args.overwrite,
    )
    datasets = args.dataset or provider.list_datasets()
    summary = consumer.consume(provider, datasets, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Synthesizer",
    "TTSResultConsumer",
    "load_synthesizer",
]
