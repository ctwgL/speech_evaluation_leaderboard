"""Run Qwen3-ASR on SpeechIO evaluation sets with local batched inference.

The output format is compatible with ``scripts/submitted_result.py``.  A JSONL
checkpoint is flushed after every batch, so an interrupted evaluation can be
continued by running the same command again.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
import wave
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    # Use the repository's qwen_asr package even when it is not pip-installed.
    sys.path.insert(0, str(SCRIPT_DIR))

from models.asr_dataset_provider import (  # noqa: E402
    ASRDatasetProvider,
    ASRSample,
    DEFAULT_DATASET_ROOT,
)


LOCAL_MODELS = {
    "1.7b": SCRIPT_DIR / "checkpoint/qwen3_asr_1_7B",
    "0.6b": SCRIPT_DIR / "checkpoint/qwen3_asr_0_6B",
}
LANGUAGE_ALIASES = {
    "auto": None,
    "": None,
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "chinese": "Chinese",
    "中文": "Chinese",
    "en": "English",
    "en-us": "English",
    "english": "English",
    "英文": "English",
}
LOGGER = logging.getLogger("qwen3_asr_evaluation")


def _language(value: str) -> str | None:
    normalized = value.strip()
    return LANGUAGE_ALIASES.get(normalized.lower(), normalized or None)


def _duration(path: Path, supplied: float | None) -> float | None:
    """Return duration without decoding the complete audio when possible."""
    if supplied is not None:
        return supplied
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception:
        try:
            with wave.open(str(path), "rb") as stream:
                return stream.getnframes() / stream.getframerate()
        except (OSError, wave.Error, ZeroDivisionError):
            return None


def _result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "result", "transcript", "recognition", "sentence"):
            if key in value:
                return _result_text(value[key])
    return ""


def _chunks(values: Sequence[ASRSample], size: int) -> Iterable[list[ASRSample]]:
    for offset in range(0, len(values), size):
        yield list(values[offset : offset + size])


def _read_records(path: Path, dataset: str, *, jsonl: bool) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    if jsonl:
        records: list[Any] = []
        lines = path.read_text(encoding="utf-8").splitlines()
        valid_lines: list[str] = []
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if line_number == len(lines):
                    LOGGER.warning("忽略 checkpoint 中未完整写入的最后一行: %s", path)
                    path.write_text(
                        ("\n".join(valid_lines) + "\n") if valid_lines else "",
                        encoding="utf-8",
                    )
                    break
                raise ValueError(f"checkpoint 第 {line_number} 行不是有效 JSON: {path}")
            valid_lines.append(line)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("dataset") != dataset:
            raise ValueError(f"已有结果文件的数据集名称不匹配: {path}")
        records = payload.get("results")
        if not isinstance(records, list):
            raise ValueError(f"已有结果文件缺少 results 数组: {path}")

    output: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("dataset") != dataset or not record.get("sample_id"):
            raise ValueError(f"结果记录格式无效: {path}")
        output[str(record["sample_id"])] = record
    return output


def _append_checkpoint(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
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


def load_model(args: argparse.Namespace) -> Any:
    """Load one of the local checkpoints using the requested offline backend."""
    import torch
    from qwen_asr import Qwen3ASRModel

    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else LOCAL_MODELS[args.model]
    if not model_path.is_dir():
        raise FileNotFoundError(f"本地模型目录不存在: {model_path}")

    LOGGER.info("加载 Qwen3-ASR %s，后端=%s，路径=%s", args.model, args.backend, model_path)
    if args.backend == "vllm":
        return Qwen3ASRModel.LLM(
            model=str(model_path),
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_inference_batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

    dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    return Qwen3ASRModel.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_implementation,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


class BatchedEvaluator:
    def __init__(self, model: Any, output_root: Path | str, *, batch_size: int,
                 language: str = "auto", overwrite: bool = False,
                 model_id: str = "qwen3-asr", backend: str = "transformers") -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.model = model
        self.output_root = Path(output_root).expanduser().resolve()
        self.batch_size = batch_size
        self.language = _language(language)
        self.overwrite = overwrite
        self.model_id = model_id
        self.backend = backend

    def _paths(self, dataset: str) -> tuple[Path, Path]:
        if Path(dataset).name != dataset or dataset in ("", ".", ".."):
            raise ValueError(f"无效数据集名称: {dataset}")
        result = self.output_root / f"{dataset}.json"
        return result, result.with_suffix(".checkpoint.jsonl")

    def _base_record(self, sample: ASRSample) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "audio_path": str(sample.audio_path),
            "reference": sample.reference,
            "duration": _duration(sample.audio_path, sample.duration),
            "result": None,
            "status": "failed",
            "latency_ms": None,
            "model": self.model_id,
            "backend": self.backend,
        }

    def _infer_batch(self, samples: Sequence[ASRSample]) -> list[dict[str, Any]]:
        records = [self._base_record(sample) for sample in samples]
        missing = [sample.audio_path for sample in samples if not sample.audio_path.is_file()]
        if missing:
            # Keep valid samples useful even if one metadata row has a bad path.
            if len(samples) > 1:
                output: list[dict[str, Any]] = []
                for sample in samples:
                    output.extend(self._infer_batch([sample]))
                return output
            records[0]["error"] = f"音频文件不存在: {missing[0]}"
            return records

        started = time.perf_counter()
        try:
            transcriptions = self.model.transcribe(
                audio=[str(sample.audio_path) for sample in samples],
                language=self.language,
                return_time_stamps=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            if len(transcriptions) != len(samples):
                raise RuntimeError(
                    f"模型返回数量不匹配: 输入 {len(samples)}，输出 {len(transcriptions)}"
                )

            # Allocate batch wall time across records.  Its sum remains the real
            # elapsed time, making submitted_result.py's aggregate RTF correct.
            durations = [record["duration"] for record in records]
            duration_sum = sum(value for value in durations if value is not None and value > 0)
            weights = (
                [(value / duration_sum if value is not None and value > 0 else 0.0) for value in durations]
                if duration_sum > 0 and all(value is not None and value > 0 for value in durations)
                else [1.0 / len(records)] * len(records)
            )
            for record, transcription, weight in zip(records, transcriptions, weights):
                text = str(getattr(transcription, "text", "")).strip()
                record["latency_ms"] = round(elapsed_ms * weight, 3)
                record["batch_latency_ms"] = round(elapsed_ms, 3)
                record["result"] = {
                    "text": text,
                    "language": str(getattr(transcription, "language", "")),
                }
                if text:
                    record["status"] = "success"
                else:
                    record["error"] = "模型未返回识别文本"
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            LOGGER.exception("批推理失败，样本=%s", ",".join(sample.sample_id for sample in samples))
            share = elapsed_ms / len(records)
            for record in records:
                record["latency_ms"] = round(share, 3)
                record["batch_latency_ms"] = round(elapsed_ms, 3)
                record["error"] = f"{type(exc).__name__}: {exc}"
        return records

    def evaluate_dataset(self, provider: ASRDatasetProvider, dataset: str,
                         *, limit: int | None = None) -> dict[str, int]:
        result_path, checkpoint_path = self._paths(dataset)
        if self.overwrite:
            result_path.unlink(missing_ok=True)
            checkpoint_path.unlink(missing_ok=True)

        previous = _read_records(result_path, dataset, jsonl=False)
        previous.update(_read_records(checkpoint_path, dataset, jsonl=True))
        samples = list(provider.iter_samples(dataset, limit=limit))
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError(f"数据集存在重复 sample_id: {dataset}")

        completed = {
            sample_id for sample_id, record in previous.items()
            if record.get("status", "success") == "success"
            and _result_text(record.get("result")).strip()
            and record.get("model") == self.model_id
            and record.get("backend") == self.backend
        }
        pending = [sample for sample in samples if sample.sample_id not in completed]
        LOGGER.info(
            "数据集 %s：总计 %d，已有成功结果 %d，待推理 %d，batch_size=%d",
            dataset, len(samples), len(samples) - len(pending), len(pending), self.batch_size,
        )

        current = dict(previous)
        for batch_index, batch in enumerate(_chunks(pending, self.batch_size), 1):
            LOGGER.info(
                "[%s] batch %d：推理 %d 条（%s ... %s）",
                dataset, batch_index, len(batch), batch[0].sample_id, batch[-1].sample_id,
            )
            records = self._infer_batch(batch)
            _append_checkpoint(checkpoint_path, records)
            current.update({str(record["sample_id"]): record for record in records})

        ordered = [current[sample.sample_id] for sample in samples]
        skipped = len(samples) - len(pending)
        pending_ids = {sample.sample_id for sample in pending}
        stats = {
            "success": sum(
                record.get("status") == "success" and record["sample_id"] in pending_ids
                for record in ordered
            ),
            "failed": sum(record.get("status") != "success" for record in ordered),
            "skipped": skipped,
        }
        latencies = [
            float(record["latency_ms"]) for record in ordered
            if record.get("status") == "success" and record.get("latency_ms") is not None
        ]
        average_latency = round(sum(latencies) / len(latencies), 3) if latencies else None
        _atomic_json(result_path, {
            "dataset": dataset,
            "results": ordered,
            "stats": stats,
            "latency_ms": average_latency,
            "inference": {
                "mode": "local_offline_batch",
                "model": self.model_id,
                "backend": self.backend,
                "batch_size": self.batch_size,
                "language": self.language or "auto",
            },
        })
        checkpoint_path.unlink(missing_ok=True)
        LOGGER.info("数据集 %s 完成：成功 %d，失败 %d，结果=%s",
                    dataset, stats["success"], stats["failed"], result_path)
        return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-ASR 本地离线批量评测")
    parser.add_argument("--model", choices=sorted(LOCAL_MODELS), default="1.7b",
                        help="使用仓库内的 1.7B 或 0.6B checkpoint（默认 1.7b）")
    parser.add_argument("--model-path", help="自定义本地 checkpoint；设置后覆盖 --model 对应路径")
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--dataset", action="append", help="评测集名称，可重复；默认评测全部数据集")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--language", default="auto",
                        help="auto/zh/en 或 Qwen3-ASR 支持的完整语言名")
    parser.add_argument("--limit", type=int, help="每个数据集最多推理的样本数，用于冒烟测试")
    parser.add_argument("--overwrite", action="store_true", help="清空已有结果并重新推理")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"),
                        default="bfloat16", help="Transformers 后端模型精度")
    parser.add_argument("--device", default="cuda:0", help="Transformers device_map（默认 cuda:0）")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"),
                        default="sdpa", help="Transformers attention 实现")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="vLLM 可使用的显存比例")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor parallel GPU 数量")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit 不能为负数")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens 必须大于 0")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization 必须在 (0, 1] 范围内")
    if args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size 必须大于 0")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    provider = ASRDatasetProvider(args.dataset_root)
    datasets = args.dataset or provider.list_datasets()
    if not datasets:
        parser.error("没有找到可评测的数据集")
    model = load_model(args)
    evaluator = BatchedEvaluator(
        model, args.output_root, batch_size=args.batch_size,
        language=args.language, overwrite=args.overwrite,
        model_id=(str(Path(args.model_path).expanduser().resolve())
                  if args.model_path else f"qwen3-asr-{args.model}"),
        backend=args.backend,
    )
    summary = {
        dataset: evaluator.evaluate_dataset(provider, dataset, limit=args.limit)
        for dataset in datasets
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
