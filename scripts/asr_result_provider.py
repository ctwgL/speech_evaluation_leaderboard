"""Consume vendor ASR clients and persist results by dataset name."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.asr_dataset_provider import (  # noqa: E402
    ASRSample, ASRDatasetProvider, DEFAULT_DATASET_ROOT, get_default_provider,
)

Recognizer = Callable[[ASRSample], Any]


class ASRResultConsumer:
    """Run a recognizer and save the complete dataset to one JSON file."""

    def __init__(self, recognizer: Recognizer, output_root: Path | str,
                 *, overwrite: bool = False, logger: logging.Logger | None = None):
        self.recognizer = recognizer
        self.output_root = Path(output_root).expanduser()
        self.overwrite = overwrite
        self.logger = logger or logging.getLogger(__name__)

    def result_path(self, dataset: str | ASRSample) -> Path:
        name = dataset.dataset if isinstance(dataset, ASRSample) else dataset
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError(f"无效数据集名称: {name}")
        return self.output_root / f"{name}.json"

    def checkpoint_path(self, dataset: str | ASRSample) -> Path:
        """Return the append-only checkpoint used while a dataset is running."""
        return self.result_path(dataset).with_suffix(".checkpoint.jsonl")

    def _load_checkpoint(self, dataset: str) -> dict[str, dict[str, Any]]:
        path = self.checkpoint_path(dataset)
        records = {}
        if not path.is_file():
            return records
        lines = path.read_text(encoding="utf-8").splitlines()
        valid_lines = []
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines):
                    self.logger.warning("忽略未完整写入的最后一条 checkpoint: %s", path)
                    with path.open("w", encoding="utf-8") as stream:
                        if valid_lines:
                            stream.write("\n".join(valid_lines) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    break
                raise ValueError(f"checkpoint 第 {index} 行不是有效 JSON: {path}")
            if not isinstance(record, dict) or record.get("dataset") != dataset or not record.get("sample_id"):
                raise ValueError(f"checkpoint 第 {index} 行格式不正确: {path}")
            records[str(record["sample_id"])] = record
            valid_lines.append(line)
        return records

    def _append_checkpoint(self, dataset: str, record: dict[str, Any]) -> None:
        """Durably append one completed request before processing the next one."""
        path = self.checkpoint_path(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _recognize(self, sample: ASRSample) -> dict[str, Any] | None:
        record = {"sample_id": sample.sample_id, "dataset": sample.dataset,
                  "audio_path": str(sample.audio_path), "reference": sample.reference,
                  "duration": sample.duration, "result": None, "status": "failed",
                  "latency_ms": None}
        if not sample.audio_path.is_file():
            record["error"] = f"音频文件不存在: {sample.audio_path}"
            return record
        started_at = time.perf_counter()
        try:
            result = self.recognizer(sample)
        except Exception as exc:
            record["error"] = str(exc)
            self.logger.exception("处理样本失败: %s", sample.sample_id)
            result = None
        finally:
            record["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
        if result is None or not self._result_text(result).strip():
            record.setdefault("error", "接口未返回识别文本")
        else:
            record["status"] = "success"
            record["result"] = result
        return record

    @staticmethod
    def _result_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("text", "result", "transcript", "recognition", "sentence"):
                if key in value:
                    return ASRResultConsumer._result_text(value[key])
        return ""

    def consume_dataset(self, provider: ASRDatasetProvider, dataset: str,
                        *, limit: int | None = None) -> dict[str, int]:
        stats = {"success": 0, "failed": 0, "skipped": 0}
        path = self.result_path(dataset)
        checkpoint = self.checkpoint_path(dataset)
        previous = {}
        if path.exists() and not self.overwrite:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("dataset") != dataset or not isinstance(data.get("results"), list):
                raise ValueError(f"已有结果文件格式不正确: {path}")
            previous = {str(item["sample_id"]): item for item in data["results"]}
        if self.overwrite:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("", encoding="utf-8")
        recovered = self._load_checkpoint(dataset)
        previous.update(recovered)
        self.logger.info("开始处理数据集 %s，已有结果 %d 条（checkpoint 恢复 %d 条），输出文件: %s",
                         dataset, len(previous), len(recovered), path)
        results = []
        processed_ids = set()
        for index, sample in enumerate(provider.iter_samples(dataset, limit=limit), 1):
            processed_ids.add(sample.sample_id)
            old = previous.get(sample.sample_id)
            if old and old.get("status", "success") == "success" and self._result_text(old.get("result")).strip():
                results.append(old)
                stats["skipped"] += 1
                self.logger.info("[%s] 第 %d 条，样本 %s：跳过已有成功结果",
                                 dataset, index, sample.sample_id)
            else:
                self.logger.info("[%s] 第 %d 条，样本 %s：开始请求，音频 %s",
                                 dataset, index, sample.sample_id, sample.audio_path)
                record = self._recognize(sample)
                self._append_checkpoint(dataset, record)
                results.append(record)
                stats["success" if record["status"] == "success" else "failed"] += 1
                self.logger.info("[%s] 第 %d 条，样本 %s：%s，延时 %s ms%s",
                                 dataset, index, sample.sample_id, record["status"],
                                 record["latency_ms"] if record["latency_ms"] is not None else "未计时",
                                 f"，原因: {record['error']}" if record.get("error") else "")
        for sample_id, old in previous.items():
            if sample_id not in processed_ids:
                results.append(old)
                stats["skipped" if old.get("status", "success") == "success" else "failed"] += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        latencies = [item["latency_ms"] for item in results
                     if item.get("status", "success") == "success" and item.get("latency_ms") is not None]
        average_latency = round(
            sum(latencies) / len(latencies), 3
        ) if latencies else None
        # Replace only after the whole JSON has been written successfully.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=f".{dataset}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"dataset": dataset, "results": results, "stats": stats,
                           "latency_ms": average_latency}, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        if checkpoint.exists():
            checkpoint.unlink()
        self.logger.info("数据集 %s 处理完成：成功 %d，失败 %d，跳过 %d，平均延时 %s ms，结果: %s",
                         dataset, stats["success"], stats["failed"], stats["skipped"],
                         average_latency if average_latency is not None else "无", path)
        return stats

    def consume(self, provider: ASRDatasetProvider, datasets: Iterable[str],
                *, limit: int | None = None) -> dict[str, dict[str, int]]:
        return {name: self.consume_dataset(provider, name, limit=limit) for name in datasets}


def load_recognizer(spec: str, *, language: str = "auto") -> Recognizer:
    """Load ``module:function``; pass language when the client accepts it."""
    try:
        module_name, function_name = spec.split(":", 1)
        function = getattr(importlib.import_module(module_name), function_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(f"无效请求接口: {spec}，格式应为 module:function") from exc
    if not callable(function):
        raise ValueError(f"请求接口不可调用: {spec}")
    signature = inspect.signature(function)
    accepts_language = "language" in signature.parameters
    try:
        signature.bind(Path("sample.wav"), **({"language": language} if accepts_language else {}))
    except TypeError as exc:
        raise ValueError(f"请求接口参数不兼容: {spec}: {exc}") from exc
    return lambda sample: function(sample.audio_path, **({"language": language} if accepts_language else {}))


def main() -> int:
    parser = argparse.ArgumentParser(description="使用公司 ASR 接口批量生成识别结果")
    parser.add_argument("--client", required=True, help="请求函数，格式：module:function")
    parser.add_argument("--dataset", action="append", help="数据集名称，可重复；默认全部")
    parser.add_argument("--language", default="auto", help="传给支持 language 参数的接口，默认 auto")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    provider = ASRDatasetProvider(args.dataset_root)
    consumer = ASRResultConsumer(load_recognizer(args.client, language=args.language), args.output_root, overwrite=args.overwrite)
    datasets = args.dataset or provider.list_datasets()
    print(json.dumps(consumer.consume(provider, datasets, limit=args.limit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["ASRResultConsumer", "load_recognizer", "ASRSample", "ASRDatasetProvider", "get_default_provider"]
