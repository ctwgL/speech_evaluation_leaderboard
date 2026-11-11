"""Consume speech-enhancement clients and persist per-dataset JSON results."""

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

from models.se_dataset_provider import (  # noqa: E402
    DATASETS,
    DEFAULT_BLIND_ROOT,
    DEFAULT_TEST_ROOT,
    SEDatasetProvider,
    SESample,
)


Enhancer = Callable[[SESample], Any]


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


class SEResultConsumer:
    """Run an enhancer and save one JSON result file per partition/track."""

    def __init__(
        self,
        enhancer: Enhancer,
        output_root: Path | str,
        *,
        overwrite: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.enhancer = enhancer
        self.output_root = Path(output_root).expanduser().resolve()
        self.overwrite = overwrite
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _validate_dataset(dataset: str) -> tuple[str, str]:
        if dataset not in DATASETS:
            raise ValueError(
                f"无效语音增强数据集: {dataset!r}；可选值: {', '.join(DATASETS)}"
            )
        partition, track = dataset.split("/", 1)
        return partition, track

    def result_path(self, dataset: str | SESample) -> Path:
        name = dataset.dataset if isinstance(dataset, SESample) else dataset
        partition, track = self._validate_dataset(name)
        return self.output_root / partition / f"{track}.json"

    def checkpoint_path(self, dataset: str | SESample) -> Path:
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
    def _base_record(sample: SESample) -> dict[str, Any]:
        return {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "partition": sample.partition,
            "track": sample.track,
            "enrol_audio_path": str(sample.enrol_audio_path),
            "noisy_audio_path": str(sample.noisy_audio_path),
            "enhanced_audio_path": str(sample.enhanced_audio_path),
            "result": None,
            "status": "failed",
            "latency_ms": None,
        }

    @staticmethod
    def _completed(record: dict[str, Any] | None, sample: SESample) -> bool:
        if not record or record.get("status") != "success":
            return False
        if record.get("enhanced_audio_path") != str(sample.enhanced_audio_path):
            return False
        output = sample.enhanced_audio_path
        return output.is_file() and output.stat().st_size > 0

    def _enhance(self, sample: SESample) -> dict[str, Any]:
        record = self._base_record(sample)
        missing = [
            path
            for path in (sample.enrol_audio_path, sample.noisy_audio_path)
            if not path.is_file()
        ]
        if missing:
            record["error"] = f"输入音频不存在: {missing[0]}"
            return record

        started_at = time.perf_counter()
        try:
            result = self.enhancer(sample)
            output = sample.enhanced_audio_path
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"客户端未生成有效的增强音频: {output}")
            record["result"] = result
            record["status"] = "success"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.logger.exception("处理样本失败: %s", sample.sample_id)
        finally:
            record["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
        return record

    def consume_dataset(
        self,
        provider: SEDatasetProvider,
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
            "开始处理数据集 %s，已有结果 %d 条（checkpoint 恢复 %d 条），输出文件: %s",
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
                "[%s] 第 %d 条，样本 %s：开始增强，noisy=%s，enrol=%s",
                dataset,
                index,
                sample.sample_id,
                sample.noisy_audio_path,
                sample.enrol_audio_path,
            )
            record = self._enhance(sample)
            self._append_checkpoint(dataset, record)
            results.append(record)
            stats["success" if record["status"] == "success" else "failed"] += 1
            self.logger.info(
                "[%s] 第 %d 条，样本 %s：%s，延时 %s ms%s",
                dataset,
                index,
                sample.sample_id,
                record["status"],
                record["latency_ms"],
                f"，原因: {record['error']}" if record.get("error") else "",
            )

        for sample_id, old in previous.items():
            if sample_id not in processed_ids:
                results.append(old)
                stats["skipped" if old.get("status") == "success" else "failed"] += 1

        latencies = [
            float(record["latency_ms"])
            for record in results
            if record.get("status") == "success"
            and record.get("latency_ms") is not None
        ]
        average_latency = round(sum(latencies) / len(latencies), 3) if latencies else None
        _atomic_json(
            path,
            {
                "dataset": dataset,
                "partition": spec.partition,
                "track": spec.track,
                "enhanced_root": str(provider.enhanced_root),
                "results": results,
                "stats": stats,
                "latency_ms": average_latency,
            },
        )
        checkpoint.unlink(missing_ok=True)
        self.logger.info(
            "数据集 %s 处理完成：成功 %d，失败 %d，跳过 %d，平均延时 %s ms，结果: %s",
            dataset,
            stats["success"],
            stats["failed"],
            stats["skipped"],
            average_latency if average_latency is not None else "无",
            path,
        )
        return stats

    def consume(
        self,
        provider: SEDatasetProvider,
        datasets: Iterable[str],
        *,
        limit: int | None = None,
    ) -> dict[str, dict[str, int]]:
        return {
            dataset: self.consume_dataset(provider, dataset, limit=limit)
            for dataset in datasets
        }


def load_enhancer(spec: str, *, trim_duration: int = 600) -> Enhancer:
    """Adapt a two-stage ``submit_task``/``get_result`` client module.

    ``spec`` accepts ``module`` or ``module:submit_function,get_function``.
    The adapter downloads to a temporary file and atomically installs the
    completed enhanced audio at ``sample.enhanced_audio_path``.
    """
    module_name, separator, function_names = spec.partition(":")
    submit_name, get_name = "submit_task", "get_result"
    if separator:
        names = [name.strip() for name in function_names.split(",")]
        if len(names) != 2 or not all(names):
            raise ValueError(
                "无效 SE 客户端格式，应为 module 或 module:submit_function,get_function"
            )
        submit_name, get_name = names
    try:
        module = importlib.import_module(module_name)
        submit_task = getattr(module, submit_name)
        get_result = getattr(module, get_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"无法加载 SE 客户端: {spec}") from exc
    if not callable(submit_task) or not callable(get_result):
        raise ValueError(f"SE 客户端函数不可调用: {spec}")

    submit_signature = inspect.signature(submit_task)
    accepts_trim = "trim_duration" in submit_signature.parameters
    try:
        submit_signature.bind(
            "sample.wav", **({"trim_duration": trim_duration} if accepts_trim else {})
        )
        inspect.signature(get_result).bind("task-id", "enhanced.mp3")
    except TypeError as exc:
        raise ValueError(f"SE 客户端参数不兼容: {spec}: {exc}") from exc

    def enhance(sample: SESample) -> dict[str, Any]:
        output = sample.enhanced_audio_path
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent,
                prefix=f".{output.stem}.",
                suffix=output.suffix or ".audio",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)

            submit_kwargs = {"trim_duration": trim_duration} if accepts_trim else {}
            task_id = submit_task(str(sample.noisy_audio_path), **submit_kwargs)
            if task_id is None:
                raise RuntimeError("提交增强任务失败，客户端未返回 taskid")
            if not get_result(task_id, str(temporary)):
                raise RuntimeError(f"下载增强结果失败，taskid={task_id}")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"下载的增强结果为空，taskid={task_id}")
            os.replace(temporary, output)
            return {"task_id": task_id}
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    return enhance


def main() -> int:
    parser = argparse.ArgumentParser(description="批量调用语音增强接口并保存结果")
    parser.add_argument(
        "--client",
        default="models.soundai.se_client",
        help="客户端模块，或 module:submit_function,get_function",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        help="数据集名称，可重复；默认处理全部四个分组",
    )
    parser.add_argument("--blind-root", default=str(DEFAULT_BLIND_ROOT))
    parser.add_argument("--test-root", default=str(DEFAULT_TEST_ROOT))
    parser.add_argument("--output-root", required=True, help="JSON 结果输出根目录")
    parser.add_argument(
        "--enhanced-root",
        help="增强音频输出根目录；默认 <output-root>/enhanced",
    )
    parser.add_argument(
        "--enhanced-suffix",
        default=".mp3",
        help="增强音频后缀；SoundAI 默认 .mp3",
    )
    parser.add_argument("--trim-duration", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.trim_duration <= 0:
        parser.error("--trim-duration 必须大于 0")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit 不能为负数")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    output_root = Path(args.output_root).expanduser().resolve()
    enhanced_root = (
        Path(args.enhanced_root).expanduser().resolve()
        if args.enhanced_root
        else output_root / "enhanced"
    )
    provider = SEDatasetProvider(
        enhanced_root,
        blind_root=args.blind_root,
        test_root=args.test_root,
        enhanced_suffix=args.enhanced_suffix,
    )
    consumer = SEResultConsumer(
        load_enhancer(args.client, trim_duration=args.trim_duration),
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
    "Enhancer",
    "SEResultConsumer",
    "load_enhancer",
]
