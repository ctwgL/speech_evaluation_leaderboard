"""Run FireRedASR/FireRedASR2 models on the shared ASR datasets."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import sys
import tempfile
import time
import wave
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.asr_dataset_provider import (  # noqa: E402
    ASRDatasetProvider,
    ASRSample,
    DEFAULT_DATASET_ROOT,
)


LEGACY_DEFAULT_REPOSITORY = Path(
    "/home/chentingwei/cosyVoice/speech_evaluation_dataset/model/"
    "fireRedasr/FireRedASR"
)
DEFAULT_REPOSITORY = Path(
    "/mnt/lustre/chentingwei/cosyVoice/speech_evaluation_dataset/model/"
    "fireRedasr/FireRedASR2S"
)
DEFAULT_MODEL_DIR = DEFAULT_REPOSITORY / "pretrained_models/FireRedASR2-LLM"
DEFAULT_AED_MODEL_DIR = DEFAULT_REPOSITORY / "pretrained_models/FireRedASR2-AED"
LEGACY_DEFAULT_MODEL_DIR = (
    LEGACY_DEFAULT_REPOSITORY / "pretrained_models/FireRedASR-LLM-L"
)
LEGACY_DEFAULT_AED_MODEL_DIR = (
    LEGACY_DEFAULT_REPOSITORY / "pretrained_models/FireRedASR-AED-L"
)
LOGGER = logging.getLogger("fireredasr_evaluation")


@dataclass(frozen=True)
class ModelSpec:
    family: str
    asr_type: str
    model_id: str
    required_files: tuple[str, ...]
    default_repository: Path
    default_model_dir: Path
    repository_marker: str
    module_name: str
    class_name: str


# Adding another backend only requires a ModelSpec and a loader branch in
# load_model; DatasetEvaluator intentionally has no model-family conditionals.
SUPPORTED_MODELS = {
    ("fireredasr2", "llm"): ModelSpec(
        family="fireredasr2",
        asr_type="llm",
        model_id="fireredasr2-llm",
        required_files=(
            "model.pth.tar",
            "asr_encoder.pth.tar",
            "cmvn.ark",
            "Qwen2-7B-Instruct",
        ),
        default_repository=DEFAULT_REPOSITORY,
        default_model_dir=DEFAULT_MODEL_DIR,
        repository_marker="fireredasr2s/fireredasr2/asr.py",
        module_name="fireredasr2s.fireredasr2.asr",
        class_name="FireRedAsr2",
    ),
    ("fireredasr2", "aed"): ModelSpec(
        family="fireredasr2",
        asr_type="aed",
        model_id="fireredasr2-aed",
        required_files=(
            "model.pth.tar",
            "cmvn.ark",
            "dict.txt",
            "train_bpe1000.model",
        ),
        default_repository=DEFAULT_REPOSITORY,
        default_model_dir=DEFAULT_AED_MODEL_DIR,
        repository_marker="fireredasr2s/fireredasr2/asr.py",
        module_name="fireredasr2s.fireredasr2.asr",
        class_name="FireRedAsr2",
    ),
    ("fireredasr", "llm"): ModelSpec(
        family="fireredasr",
        asr_type="llm",
        model_id="fireredasr-llm-l",
        required_files=(
            "model.pth.tar",
            "asr_encoder.pth.tar",
            "cmvn.ark",
            "Qwen2-7B-Instruct",
        ),
        default_repository=LEGACY_DEFAULT_REPOSITORY,
        default_model_dir=LEGACY_DEFAULT_MODEL_DIR,
        repository_marker="fireredasr/models/fireredasr.py",
        module_name="fireredasr.models.fireredasr",
        class_name="FireRedAsr",
    ),
    ("fireredasr", "aed"): ModelSpec(
        family="fireredasr",
        asr_type="aed",
        model_id="fireredasr-aed-l",
        required_files=(
            "model.pth.tar",
            "cmvn.ark",
            "dict.txt",
            "train_bpe1000.model",
        ),
        default_repository=LEGACY_DEFAULT_REPOSITORY,
        default_model_dir=LEGACY_DEFAULT_AED_MODEL_DIR,
        repository_marker="fireredasr/models/fireredasr.py",
        module_name="fireredasr.models.fireredasr",
        class_name="FireRedAsr",
    ),
}


DEFAULT_MODEL_DIRS = {
    "llm": DEFAULT_MODEL_DIR,
    "aed": DEFAULT_AED_MODEL_DIR,
}
DEFAULT_DECODE_OPTIONS_BY_TYPE: dict[str, dict[str, Any]] = {
    "llm": {
        "use_gpu": True,
        "use_half": False,
        "beam_size": 3,
        "decode_max_len": 0,
        "decode_min_len": 0,
        "repetition_penalty": 3.0,
        "llm_length_penalty": 1.0,
        "temperature": 1.0,
    },
    "aed": {
        "use_gpu": True,
        "use_half": False,
        "beam_size": 3,
        "nbest": 1,
        "decode_max_len": 0,
        "softmax_smoothing": 1.25,
        "aed_length_penalty": 0.6,
        "eos_penalty": 1.0,
    },
}
# Backward-compatible name for callers that imported the former LLM defaults.
DEFAULT_DECODE_OPTIONS = DEFAULT_DECODE_OPTIONS_BY_TYPE["llm"]


def _model_spec(family: str, asr_type: str) -> ModelSpec:
    try:
        return SUPPORTED_MODELS[(family.lower(), asr_type.lower())]
    except KeyError as exc:
        supported = ", ".join(f"{key[0]}/{key[1]}" for key in SUPPORTED_MODELS)
        raise ValueError(
            f"尚未支持模型组合 {family}/{asr_type}；当前支持: {supported}"
        ) from exc


def _validate_model_dir(model_dir: Path, spec: ModelSpec) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"FireRedASR 模型目录不存在: {model_dir}")
    missing = [name for name in spec.required_files if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"FireRedASR 模型目录缺少文件: {', '.join(missing)} ({model_dir})"
        )


class _FireRedAsr2Adapter:
    """Expose the legacy three-argument transcribe interface for FireRedASR2."""

    def __init__(self, model: Any, config: Any) -> None:
        self.model = model
        self.config = config

    def transcribe(
        self,
        batch_uttid: Sequence[str],
        batch_wav_path: Sequence[str],
        decode_options: dict[str, Any],
    ) -> Any:
        for name, value in decode_options.items():
            if not hasattr(self.config, name):
                raise ValueError(f"FireRedASR2 不支持解码参数: {name}")
            if name == "use_gpu" and value != self.config.use_gpu:
                raise ValueError("FireRedASR2 的 device 必须在加载模型时确定")
            setattr(self.config, name, value)
        return self.model.transcribe(list(batch_uttid), list(batch_wav_path))


def load_model(
    repository: Path | str | None = None,
    model_dir: Path | str | None = None,
    *,
    family: str = "fireredasr2",
    asr_type: str = "llm",
    decode_options: dict[str, Any] | None = None,
) -> Any:
    """Load a local FireRedASR model through the matching upstream API."""
    spec = _model_spec(family, asr_type)
    repository_path = Path(repository or spec.default_repository).expanduser().resolve()
    selected_model_dir = model_dir or spec.default_model_dir
    model_path = Path(selected_model_dir).expanduser().resolve()
    if not repository_path.is_dir():
        raise FileNotFoundError(f"FireRedASR 仓库不存在: {repository_path}")
    if not (repository_path / spec.repository_marker).is_file():
        raise FileNotFoundError(
            f"不是有效的 {spec.family} 仓库（缺少 {spec.repository_marker}）: "
            f"{repository_path}"
        )
    _validate_model_dir(model_path, spec)

    # The upstream project is not installed as a package in the evaluation
    # environment.  Import it lazily so --help and unit tests need no torch.
    repository_string = str(repository_path)
    if repository_string not in sys.path:
        sys.path.insert(0, repository_string)
    try:
        module = importlib.import_module(spec.module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"无法导入 {spec.family}，请检查仓库依赖是否已安装: {exc}"
        ) from exc

    LOGGER.info("加载 %s，模型目录=%s", spec.model_id, model_path)
    model_class = getattr(module, spec.class_name)
    if spec.family == "fireredasr2":
        options = dict(DEFAULT_DECODE_OPTIONS_BY_TYPE[spec.asr_type])
        if decode_options:
            options.update(decode_options)
        config = module.FireRedAsr2Config(**options)
        model = model_class.from_pretrained(spec.asr_type, str(model_path), config)
        return _FireRedAsr2Adapter(model, config)
    return model_class.from_pretrained(spec.asr_type, str(model_path))


def _duration(path: Path, supplied: float | None) -> float | None:
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
        if (
            not isinstance(record, dict)
            or record.get("dataset") != dataset
            or not record.get("sample_id")
        ):
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
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class DatasetEvaluator:
    """Evaluate datasets with any model implementing FireRedASR transcribe()."""

    def __init__(
        self,
        model: Any,
        output_root: Path | str,
        *,
        batch_size: int = 1,
        decode_options: dict[str, Any] | None = None,
        overwrite: bool = False,
        model_id: str = "fireredasr2-llm",
        backend: str = "fireredasr2",
        asr_type: str = "llm",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.model = model
        self.output_root = Path(output_root).expanduser().resolve()
        self.batch_size = batch_size
        normalized_asr_type = asr_type.lower()
        if normalized_asr_type not in DEFAULT_DECODE_OPTIONS_BY_TYPE:
            raise ValueError(f"不支持的 asr_type: {asr_type}")
        self.asr_type = normalized_asr_type
        self.decode_options = dict(DEFAULT_DECODE_OPTIONS_BY_TYPE[self.asr_type])
        if decode_options:
            self.decode_options.update(decode_options)
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
            if len(samples) > 1:
                return [record for sample in samples for record in self._infer_batch([sample])]
            records[0]["error"] = f"音频文件不存在: {missing[0]}"
            return records

        started = time.perf_counter()
        try:
            results = self.model.transcribe(
                [sample.sample_id for sample in samples],
                [str(sample.audio_path) for sample in samples],
                dict(self.decode_options),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
                raise RuntimeError("模型返回值不是结果列表")
            if len(results) != len(samples):
                raise RuntimeError(
                    f"模型返回数量不匹配: 输入 {len(samples)}，输出 {len(results)}"
                )
            returned = {
                str(result.get("uttid")): result
                for result in results
                if isinstance(result, dict) and result.get("uttid") is not None
            }
            if len(returned) != len(samples):
                raise RuntimeError("模型结果缺少或包含重复 uttid")

            durations = [record["duration"] for record in records]
            duration_sum = sum(
                value for value in durations if value is not None and value > 0
            )
            use_duration_weights = duration_sum > 0 and all(
                value is not None and value > 0 for value in durations
            )
            for record, sample in zip(records, samples):
                result = returned.get(sample.sample_id)
                if result is None:
                    raise RuntimeError(f"模型结果缺少 uttid: {sample.sample_id}")
                text = _result_text(result).strip()
                weight = (
                    float(record["duration"]) / duration_sum
                    if use_duration_weights
                    else 1.0 / len(records)
                )
                record["latency_ms"] = round(elapsed_ms * weight, 3)
                record["batch_latency_ms"] = round(elapsed_ms, 3)
                record["result"] = {
                    "text": text,
                    "rtf": result.get("rtf"),
                }
                if text:
                    record["status"] = "success"
                else:
                    record["error"] = "模型未返回识别文本"
        except Exception as exc:
            # A batch-level failure is retried sample-by-sample so one corrupt
            # audio file does not discard all otherwise valid results.
            if len(samples) > 1:
                LOGGER.warning(
                    "批推理失败，将逐条重试（%s）: %s", type(exc).__name__, exc
                )
                return [record for sample in samples for record in self._infer_batch([sample])]
            elapsed_ms = (time.perf_counter() - started) * 1000
            LOGGER.exception("推理失败，样本=%s", samples[0].sample_id)
            records[0]["latency_ms"] = round(elapsed_ms, 3)
            records[0]["batch_latency_ms"] = round(elapsed_ms, 3)
            records[0]["error"] = f"{type(exc).__name__}: {exc}"
        return records

    def evaluate_dataset(
        self,
        provider: ASRDatasetProvider,
        dataset: str,
        *,
        limit: int | None = None,
    ) -> dict[str, int]:
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
            sample_id
            for sample_id, record in previous.items()
            if record.get("status", "success") == "success"
            and _result_text(record.get("result")).strip()
            and record.get("model") == self.model_id
            and record.get("backend") == self.backend
        }
        pending = [sample for sample in samples if sample.sample_id not in completed]
        LOGGER.info(
            "数据集 %s：总计 %d，已有成功结果 %d，待推理 %d，batch_size=%d",
            dataset,
            len(samples),
            len(samples) - len(pending),
            len(pending),
            self.batch_size,
        )

        current = dict(previous)
        for batch_index, batch in enumerate(_chunks(pending, self.batch_size), 1):
            LOGGER.info(
                "[%s] batch %d：推理 %d 条（%s ... %s）",
                dataset,
                batch_index,
                len(batch),
                batch[0].sample_id,
                batch[-1].sample_id,
            )
            records = self._infer_batch(batch)
            _append_checkpoint(checkpoint_path, records)
            current.update({str(record["sample_id"]): record for record in records})

        ordered = [current[sample.sample_id] for sample in samples]
        pending_ids = {sample.sample_id for sample in pending}
        stats = {
            "success": sum(
                record.get("status") == "success"
                and record["sample_id"] in pending_ids
                for record in ordered
            ),
            "failed": sum(record.get("status") != "success" for record in ordered),
            "skipped": len(samples) - len(pending),
        }
        latencies = [
            float(record["latency_ms"])
            for record in ordered
            if record.get("status") == "success"
            and record.get("latency_ms") is not None
        ]
        average_latency = round(sum(latencies) / len(latencies), 3) if latencies else None
        _atomic_json(
            result_path,
            {
                "dataset": dataset,
                "results": ordered,
                "stats": stats,
                "latency_ms": average_latency,
                "inference": {
                    "mode": "local_offline_batch",
                    "model": self.model_id,
                    "backend": self.backend,
                    "asr_type": self.asr_type,
                    "batch_size": self.batch_size,
                    "decode_options": self.decode_options,
                },
            },
        )
        checkpoint_path.unlink(missing_ok=True)
        LOGGER.info(
            "数据集 %s 完成：成功 %d，失败 %d，跳过 %d，结果=%s",
            dataset,
            stats["success"],
            stats["failed"],
            stats["skipped"],
            result_path,
        )
        return stats


_RECOGNIZER_CACHE: dict[tuple[str, ...], Any] = {}


def recognize(
    file_path: str | Path,
    language: str = "auto",
    *,
    repository: Path | str | None = None,
    model_dir: Path | str | None = None,
    family: str = "fireredasr2",
    asr_type: str = "llm",
    decode_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single-file adapter compatible with ``scripts/asr_result_provider.py``.

    FireRedASR performs language detection internally, so ``language`` is
    accepted for the shared consumer interface but is intentionally unused.
    For a full dataset, the CLI is faster because it supports batching.
    """
    del language
    audio_path = Path(file_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    spec = _model_spec(family, asr_type)
    repository_path = str(Path(repository or spec.default_repository).expanduser().resolve())
    selected_model_dir = model_dir or spec.default_model_dir
    model_path = str(Path(selected_model_dir).expanduser().resolve())
    options = dict(DEFAULT_DECODE_OPTIONS_BY_TYPE[spec.asr_type])
    if decode_options:
        options.update(decode_options)
    options_key = json.dumps(options, sort_keys=True, separators=(",", ":"))
    key = (family.lower(), spec.asr_type, repository_path, model_path, options_key)
    model = _RECOGNIZER_CACHE.get(key)
    if model is None:
        model = load_model(
            repository_path,
            model_path,
            family=family,
            asr_type=spec.asr_type,
            decode_options=options,
        )
        _RECOGNIZER_CACHE[key] = model
    results = model.transcribe([audio_path.stem], [str(audio_path)], options)
    if not results or not isinstance(results[0], dict):
        raise RuntimeError("FireRedASR 未返回有效识别结果")
    text = _result_text(results[0]).strip()
    if not text:
        raise RuntimeError("FireRedASR 未返回识别文本")
    return {"text": text, "rtf": results[0].get("rtf")}


def recognize_aed(
    file_path: str | Path,
    language: str = "auto",
    *,
    repository: Path | str | None = None,
    model_dir: Path | str | None = None,
    decode_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recognize one file with FireRedASR2-AED for the shared provider."""
    return recognize(
        file_path,
        language,
        repository=repository,
        model_dir=model_dir,
        family="fireredasr2",
        asr_type="aed",
        decode_options=decode_options,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FireRedASR2/FireRedASR 本地离线批量评测")
    parser.add_argument(
        "--model-family",
        choices=("fireredasr2", "fireredasr"),
        default="fireredasr2",
        help="模型代际；默认使用 FireRedASR2S 仓库中的 FireRedASR2",
    )
    parser.add_argument("--asr-type", choices=("llm", "aed"), default="llm")
    parser.add_argument(
        "--repository",
        help="自定义上游仓库；默认根据 --model-family 自动选择",
    )
    parser.add_argument(
        "--model-dir",
        help="自定义模型目录；默认根据模型代际和 --asr-type 自动选择",
    )
    parser.add_argument("--dataset", action="append", help="评测集名称，可重复；默认全部")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="每个数据集最多推理的样本数")
    parser.add_argument("--overwrite", action="store_true", help="清空已有结果并重新推理")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--use-half",
        action="store_true",
        help="FireRedASR2 在 GPU 上使用 BF16/FP16；CPU 下不要启用",
    )
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--decode-max-len", type=int, default=0)
    parser.add_argument("--decode-min-len", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=3.0)
    parser.add_argument("--llm-length-penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--softmax-smoothing", type=float, default=1.25)
    parser.add_argument("--aed-length-penalty", type=float, default=0.6)
    parser.add_argument("--eos-penalty", type=float, default=1.0)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> ModelSpec:
    try:
        spec = _model_spec(args.model_family, args.asr_type)
    except ValueError as exc:
        parser.error(str(exc))
    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit 不能为负数")
    if args.beam_size <= 0:
        parser.error("--beam-size 必须大于 0")
    if args.decode_max_len < 0:
        parser.error("--decode-max-len 不能为负数")
    if args.use_half and args.device != "cuda":
        parser.error("--use-half 只能与 --device cuda 一起使用")
    if args.use_half and spec.family != "fireredasr2":
        parser.error("--use-half 仅支持 --model-family fireredasr2")
    if spec.asr_type == "llm":
        if args.decode_min_len < 0:
            parser.error("--decode-min-len 不能为负数")
        if not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0:
            parser.error("--repetition-penalty 必须是正数")
        if not math.isfinite(args.llm_length_penalty):
            parser.error("--llm-length-penalty 必须是有限数值")
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            parser.error("--temperature 必须是正数")
    else:
        if args.nbest <= 0:
            parser.error("--nbest 必须大于 0")
        if not math.isfinite(args.softmax_smoothing) or args.softmax_smoothing <= 0:
            parser.error("--softmax-smoothing 必须是正数")
        if not math.isfinite(args.aed_length_penalty):
            parser.error("--aed-length-penalty 必须是有限数值")
        if not math.isfinite(args.eos_penalty):
            parser.error("--eos-penalty 必须是有限数值")
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    spec = _validate_args(parser, args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    provider = ASRDatasetProvider(args.dataset_root)
    datasets = args.dataset or provider.list_datasets()
    if not datasets:
        parser.error("没有找到可评测的数据集")
    common_decode_options = {
        "use_gpu": args.device == "cuda",
        "use_half": args.use_half,
        "beam_size": args.beam_size,
        "decode_max_len": args.decode_max_len,
    }
    decode_options = (
        {
            **common_decode_options,
            "decode_min_len": args.decode_min_len,
            "repetition_penalty": args.repetition_penalty,
            "llm_length_penalty": args.llm_length_penalty,
            "temperature": args.temperature,
        }
        if spec.asr_type == "llm"
        else {
            **common_decode_options,
            "nbest": args.nbest,
            "softmax_smoothing": args.softmax_smoothing,
            "aed_length_penalty": args.aed_length_penalty,
            "eos_penalty": args.eos_penalty,
        }
    )
    model = load_model(
        args.repository,
        args.model_dir,
        family=args.model_family,
        asr_type=args.asr_type,
        decode_options=decode_options,
    )
    evaluator = DatasetEvaluator(
        model,
        args.output_root,
        batch_size=args.batch_size,
        decode_options=decode_options,
        overwrite=args.overwrite,
        model_id=spec.model_id,
        backend=spec.family,
        asr_type=spec.asr_type,
    )
    summary = {
        dataset: evaluator.evaluate_dataset(provider, dataset, limit=args.limit)
        for dataset in datasets
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DatasetEvaluator",
    "DEFAULT_AED_MODEL_DIR",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_REPOSITORY",
    "load_model",
    "recognize",
    "recognize_aed",
]
