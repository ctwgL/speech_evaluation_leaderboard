"""Provide Seed-TTS Eval samples to synthesis clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Mapping


DEFAULT_SEED_TTS_ROOT = Path(
    "/mnt/lustre/chentingwei/cosyVoice/speech_evaluation_dataset/"
    "seed_tts_eval_dataset/seedtts_testset"
)
Family = Literal["seed_tts_eval"]

SEED_DATASETS = ("seed_tts_eval/zh", "seed_tts_eval/en")
DATASETS = SEED_DATASETS


@dataclass(frozen=True)
class TTSDatasetSpec:
    """Identify one independently runnable TTS benchmark group."""

    name: str
    family: Family
    subset: str


@dataclass(frozen=True)
class TTSSample:
    """One zero-shot speech-synthesis request and its evaluation context."""

    sample_id: str
    dataset: str
    family: Family
    subset: str
    language: str
    target_text: str
    prompt_text: str
    prompt_audio_path: Path
    ground_truth_audio_path: Path | None = None
    generated_audio_path: Path | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Compatibility alias for the text that the model must synthesize."""
        return self.target_text

    @property
    def audio_path(self) -> Path:
        """Compatibility alias for the voice-reference audio."""
        return self.prompt_audio_path

    @property
    def reference_audio_path(self) -> Path:
        """Return the prompt audio used for zero-shot voice conditioning."""
        return self.prompt_audio_path


class TTSDatasetProvider:
    """Read Seed-TTS Eval without modifying source data.

    ``generated_root`` is optional.  When supplied, each sample receives a
    deterministic path under ``<generated_root>/<dataset>/<sample_id>``.
    The provider does not create or overwrite generated files.  Set
    ``require_generated=True`` when loading completed model outputs for
    evaluation.
    """

    def __init__(
        self,
        generated_root: Path | str | None = None,
        *,
        seed_tts_root: Path | str = DEFAULT_SEED_TTS_ROOT,
        generated_suffix: str = ".wav",
        require_generated: bool = False,
    ) -> None:
        self.seed_tts_root = Path(seed_tts_root).expanduser().resolve()
        self.generated_root = (
            Path(generated_root).expanduser().resolve()
            if generated_root is not None
            else None
        )
        if not generated_suffix.startswith("."):
            generated_suffix = f".{generated_suffix}"
        if generated_suffix == ".":
            raise ValueError("generated_suffix 不能为空")
        self.generated_suffix = generated_suffix
        self.require_generated = require_generated

        if not self.seed_tts_root.is_dir():
            raise FileNotFoundError(f"Seed-TTS Eval 数据目录不存在: {self.seed_tts_root}")
        for language in ("zh", "en"):
            directory = self.seed_tts_root / language
            if not directory.is_dir() or not (directory / "meta.lst").is_file():
                raise FileNotFoundError(f"Seed-TTS Eval 元数据不存在: {directory / 'meta.lst'}")
        if require_generated and self.generated_root is None:
            raise ValueError("require_generated=True 时必须提供 generated_root")

    def list_datasets(self) -> list[str]:
        """Return the Chinese and English Seed-TTS task IDs."""
        return list(DATASETS)

    @staticmethod
    def _parse_dataset(dataset: str) -> TTSDatasetSpec:
        if dataset in SEED_DATASETS:
            return TTSDatasetSpec(dataset, "seed_tts_eval", dataset.split("/", 1)[1])
        raise ValueError(
            f"无效语音合成数据集: {dataset!r}；可选值: {', '.join(DATASETS)}"
        )

    def get_dataset_spec(self, dataset: str) -> TTSDatasetSpec:
        return self._parse_dataset(dataset)

    @staticmethod
    def _resolve_file(root: Path, value: str, *, source: Path) -> Path:
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError(f"{source} 包含绝对路径: {value}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"{source} 中的路径超出数据目录: {value}")
        if not path.is_file():
            raise FileNotFoundError(f"{source} 引用的音频不存在: {path}")
        return path

    def output_directory(self, dataset: str) -> Path | None:
        self._parse_dataset(dataset)
        if self.generated_root is None:
            return None
        return self.generated_root.joinpath(*dataset.split("/"))

    def _generated_path(self, dataset: str, sample_id: str) -> Path | None:
        directory = self.output_directory(dataset)
        if directory is None:
            return None
        relative = Path(sample_id)
        if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise ValueError(f"无效 TTS sample_id: {sample_id!r}")
        path = Path(f"{directory / relative}{self.generated_suffix}").resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"TTS 输出路径超出结果目录: {sample_id!r}")
        return path

    def _seed_samples(self, spec: TTSDatasetSpec) -> Iterator[TTSSample]:
        directory = self.seed_tts_root / spec.subset
        metadata_path = directory / "meta.lst"
        seen: set[str] = set()
        with metadata_path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                columns = line.rstrip("\r\n").split("|", 3)
                if len(columns) != 4 or not all(value.strip() for value in columns):
                    raise ValueError(f"{metadata_path} 第 {line_number} 行字段无效")
                sample_id, prompt_text, prompt_value, target_text = (
                    value.strip() for value in columns
                )
                if sample_id in seen:
                    raise ValueError(f"{metadata_path} 包含重复 ID: {sample_id}")
                seen.add(sample_id)
                prompt_path = self._resolve_file(
                    directory, prompt_value, source=metadata_path
                )
                ground_truth = self._resolve_file(
                    directory,
                    f"wavs/{sample_id}.wav",
                    source=metadata_path,
                )
                yield TTSSample(
                    sample_id=sample_id,
                    dataset=spec.name,
                    family=spec.family,
                    subset=spec.subset,
                    language=spec.subset,
                    target_text=target_text,
                    prompt_text=prompt_text,
                    prompt_audio_path=prompt_path,
                    ground_truth_audio_path=ground_truth,
                    generated_audio_path=self._generated_path(spec.name, sample_id),
                    metadata={"source": str(metadata_path), "group": spec.subset},
                )

    def iter_samples(
        self, dataset: str, *, limit: int | None = None
    ) -> Iterator[TTSSample]:
        """Yield samples from one TTS benchmark group."""
        if limit is not None and limit < 0:
            raise ValueError("limit 不能为负数")
        if limit == 0:
            return
        spec = self._parse_dataset(dataset)
        for index, sample in enumerate(self._seed_samples(spec)):
            if limit is not None and index >= limit:
                break
            if self.require_generated:
                assert sample.generated_audio_path is not None
                if not sample.generated_audio_path.is_file():
                    raise FileNotFoundError(
                        f"{dataset} 缺少合成结果: {sample.generated_audio_path}"
                    )
            yield sample

    def get_samples(
        self, dataset: str, *, limit: int | None = None
    ) -> list[TTSSample]:
        return list(self.iter_samples(dataset, limit=limit))

    def iter_all_samples(
        self, *, limit_per_dataset: int | None = None
    ) -> Iterator[TTSSample]:
        for dataset in self.list_datasets():
            yield from self.iter_samples(dataset, limit=limit_per_dataset)


def get_default_provider(
    generated_root: Path | str | None = None,
    *,
    require_generated: bool = False,
    generated_suffix: str = ".wav",
) -> TTSDatasetProvider:
    return TTSDatasetProvider(
        generated_root,
        require_generated=require_generated,
        generated_suffix=generated_suffix,
    )


__all__ = [
    "DATASETS",
    "DEFAULT_SEED_TTS_ROOT",
    "SEED_DATASETS",
    "TTSDatasetProvider",
    "TTSDatasetSpec",
    "TTSSample",
    "get_default_provider",
]
