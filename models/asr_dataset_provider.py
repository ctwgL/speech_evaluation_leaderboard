"""Read SpeechIO ASR datasets for vendor evaluation clients."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

DEFAULT_DATASET_ROOT = Path(
    "/mnt/lustre/chentingwei/cosyVoice/speech_evaluation_dataset/speechio/datasets"
)


@dataclass(frozen=True)
class ASRSample:
    sample_id: str
    audio_path: Path
    reference: str
    duration: float | None = None
    dataset: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.reference


class ASRDatasetProvider:
    def __init__(self, root: Path | str = DEFAULT_DATASET_ROOT) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"ASR 数据集目录不存在: {self.root}")

    def list_datasets(self) -> list[str]:
        return sorted(path.name for path in self.root.iterdir() if path.is_dir())

    def _directory(self, dataset: str) -> Path:
        directory = (self.root / dataset).resolve()
        if directory.parent != self.root or not directory.is_dir():
            raise ValueError(f"无效数据集名称: {dataset}")
        return directory

    @staticmethod
    def _audio_path(directory: Path, value: str) -> Path:
        path = (directory / value).resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"音频路径超出数据集目录: {value}")
        return path

    def _metadata(self, directory: Path) -> Iterator[ASRSample]:
        with (directory / "metadata.tsv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            required = {"ID", "AUDIO", "TEXT"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"{directory}/metadata.tsv 缺少 {sorted(required)} 字段")
            for line_number, row in enumerate(reader, 2):
                if not row or not any(row.values()):
                    continue
                if not row.get("ID") or not row.get("AUDIO") or row.get("TEXT") is None:
                    raise ValueError(f"{directory}/metadata.tsv 第 {line_number} 行无效")
                duration = row.get("DURATION")
                yield ASRSample(
                    sample_id=row["ID"],
                    audio_path=self._audio_path(directory, row["AUDIO"]),
                    reference=row["TEXT"],
                    duration=float(duration) if duration else None,
                    dataset=directory.name,
                    metadata={key.lower(): value for key, value in row.items() if key and value is not None},
                )

    def _kaldi(self, directory: Path) -> Iterator[ASRSample]:
        def read(name: str) -> dict[str, str]:
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open(encoding="utf-8") as stream:
                return {
                    parts[0]: parts[1] for line in stream if line.strip()
                    for parts in [line.rstrip("\n").split(None, 1)]
                }

        audio = read("wav.scp")
        references = read("trans.txt")
        missing = audio.keys() - references.keys()
        if missing:
            raise ValueError(f"{directory}/trans.txt 缺少 {len(missing)} 条转写")
        for sample_id, value in audio.items():
            yield ASRSample(sample_id, self._audio_path(directory, value),
                            references[sample_id], dataset=directory.name)

    def iter_samples(self, dataset: str, *, limit: int | None = None) -> Iterator[ASRSample]:
        if limit is not None and limit < 0:
            raise ValueError("limit 不能为负数")
        directory = self._directory(dataset)
        if limit == 0:
            return
        source = self._metadata(directory) if (directory / "metadata.tsv").is_file() else self._kaldi(directory)
        for index, sample in enumerate(source):
            if limit is not None and index >= limit:
                break
            yield sample

    def get_samples(self, dataset: str, *, limit: int | None = None) -> list[ASRSample]:
        return list(self.iter_samples(dataset, limit=limit))


def get_default_provider() -> ASRDatasetProvider:
    return ASRDatasetProvider()
