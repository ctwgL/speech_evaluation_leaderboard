"""Provide paired audio for the V5 personalized speech-enhancement sets.

Each sample contains the enrollment audio, its same-named noisy input, and the
    location where an enhancement system should write (or has written) its output.
The enhanced directory mirrors the partition/track layout so files from the
blind and test sets cannot collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal


DEFAULT_BLIND_ROOT = Path(
    "/mnt/lustre/chentingwei/cosyVoice/speech_evaluation_dataset/"
    "V5_BlindTestSet_mono"
)
DEFAULT_TEST_ROOT = Path(
    "/mnt/lustre/chentingwei/cosyVoice/speech_evaluation_dataset/"
    "V5_dev_testset"
)

Partition = Literal["blind", "test"]
TRACKS = ("Track1_Headset", "Track2_Speakerphone")
DATASETS = tuple(
    f"{partition}/{track}"
    for partition in ("blind", "test")
    for track in TRACKS
)


@dataclass(frozen=True)
class SEDatasetSpec:
    """Identify one partition/track combination."""

    name: str
    partition: Partition
    track: str


@dataclass(frozen=True)
class SESample:
    """One paired speech-enhancement evaluation sample."""

    sample_id: str
    dataset: str
    partition: Partition
    track: str
    enrol_audio_path: Path
    noisy_audio_path: Path
    enhanced_audio_path: Path

    @property
    def filename(self) -> str:
        """Input filename shared by enrollment and noisy audio."""
        return self.noisy_audio_path.name

    @property
    def audio_path(self) -> Path:
        """Compatibility alias: the audio to process is the noisy input."""
        return self.noisy_audio_path


class SEDatasetProvider:
    """Read and pair the V5 blind/test speech-enhancement datasets.

    ``enhanced_root`` is model/result specific.  Enhanced paths are generated
    below ``<enhanced_root>/<partition>/<track>/``.  The provider never
    creates or overwrites those files.  Set ``require_enhanced=True`` when
    loading completed outputs for metric calculation.  ``enhanced_suffix`` can
    be used when a client returns a different format, such as MP3.
    """

    def __init__(
        self,
        enhanced_root: Path | str,
        *,
        blind_root: Path | str = DEFAULT_BLIND_ROOT,
        test_root: Path | str = DEFAULT_TEST_ROOT,
        require_enhanced: bool = False,
        enhanced_suffix: str | None = None,
    ) -> None:
        self.enhanced_root = Path(enhanced_root).expanduser().resolve()
        self.roots: dict[Partition, Path] = {
            "blind": Path(blind_root).expanduser().resolve(),
            "test": Path(test_root).expanduser().resolve(),
        }
        self.require_enhanced = require_enhanced
        if enhanced_suffix is not None and not enhanced_suffix.startswith("."):
            enhanced_suffix = f".{enhanced_suffix}"
        self.enhanced_suffix = enhanced_suffix

        for partition, root in self.roots.items():
            if not root.is_dir():
                raise FileNotFoundError(
                    f"语音增强{self._partition_label(partition)}目录不存在: {root}"
                )
            for track in TRACKS:
                for kind in ("enrol", "noisy"):
                    directory = root / track / kind
                    if not directory.is_dir():
                        raise FileNotFoundError(f"语音增强数据目录不存在: {directory}")

    @staticmethod
    def _partition_label(partition: Partition) -> str:
        return "盲测集" if partition == "blind" else "测试集"

    @staticmethod
    def _parse_dataset(dataset: str) -> SEDatasetSpec:
        try:
            partition_value, track = dataset.split("/", 1)
        except ValueError as exc:
            raise ValueError(
                f"无效语音增强数据集: {dataset!r}；可选值: {', '.join(DATASETS)}"
            ) from exc
        if partition_value not in ("blind", "test") or track not in TRACKS:
            raise ValueError(
                f"无效语音增强数据集: {dataset!r}；可选值: {', '.join(DATASETS)}"
            )
        partition: Partition = partition_value
        return SEDatasetSpec(dataset, partition, track)

    @staticmethod
    def _wav_files(directory: Path) -> dict[str, Path]:
        files = {
            path.name: path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".wav"
        }
        if not files:
            raise ValueError(f"语音增强数据目录中没有 WAV 文件: {directory}")
        return files

    def list_datasets(self) -> list[str]:
        """Return the four independently evaluable partition/track IDs."""
        return list(DATASETS)

    def get_dataset_spec(self, dataset: str) -> SEDatasetSpec:
        """Return the partition and track represented by ``dataset``."""
        return self._parse_dataset(dataset)

    def output_directory(self, dataset: str) -> Path:
        """Return the enhanced-audio directory for a dataset."""
        spec = self._parse_dataset(dataset)
        return self.enhanced_root / spec.partition / spec.track

    def _paired_files(self, spec: SEDatasetSpec) -> list[tuple[str, Path, Path]]:
        track_root = self.roots[spec.partition] / spec.track
        enrol = self._wav_files(track_root / "enrol")
        noisy = self._wav_files(track_root / "noisy")

        enrol_only = sorted(enrol.keys() - noisy.keys())
        noisy_only = sorted(noisy.keys() - enrol.keys())
        if enrol_only or noisy_only:
            details = []
            if enrol_only:
                details.append(
                    f"仅 enrol 存在 {len(enrol_only)} 个（如 {enrol_only[0]}）"
                )
            if noisy_only:
                details.append(
                    f"仅 noisy 存在 {len(noisy_only)} 个（如 {noisy_only[0]}）"
                )
            raise ValueError(f"{spec.name} 文件无法一一配对: {'；'.join(details)}")

        return [(name, enrol[name], noisy[name]) for name in sorted(noisy)]

    def iter_samples(
        self, dataset: str, *, limit: int | None = None
    ) -> Iterator[SESample]:
        """Yield paired samples from one partition/track.

        ``limit`` is intended for smoke tests.  When ``require_enhanced`` is
        enabled, only the selected samples are required to have output files.
        """
        if limit is not None and limit < 0:
            raise ValueError("limit 不能为负数")
        if limit == 0:
            return

        spec = self._parse_dataset(dataset)
        pairs = self._paired_files(spec)
        if limit is not None:
            pairs = pairs[:limit]

        output_directory = self.output_directory(dataset)
        samples = [
            SESample(
                sample_id=Path(filename).stem,
                dataset=spec.name,
                partition=spec.partition,
                track=spec.track,
                enrol_audio_path=enrol_path,
                noisy_audio_path=noisy_path,
                enhanced_audio_path=output_directory / (
                    f"{Path(filename).stem}{self.enhanced_suffix}"
                    if self.enhanced_suffix is not None
                    else filename
                ),
            )
            for filename, enrol_path, noisy_path in pairs
        ]

        if self.require_enhanced:
            missing = [
                sample.enhanced_audio_path
                for sample in samples
                if not sample.enhanced_audio_path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{dataset} 缺少 {len(missing)} 个增强结果（如 {missing[0]}）"
                )

        yield from samples

    def get_samples(
        self, dataset: str, *, limit: int | None = None
    ) -> list[SESample]:
        return list(self.iter_samples(dataset, limit=limit))

    def iter_all_samples(self, *, limit_per_dataset: int | None = None) -> Iterator[SESample]:
        """Yield samples from all four datasets without merging their identity."""
        for dataset in self.list_datasets():
            yield from self.iter_samples(dataset, limit=limit_per_dataset)


def get_default_provider(
    enhanced_root: Path | str,
    *,
    require_enhanced: bool = False,
    enhanced_suffix: str | None = None,
) -> SEDatasetProvider:
    return SEDatasetProvider(
        enhanced_root,
        require_enhanced=require_enhanced,
        enhanced_suffix=enhanced_suffix,
    )


__all__ = [
    "DATASETS",
    "DEFAULT_BLIND_ROOT",
    "DEFAULT_TEST_ROOT",
    "TRACKS",
    "SEDatasetProvider",
    "SEDatasetSpec",
    "SESample",
    "get_default_provider",
]
