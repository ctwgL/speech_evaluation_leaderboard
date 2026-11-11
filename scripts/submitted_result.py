"""Score saved ASR results with SpeechIO and build leaderboard model JSON.

Every dataset uses one metric selected from its dataset metadata: Chinese
datasets use CER and English datasets use WER. A mixed-language sentence is
therefore evaluated with the metric of its containing dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

METRIC_DIR = Path(__file__).resolve().parent / "speechio_metrics"
METRIC_BY_LANGUAGE = {"zh": "cer", "en": "wer"}
DETAIL_UID = re.compile(r'^\{"uid":(utt_(\d{8})),')
DETAIL_COUNTS = {
    name: re.compile(rf'"{name}":\s*(\d+)')
    for name in ("cor", "sub", "ins", "del")
}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "result", "transcript", "recognition", "sentence"):
            if key in value:
                text = _text(value[key])
                if text:
                    return text
    return ""


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        if "No module named 'pynini'" in detail:
            detail = "英文 SpeechIO 评测需要在当前 Python 环境安装 pynini。" + detail
        raise RuntimeError(f"SpeechIO 命令失败: {' '.join(command[:2])}: {detail}")
    return completed.stdout


def _utt_id(index: int) -> str:
    return f"utt_{index:08d}"


def _load_kaldi_text(path: Path) -> dict[str, str]:
    utts: dict[str, str] = {}
    if not path.exists():
        return utts
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            cols = line.strip().split(maxsplit=1)
            utts[cols[0]] = cols[1] if len(cols) == 2 else ""
    return utts


def _restore_missing_utts(path: Path, count: int, existing: dict[str, str]) -> None:
    missing = [_utt_id(index) for index in range(count) if _utt_id(index) not in existing]
    if not missing:
        return
    with path.open("a", encoding="utf-8") as stream:
        for uid in missing:
            stream.write(f"{uid}\t\n")


def _detail_counts(
    path: Path, count: int
) -> list[tuple[int, int] | None]:
    """Read exact per-utterance edit counts from SpeechIO's details file."""
    scores: list[tuple[int, int] | None] = [None] * count
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            uid_match = DETAIL_UID.match(line)
            if not uid_match:
                continue
            index = int(uid_match.group(2))
            if index >= count:
                raise ValueError(f"SpeechIO 返回了未知句子 ID: {uid_match.group(1)}")
            values = {
                name: pattern.search(line)
                for name, pattern in DETAIL_COUNTS.items()
            }
            if any(match is None for match in values.values()):
                raise ValueError(f"无法解析 SpeechIO 句级编辑数: {line.strip()}")
            parsed = {name: int(match.group(1)) for name, match in values.items()}
            if scores[index] is not None:
                raise ValueError(f"SpeechIO 返回了重复句子 ID: {uid_match.group(1)}")
            scores[index] = (
                parsed["sub"] + parsed["ins"] + parsed["del"],
                parsed["cor"] + parsed["sub"] + parsed["del"],
            )
    return scores


def _score_group_with_details(
    items: list[dict[str, Any]], language: str, unit: str
) -> tuple[tuple[int, int], list[tuple[int, int] | None]]:
    if not items:
        return (0, 0), []
    with tempfile.TemporaryDirectory(prefix="speechio_score_") as directory:
        folder = Path(directory)
        raw_ref, raw_hyp = folder / "raw_ref.txt", folder / "raw_hyp.txt"
        ref, hyp, details = folder / "ref.txt", folder / "hyp.txt", folder / "details.txt"
        with raw_ref.open("w", encoding="utf-8") as refs, raw_hyp.open("w", encoding="utf-8") as hyps:
            for index, item in enumerate(items):
                # Use generated IDs so original IDs cannot break Kaldi text lines.
                uid = _utt_id(index)
                reference = str(item["reference"]).replace("\n", " ").replace("\r", " ")
                hypothesis = (
                    _text(item.get("result"))
                    if item.get("status", "success") == "success"
                    else ""
                ).replace("\n", " ").replace("\r", " ")
                refs.write(f"{uid}\t{reference}\n")
                hyps.write(f"{uid}\t{hypothesis}\n")

        normalizer = METRIC_DIR / f"textnorm_{language}.py"
        options = (["--format=ark", "--to_upper", "--to_banjiao",
                    "--remove_fillers", "--remove_erhua"] if language == "zh"
                   else ["--has_key", "--to_upper"])
        _run([sys.executable, str(normalizer), *options, str(raw_ref), str(ref)])
        _run([sys.executable, str(normalizer), *options, str(raw_hyp), str(hyp)])

        # SpeechIO drops lines that become empty after normalization, and later
        # skips remaining whitespace-only references. Restore missing IDs so
        # failed hypotheses still count as deletions, but only score refs that
        # still have tokens after official textnorm.
        normalized_refs = _load_kaldi_text(ref)
        _restore_missing_utts(ref, len(items), normalized_refs)
        normalized_hyps = _load_kaldi_text(hyp)
        _restore_missing_utts(hyp, len(items), normalized_hyps)
        scorable = sum(1 for index in range(len(items))
                       if normalized_refs.get(_utt_id(index), "").strip())
        skipped = [str(items[index]["sample_id"]) for index in range(len(items))
                   if not normalized_refs.get(_utt_id(index), "").strip()]
        if skipped:
            preview = ", ".join(skipped[:5])
            extra = f" 等 {len(skipped)} 条" if len(skipped) > 5 else ""
            print(f"警告: {language}/{unit} 归一化后参考为空，已按 SpeechIO 跳过: {preview}{extra}",
                  file=sys.stderr)
        if scorable == 0:
            raise ValueError(f"{language}/{unit} 参考文本归一化后全部为空，无法计算有效错误率")

        # SpeechIO's English evaluator has word-level GLM rules; char mode
        # cannot compile those rules. Its Chinese evaluator implements plain
        # character edit distance and also works on normalized English text.
        evaluator_name = "error_rate_en" if language == "en" and unit == "wer" else "error_rate_zh"
        evaluator = METRIC_DIR / evaluator_name
        command = [sys.executable, str(evaluator), "--tokenizer",
                   "char" if unit == "cer" else "whitespace"]
        if evaluator_name == "error_rate_en":
            # error_rate_en resolves GLM relative to its own directory.
            command += ["--glm", "glm_en.csv"]
        stdout = _run([*command, "--ref", str(ref), "--hyp", str(hyp), str(details)])
        for line in stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if {"C", "S", "I", "D", "num_eval_utts"}.issubset(data):
                if data["num_eval_utts"] != scorable:
                    raise ValueError(
                        f"SpeechIO 实际评测条数与可评分条数不一致"
                        f"（{language}/{unit}: eval={data['num_eval_utts']},"
                        f" scorable={scorable}, input={len(items)}）")
                aggregate = (
                    data["S"] + data["I"] + data["D"],
                    data["C"] + data["S"] + data["D"],
                )
                sentence_counts = _detail_counts(details, len(items))
                parsed = [score for score in sentence_counts if score is not None]
                if len(parsed) != scorable:
                    raise ValueError(
                        f"SpeechIO 句级结果数量不一致（{language}/{unit}: "
                        f"details={len(parsed)}, scorable={scorable}）"
                    )
                summed = (
                    sum(errors for errors, _ in parsed),
                    sum(total for _, total in parsed),
                )
                if summed != aggregate:
                    raise ValueError(
                        f"SpeechIO 句级结果与汇总结果不一致（{language}/{unit}: "
                        f"details={summed}, aggregate={aggregate}）"
                    )
                return aggregate, sentence_counts
        raise RuntimeError(f"无法解析 SpeechIO 评测结果（{language}/{unit}）")


def _score_group(items: list[dict[str, Any]], language: str, unit: str) -> tuple[int, int]:
    """Return corpus edit counts while retaining the existing helper API."""
    return _score_group_with_details(items, language, unit)[0]


def _dataset_language(name: str, *metadata: Mapping[str, Any]) -> str:
    """Resolve one language for the whole dataset without inspecting sentences."""
    aliases = {
        "zh": "zh",
        "chinese": "zh",
        "中文": "zh",
        "en": "en",
        "english": "en",
        "英文": "en",
    }
    for source in metadata:
        value = str(source.get("language", "")).strip().lower()
        if value in aliases:
            return aliases[value]

    for source in metadata:
        scenario = str(source.get("scenario", ""))
        matches = {language for marker, language in (("中文", "zh"), ("英文", "en"))
                   if marker in scenario}
        if len(matches) == 1:
            return matches.pop()
        if len(matches) > 1:
            raise ValueError(f"数据集语言标注不明确: {name}: {scenario}")

    upper_name = name.upper()
    if upper_name.startswith(("LIBRISPEECH_", "TEDLIUM_")):
        return "en"
    if upper_name.startswith(("AISHELL", "ALIMEETING_", "KING-ASR-",
                              "SPEECHIO_ASR_ZH")):
        return "zh"
    raise ValueError(
        f"无法确定数据集语言: {name}；请在结果 JSON 或模板数据集中设置 "
        "language=zh/en，或在 scenario 中标注中文/英文"
    )


def _counts(
    items: list[dict[str, Any]], language: str, *, annotate_sentences: bool = False
) -> dict[str, tuple[int, int]]:
    if not items:
        raise ValueError("没有可评测的识别结果")
    if language not in METRIC_BY_LANGUAGE:
        raise ValueError(f"不支持的数据集语言: {language}")
    sample_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not item.get("sample_id") or not str(item.get("reference", "")).strip():
            raise ValueError("识别记录缺少 sample_id 或非空 reference")
        sample_id = str(item["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"识别记录包含重复 sample_id: {sample_id}")
        sample_ids.add(sample_id)
    unit = METRIC_BY_LANGUAGE[language]
    if annotate_sentences:
        count, sentence_counts = _score_group_with_details(items, language, unit)
        for item, score in zip(items, sentence_counts):
            item["sentence_metric"] = {
                "name": unit,
                "value": (
                    round(100 * score[0] / score[1], 4)
                    if score is not None and score[1]
                    else None
                ),
                "errors": score[0] if score is not None else None,
                "reference_tokens": score[1] if score is not None else None,
            }
    else:
        count = _score_group(items, language, unit)
    if count[1] == 0:
        raise ValueError(f"{language}/{unit} 的参考文本没有可评测 token")
    return {unit: count}


def _write_result_json(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically persist sentence metrics back to a source result JSON."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.metrics.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _metrics(counts: Mapping[str, tuple[int, int] | list[int]]) -> dict[str, float]:
    """Convert edit counts to submission metrics.

    Request timing belongs to raw inference results and is intentionally not
    copied into leaderboard submissions.
    """
    return {
        key: round(100 * errors / total, 4)
        for key, (errors, total) in counts.items()
    }


def calculate_dataset(
    data: dict[str, Any], language: str | None = None
) -> dict[str, float]:
    items = data.get("results")
    if not isinstance(items, list):
        raise ValueError("结果文件缺少 results 数组")
    dataset_name = str(data.get("dataset", ""))
    selected = language or _dataset_language(dataset_name, data)
    return _metrics(_counts(items, selected))


def build_submission(result_root: Path | str, template: Path | str) -> dict[str, Any]:
    root, template = Path(result_root), Path(template)
    model = json.loads(template.read_text(encoding="utf-8"))
    known = {item["name"]: item for item in model.get("datasets", [])}
    paths = sorted(root.glob("*.json"))
    if not paths:
        raise ValueError(f"结果目录没有数据集 JSON: {root}")
    datasets = []
    aggregate_counts: dict[str, list[int]] = {}
    names = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("dataset")
        if not isinstance(name, str) or not name or name != path.stem or name in names:
            raise ValueError(f"数据集名称与文件名不匹配或重复: {path}")
        names.add(name)
        items = data.get("results")
        if not isinstance(items, list):
            raise ValueError(f"结果文件缺少 results 数组: {path}")
        stats = data.get("stats")
        if isinstance(stats, dict):
            recorded = sum(int(stats.get(key, 0)) for key in ("success", "failed", "skipped"))
            if recorded != len(items):
                raise ValueError(f"结果文件漏记样本，不能计算可信错误率: {path}")
        if any(item.get("dataset", name) != name for item in items):
            raise ValueError(f"样本数据集名称不匹配: {path}")
        template_dataset = known.get(name, {})
        dataset = {key: value for key, value in template_dataset.items()
                   if key not in ("metrics", "name")}
        try:
            language = _dataset_language(name, data, template_dataset)
            counts = _counts(items, language, annotate_sentences=True)
            _write_result_json(path, data)
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f"{path}: {exc}") from exc
        for unit, (errors, total) in counts.items():
            aggregate_counts.setdefault(unit, [0, 0])
            aggregate_counts[unit][0] += errors
            aggregate_counts[unit][1] += total
        dataset.update({"name": name, "scenario": dataset.get("scenario", ""),
                        "size": dataset.get("size", ""), "metrics": _metrics(counts)})
        datasets.append(dataset)
    model["datasets"] = datasets
    model["metrics"] = _metrics(aggregate_counts)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用 SpeechIO 按中文 CER、英文 WER 计算 ASR 指标并生成榜单 JSON")
    parser.add_argument("--result-root", required=True)
    parser.add_argument(
        "--template",
        default=str(
            Path(__file__).resolve().parents[1]
            / "dist/data/models/asr/template.json"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = build_submission(args.result_root, args.template)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
