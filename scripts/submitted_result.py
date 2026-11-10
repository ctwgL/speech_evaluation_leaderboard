"""Score saved ASR results with SpeechIO and build leaderboard model JSON.

Both CER and WER are computed for each language group using SpeechIO's
normalizer and evaluator. Chinese WER uses whitespace tokens as requested;
SpeechIO's official Chinese ranking metric is CER.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

METRIC_DIR = Path(__file__).resolve().parent / "speechio_metrics"
HAN = re.compile(r"[\u3400-\u9fff]")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "result", "transcript", "recognition", "sentence"):
            if key in value:
                return _text(value[key])
    return ""


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        if "No module named 'pynini'" in detail:
            detail = "英文 SpeechIO 评测需要在当前 Python 环境安装 pynini。" + detail
        raise RuntimeError(f"SpeechIO 命令失败: {' '.join(command[:2])}: {detail}")
    return completed.stdout


def _score_group(items: list[dict[str, Any]], language: str, unit: str) -> tuple[int, int]:
    if not items:
        return 0, 0
    with tempfile.TemporaryDirectory(prefix="speechio_score_") as directory:
        folder = Path(directory)
        raw_ref, raw_hyp = folder / "raw_ref.txt", folder / "raw_hyp.txt"
        ref, hyp, details = folder / "ref.txt", folder / "hyp.txt", folder / "details.txt"
        with raw_ref.open("w", encoding="utf-8") as refs, raw_hyp.open("w", encoding="utf-8") as hyps:
            for index, item in enumerate(items):
                # Use generated IDs so original IDs cannot break Kaldi text lines.
                uid = f"utt_{index:08d}"
                reference = str(item["reference"]).replace("\n", " ").replace("\r", " ")
                hypothesis = _text(item.get("result")).replace("\n", " ").replace("\r", " ")
                refs.write(f"{uid}\t{reference}\n")
                hyps.write(f"{uid}\t{hypothesis}\n")

        normalizer = METRIC_DIR / f"textnorm_{language}.py"
        options = (["--format=ark", "--to_upper", "--to_banjiao",
                    "--remove_fillers", "--remove_erhua"] if language == "zh"
                   else ["--has_key", "--to_upper"])
        _run([sys.executable, str(normalizer), *options, str(raw_ref), str(ref)])
        _run([sys.executable, str(normalizer), *options, str(raw_hyp), str(hyp)])

        with ref.open(encoding="utf-8") as stream:
            normalized_refs = {line.split(None, 1)[0] for line in stream if line.strip()}
        if len(normalized_refs) != len(items):
            raise ValueError("参考文本归一化后存在空文本，无法计算有效错误率")
        # The SpeechIO normalizer may omit empty hypotheses. Restore them as
        # empty transcripts so failed requests count as full deletions.
        with hyp.open(encoding="utf-8") as stream:
            normalized_hyps = {line.split(None, 1)[0] for line in stream if line.strip()}
        with hyp.open("a", encoding="utf-8") as stream:
            for index in range(len(items)):
                uid = f"utt_{index:08d}"
                if uid not in normalized_hyps:
                    stream.write(f"{uid}\t\n")

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
                if data["num_eval_utts"] != len(items):
                    raise ValueError("SpeechIO 实际评测条数与输入条数不一致")
                return data["S"] + data["I"] + data["D"], data["C"] + data["S"] + data["D"]
        raise RuntimeError("无法解析 SpeechIO 评测结果")


def _counts(items: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    if not items:
        raise ValueError("没有可评测的识别结果")
    groups = {"zh": [], "en": []}
    for item in items:
        if not isinstance(item, dict) or not item.get("sample_id") or not str(item.get("reference", "")).strip():
            raise ValueError("识别记录缺少 sample_id 或非空 reference")
        groups["zh" if HAN.search(str(item["reference"])) else "en"].append(item)
    counts = {}
    for unit in ("cer", "wer"):
        parts = [_score_group(group, language, unit) for language, group in groups.items() if group]
        counts[unit] = (sum(errors for errors, _ in parts), sum(total for _, total in parts))
        if counts[unit][1] == 0:
            raise ValueError(f"{unit.upper()} 的参考文本没有可评测 token")
    return counts


def _metrics(items: list[dict[str, Any]], counts: dict[str, tuple[int, int]] | None = None) -> dict[str, float]:
    counts = counts if counts is not None else _counts(items)
    metrics = {key: round(100 * errors / total, 4)
               for key, (errors, total) in counts.items()}
    successful = [item for item in items if item.get("status", "success") == "success"
                  and _text(item.get("result")).strip()]
    latencies = [float(item["latency_ms"]) for item in successful
                 if item.get("latency_ms") is not None]
    if latencies:
        if not all(math.isfinite(value) and value >= 0 for value in latencies):
            raise ValueError("latency_ms 必须为非负有限数值")
        metrics["latency_ms"] = round(sum(latencies) / len(latencies), 3)
    durations = [float(item["duration"]) for item in successful
                 if item.get("duration") is not None and item.get("latency_ms") is not None]
    if durations:
        if not all(math.isfinite(value) and value >= 0 for value in durations):
            raise ValueError("duration 必须为非负有限数值")
        total_audio = sum(durations)
        if total_audio > 0:
            paired_latency = sum(float(item["latency_ms"]) for item in successful
                                 if item.get("duration") is not None and item.get("latency_ms") is not None)
            metrics["rtf"] = round(paired_latency / 1000 / total_audio, 4)
    return metrics


def calculate_dataset(data: dict[str, Any]) -> dict[str, float]:
    items = data.get("results")
    if not isinstance(items, list):
        raise ValueError("结果文件缺少 results 数组")
    return _metrics(items)


def build_submission(result_root: Path | str, template: Path | str) -> dict[str, Any]:
    root, template = Path(result_root), Path(template)
    model = json.loads(template.read_text(encoding="utf-8"))
    known = {item["name"]: item for item in model.get("datasets", [])}
    paths = sorted(root.glob("*.json"))
    if not paths:
        raise ValueError(f"结果目录没有数据集 JSON: {root}")
    datasets = []
    all_results = []
    aggregate_counts = {"cer": [0, 0], "wer": [0, 0]}
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
        all_results.extend(items)
        dataset = {key: value for key, value in known.get(name, {}).items()
                   if key not in ("metrics", "name")}
        counts = _counts(items)
        for unit, (errors, total) in counts.items():
            aggregate_counts[unit][0] += errors
            aggregate_counts[unit][1] += total
        dataset.update({"name": name, "scenario": dataset.get("scenario", ""),
                        "size": dataset.get("size", ""), "metrics": _metrics(items, counts)})
        datasets.append(dataset)
    model["datasets"] = datasets
    model["metrics"] = _metrics(all_results, aggregate_counts)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 SpeechIO 计算 ASR CER/WER 并生成榜单 JSON")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "dist/data/models/asr/test-speech.json"))
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
