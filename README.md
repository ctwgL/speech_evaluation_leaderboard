# SpeechBench Lab

面向开源/闭源语音模型与 API 的统一评测可视化榜单。目前覆盖 ASR 与 TTS，支持总览、单模型分测试集指标和 2–3 个同任务模型的全指标视觉对比。

## 本地预览

```bash
python3 -m http.server 8000 -d dist
```

访问 `http://localhost:8000`。

## 更新评测结果

页面只读取仓库内的固定文件：

- `dist/data/benchmark-results.json`

后续更新榜单时直接编辑该文件，无文件上传入口。`rankingMetrics` 指定 ASR/TTS 排名指标；`models[].metrics` 保存综合指标；`models[].datasets[].metrics` 保存各测试集的实际指标。

排序方向已内置：`WER`、`CER`、`RTF`、延迟和成本越小越好；`MOS`、`SIM`、准确率、`PESQ`、`STOI`、`SI-SDR` 等越大越好。

## 数据校验

```bash
node tests/fixed-data-smoke.cjs
```

## GitHub Pages

仓库已配置 `.github/workflows/deploy-pages.yml`。推送到 `master` 或 `main` 后会自动发布 `dist/`，预期地址：

<https://ctwgL.github.io/speech_evaluation_leaderboard/>

首次使用时，请在仓库 **Settings → Pages → Build and deployment** 中将 Source 设为 **GitHub Actions**。
