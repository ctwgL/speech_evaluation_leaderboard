# SpeechBench Lab

面向开源/闭源语音模型与 API 的统一评测可视化榜单。目前首版覆盖 ASR、TTS，页面数据为界面演示样例，后续可接入真实评测结果。

## 本地预览

```bash
python3 -m http.server 8000 -d dist
```

访问 `http://localhost:8000`。

## GitHub Pages

仓库已配置 `.github/workflows/deploy-pages.yml`。推送到 `master` 或 `main` 后会自动发布 `dist/`，预期地址：

<https://ctwgL.github.io/speech_evaluation_leaderboard/>

首次使用时，请在仓库 **Settings → Pages → Build and deployment** 中将 Source 设为 **GitHub Actions**。
