# SpeechBench Lab

面向开源/闭源语音模型与 API 的统一评测可视化榜单。目前覆盖 ASR 与 TTS，支持总览、单模型分测试集指标和 2–3 个同任务模型的全指标视觉对比。页面提供浅色/暗色主题、指标热力与进度条、排名变化动效以及归一化多指标雷达图。

## 本地预览

```bash
node scripts/build-data-index.cjs
python3 -m http.server 8000 -d dist
```

访问 `http://localhost:8000`。

## 更新评测结果

数据按模型独立维护，目录结构如下：

```text
dist/data/
├── config.json                   # 数据格式版本、ASR/TTS 排名指标
├── index.json                    # 脚本生成的文件索引，不手动维护
└── models/
    ├── asr/paraformer-large.json  # 一个文件对应一个模型
    └── tts/cosyvoice-2.json
```

新增或修改模型时，编辑 `models/` 下的 JSON，再运行 `node scripts/build-data-index.cjs`。脚本递归扫描所有模型文件，校验数据并生成只包含文件路径和全局配置的轻量索引，不会把模型内容合成大文件。GitHub Pages 部署前会自动执行此步骤；本地预览也需先运行。删除或重命名文件后同样重新生成索引。

页面读取索引后并行加载各模型文件，自动汇总榜单、详情和对比数据。静态托管不能直接枚举目录，因此通过自动生成索引发现新文件。任一文件读取失败会提示错误，避免悄悄遗漏模型导致榜单不完整。

单个模型文件示例（可复制现有模型文件作为模板）：

```json
{
  "task": "asr",
  "name": "Example ASR",
  "org": "Example Team",
  "kind": "open",
  "logo": "EA",
  "cost": "免费",
  "metrics": { "wer": 4.12, "cer": 3.76, "rtf": 0.06, "latency_ms": 82 },
  "datasets": [
    {
      "name": "Example Dataset",
      "scenario": "中文 · 普通话",
      "size": "10h",
      "metrics": { "wer": 4.12, "cer": 3.76 }
    }
  ],
  "introduction": {
    "summary": "模型简介",
    "specifications": { "模型架构": "Transformer", "参数量": "100M" },
    "architecture": {
      "image": "./assets/example-architecture.svg",
      "alt": "模型架构图",
      "caption": "架构说明及来源"
    },
    "links": [{ "label": "项目主页", "url": "https://example.com" }]
  }
}
```

`task` 为 `asr` 或 `tts`，`name` 全局唯一，`kind` 为 `open` 或 `api`。指标键使用小写，值为数值，缺失指标直接省略；`metrics` 必须包含 `config.json` 的 `rankingMetrics` 对应指标。`metrics` 保存综合评测结果，`datasets[].metrics` 保存单测试集结果，页面不会自动取平均值。`introduction` 保存模型介绍、规格、架构图及参考链接。建议文件名使用稳定的英文小写短横线命名。

排序方向已内置：`WER`、`CER`、`RTF`、延迟和成本越小越好；`MOS`、`SIM`、准确率、`PESQ`、`STOI`、`SI-SDR` 等越大越好。

## 数据校验

```bash
node scripts/build-data-index.cjs
node tests/fixed-data-smoke.cjs
node tests/model-profile.cjs
node tests/data-loader.cjs
```

## GitHub Pages

仓库已配置 `.github/workflows/deploy-pages.yml`。推送到 `master` 或 `main` 后会自动发布 `dist/`，预期地址：

<https://ctwgL.github.io/speech_evaluation_leaderboard/>

首次使用时，请在仓库 **Settings → Pages → Build and deployment** 中将 Source 设为 **GitHub Actions**。

## 模型介绍与详情页

每个模型的指标和介绍维护在自己的 `dist/data/models/<任务>/<模型名>.json` 文件中。新增模型无需修改 HTML 或 JavaScript，生成索引即可。

每个模型的 `introduction` 示例：

```json
{
  "summary": "模型基本介绍，可使用换行分段。",
  "specifications": {
    "模型架构": "填写架构名称",
    "参数量": "填写参数量",
    "模型文件大小": "填写文件大小",
    "支持语言": "填写支持的语言",
    "许可证": "填写许可证"
  },
  "architecture": {
    "image": "./assets/example-architecture.svg",
    "alt": "模型架构图",
    "caption": "架构图说明及来源"
  },
  "links": [
    { "label": "项目主页", "url": "https://example.com" }
  ]
}
```

规格字段可以自由增减。架构图路径相对于页面地址，也可使用 HTTPS 图片地址；将图片放入 `dist/assets/`。空字段显示待补充，未提供架构图时保留展示区域。仓库现有模型介绍和规格尚待填入真实资料。

`metrics` 保存模型全部综合指标；`datasets[].metrics` 保存各测试集指标，详情表格自动取指标并集，缺失值显示 `—`，新增指标会自动展示。测试集的 `name`、`scenario`、`size` 也均来自此文件。

详情页使用 `#model/模型名称` 路由，支持复制链接、刷新后直接打开，以及浏览器前进/后退。
