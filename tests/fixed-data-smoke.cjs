const fs = require('fs');

const file = 'dist/data/benchmark-results.json';
const payload = JSON.parse(fs.readFileSync(file, 'utf8'));
const html = fs.readFileSync('dist/index.html', 'utf8');
const app = fs.readFileSync('dist/app.js', 'utf8');
const models = payload.models || [];
const asr = models.filter(model => model.task === 'asr');
const tts = models.filter(model => model.task === 'tts');

if (payload.rankingMetrics?.asr !== 'wer') throw new Error('ASR ranking metric must be WER');
if (payload.rankingMetrics?.tts !== 'mos') throw new Error('TTS ranking metric must be MOS');
if (asr.length !== 10 || tts.length !== 10) throw new Error('Fixed file must contain 10 ASR and 10 TTS models');

for (const model of asr) {
  for (const metric of ['wer', 'cer', 'rtf', 'latency_ms']) {
    if (!Number.isFinite(model.metrics?.[metric])) throw new Error(`${model.name} missing ${metric}`);
  }
  if (!Array.isArray(model.datasets) || model.datasets.length < 3) throw new Error(`${model.name} needs dataset metrics`);
}

for (const model of tts) {
  for (const metric of ['mos', 'sim', 'wer', 'rtf', 'latency_ms']) {
    if (!Number.isFinite(model.metrics?.[metric])) throw new Error(`${model.name} missing ${metric}`);
  }
  if (!Array.isArray(model.datasets) || model.datasets.length < 3) throw new Error(`${model.name} needs dataset metrics`);
}

const uniqueNames = new Set(models.map(model => model.name));
if (uniqueNames.size !== models.length) throw new Error('Model names must be unique');
if (!html.includes('id="themeToggle"') || !html.includes('id="compareRadar"')) throw new Error('Theme or radar UI missing');
if (!app.includes("'立即对比 →'") || !app.includes('rankMotion')) throw new Error('Floating compare or rank motion missing');
if (!app.includes('magnitudePercent(metricValue(m,metric),values)')) throw new Error('Top 10 bars must reflect raw metric magnitude');
if (html.includes('固定数据源') || html.includes('<th>趋势</th>')) throw new Error('Removed source/trend copy reappeared');

console.log(`fixed data smoke test: ok (${asr.length} ASR, ${tts.length} TTS)`);
