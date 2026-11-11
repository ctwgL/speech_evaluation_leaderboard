const fs = require('fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const { buildIndex, dataDir } = require('../scripts/build-data-index.cjs');

const index = buildIndex();
assert.deepEqual(JSON.parse(fs.readFileSync(path.join(dataDir, 'index.json'), 'utf8')), index, '索引过期，请重新生成');
const payload = { ...index, models: index.modelFiles.map(file => JSON.parse(fs.readFileSync(path.join(dataDir, file), 'utf8'))) };
const html = fs.readFileSync('dist/index.html', 'utf8');
const app = fs.readFileSync('dist/app.js', 'utf8');
const models = payload.models || [];
if (index.modelFiles.some(file => path.basename(file) === 'template.json')) throw new Error('Template files must not be published as models');
const asr = models.filter(model => model.task === 'asr');
const tts = models.filter(model => model.task === 'tts');

if (!['wer', 'cer'].includes(payload.rankingMetrics?.asr)) throw new Error('ASR ranking metric must be WER or CER');
if (payload.rankingMetrics?.tts !== 'mos') throw new Error('TTS ranking metric must be MOS');
if (!models.length) throw new Error('No model files found');

for (const model of asr) {
  for (const metric of ['wer', 'cer']) {
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
for (const model of models.filter(model => model.logo_image)) {
  const logo = path.resolve(dataDir, '..', model.logo_image);
  if (!fs.existsSync(logo) || !fs.statSync(logo).isFile()) throw new Error(`${model.name} logo image missing`);
}
if (!html.includes('id="themeToggle"') || !html.includes('id="compareRadar"')) throw new Error('Theme or radar UI missing');
if (!app.includes("'立即对比 →'") || !app.includes('rankMotion')) throw new Error('Floating compare or rank motion missing');
if (!app.includes('magnitudePercent(metricValue(m,metric),values)')) throw new Error('Top 10 bars must reflect raw metric magnitude');
if (html.includes('固定数据源') || html.includes('<th>趋势</th>')) throw new Error('Removed source/trend copy reappeared');

console.log(`fixed data smoke test: ok (${asr.length} ASR, ${tts.length} TTS)`);
