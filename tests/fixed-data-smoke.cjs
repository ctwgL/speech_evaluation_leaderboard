const fs = require('fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const { buildIndex, dataDir } = require('../scripts/build-data-index.cjs');

const index = buildIndex();
assert.deepEqual(JSON.parse(fs.readFileSync(path.join(dataDir, 'index.json'), 'utf8')), index, '索引过期，请重新生成');
const payload = { ...index, models: index.modelFiles.map(file => JSON.parse(fs.readFileSync(path.join(dataDir, file), 'utf8'))) };
const html = fs.readFileSync('dist/index.html', 'utf8');
const app = fs.readFileSync('dist/app.js', 'utf8');
const styles = fs.readFileSync('dist/styles.css', 'utf8');
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
if (!html.includes('id="datasetMore"') || !html.includes('id="datasetMoreCount"')) throw new Error('Dataset explorer controls missing');
if (html.includes('<th>实时率') || html.includes('<th>成本')) throw new Error('Leaderboard must not display RTF or cost columns');
if (!app.includes("'立即对比 →'") || !app.includes('rankMotion')) throw new Error('Floating compare or rank motion missing');
if (!app.includes("openSearch('datasets')") || !app.includes("showDatasetRanking(button.dataset.group")) throw new Error('Dataset explorer interactions missing');
if (!styles.includes('--open-color:') || !styles.includes('--api-color:') || !styles.includes('.type-dot.api')) throw new Error('Distinct model type colors missing');
if (!app.includes('magnitudePercent(metricValue(m,metric),values)')) throw new Error('Top 10 bars must reflect raw metric magnitude');
if (!app.includes('const level=magnitudePercent(value,scoreValues)')) throw new Error('Leaderboard bars must reflect raw metric magnitude');
if (!app.includes('function comparisonWidth(value,values){return magnitudePercent(value,values)}')) throw new Error('Comparison bars must reflect raw metric magnitude');
if (html.includes('固定数据源') || html.includes('<th>趋势</th>')) throw new Error('Removed source/trend copy reappeared');

console.log(`fixed data smoke test: ok (${asr.length} ASR, ${tts.length} TTS)`);

// Exercise the shipped comparison renderer against the published payload.
const vm = require('node:vm');
const nodes = new Map();
const context = vm.createContext({
  document: {
    documentElement: { dataset: {} },
    querySelectorAll: () => [],
    querySelector(id) {
      if (!nodes.has(id)) nodes.set(id, { querySelectorAll: () => [], classList: { add() {}, remove() {} } });
      return nodes.get(id);
    },
  },
  localStorage: { getItem() {}, setItem() {} },
  window: { addEventListener() {} },
  URL,
  location: { href: 'https://example.test/speech_evaluation_leaderboard/' },
  fetch: async () => ({ ok: false, status: 404 }),
  setTimeout() {},
  clearTimeout() {},
  payload,
});
vm.runInContext(app.slice(0, app.indexOf("document.querySelectorAll('.segmented [data-task]').forEach(b=>b.addEventListener")), context);
const run = (js) => vm.runInContext(js, context);
const compare = (names) => {
  context.names = names;
  run('selected.clear(); names.forEach(name => selected.add(name)); renderComparePage()');
  return nodes.get('#compareMetricGrid').innerHTML;
};
run('DATA=normalizePayload(payload)');
const wins = () => Array.from(nodes.get('#compareModels').innerHTML.matchAll(/<strong>(\d+)<\/strong>/g), match => Number(match[1]));
let pairs = 0;
for (let i = 0; i < asr.length; i++) {
  for (let j = i + 1; j < asr.length; j++) {
    const group = [asr[i], asr[j]];
    const markup = compare(group.map(model => model.name));
    for (const key of ['wer', 'cer', 'rtf', 'latency_ms']) {
      assert.equal(markup.includes(key.toUpperCase() + ' ↓'), group.every(model => Number.isFinite(model.metrics[key])), key + ' must be measured for both models');
    }
    pairs++;
  }
}
assert.deepEqual(JSON.parse(run('JSON.stringify(normalizeMetrics({CER:0, RTF:null, latency_ms:false, WER:"", MOS:"4.5"}))')), { cer: 0 });

context.fixture = {
  ...payload,
  models: [
    { task: 'asr', name: 'fixture-a', metrics: { cer: 4.28, wer: 2.74 } },
    { task: 'asr', name: 'fixture-b', metrics: { cer: 5.23, wer: 2.10 } },
    ...asr,
  ],
};
run('DATA=normalizePayload(fixture)');
assert.doesNotMatch(compare(['fixture-a', 'fixture-b']), /RTF|LATENCY_MS/);
assert.deepEqual(wins(), [1, 1]);
assert.doesNotMatch(nodes.get('#compareRadar').innerHTML, /<svg/);
context.fixture.models[0].metrics.rtf = 0.5;
assert.doesNotMatch(compareAfterReload(), /RTF/);
context.fixture.models[1].metrics.rtf = 0.75;
assert.match(compareAfterReload(), /0\.50×/);
assert.match(nodes.get('#compareRadar').innerHTML, /<svg/);
assert.doesNotMatch(nodes.get('#compareRadar').innerHTML, /NaN/);

function compareAfterReload() {
  run('DATA=normalizePayload(fixture)');
  return compare(['fixture-a', 'fixture-b']);
}
for (const id of ['asrBestValue', 'ttsBestValue', 'rtfBestValue']) {
  assert.ok(html.includes(`id="${id}">—<`), 'initial KPI must not present sample data');
}
run('loadFixedData()').then(() => {
  for (const id of ['#asrBestValue', '#ttsBestValue', '#rtfBestValue']) {
    assert.equal(nodes.get(id).textContent, '—');
  }
  assert.match(nodes.get('#leaderboardBody').innerHTML, /评测结果读取失败/);
  console.log(`metric comparison: ok (${pairs} published ASR pairs, measured RTF, missing metrics, load failure)`);
}).catch(error => { console.error(error); process.exitCode = 1; });
