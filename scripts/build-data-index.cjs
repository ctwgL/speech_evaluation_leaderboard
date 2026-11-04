const fs = require('node:fs');
const path = require('node:path');

const dataDir = path.resolve(__dirname, '../dist/data');
function fail(file, message) { throw new Error(`${file}: ${message}`); }
function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); }
  catch (error) { fail(file, error.message); }
}
function scan(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const file = path.join(dir, entry.name);
    return entry.isDirectory() ? scan(file) : entry.isFile() && entry.name.endsWith('.json') ? [file] : [];
  }).sort();
}
function buildIndex() {
  const config = readJson(path.join(dataDir, 'config.json'));
  if (config.schemaVersion !== 1) fail('config.json', 'schemaVersion 必须为 1');
  for (const task of ['asr', 'tts']) {
    if (typeof config.rankingMetrics?.[task] !== 'string' || !config.rankingMetrics[task]) fail('config.json', `缺少 rankingMetrics.${task}`);
  }
  const files = scan(path.join(dataDir, 'models'));
  if (!files.length) fail('models', '至少需要一个模型 JSON');
  const names = new Set();
  for (const file of files) {
    const model = readJson(file);
    if (!model || !['asr', 'tts'].includes(model.task)) fail(file, 'task 必须为 asr 或 tts');
    if (typeof model.name !== 'string' || !model.name.trim()) fail(file, '缺少模型名称');
    if (names.has(model.name)) fail(file, `模型名称重复：${model.name}`);
    names.add(model.name);
    if (!['open', 'api'].includes(model.kind)) fail(file, 'kind 必须为 open 或 api');
    const checkMetrics = (metrics, label) => {
      if (!metrics || typeof metrics !== 'object' || Array.isArray(metrics)) fail(file, `${label} 必须为对象`);
      for (const [key, value] of Object.entries(metrics)) {
        if (!key || key !== key.toLowerCase() || !Number.isFinite(value)) fail(file, `${label}.${key} 必须使用小写键和有限数值`);
      }
    };
    checkMetrics(model.metrics, 'metrics');
    if (!Number.isFinite(model.metrics[config.rankingMetrics[model.task]])) fail(file, '缺少综合排名指标');
    if (!Array.isArray(model.datasets)) fail(file, 'datasets 必须为数组');
    for (const dataset of model.datasets) {
      if (!dataset || typeof dataset.name !== 'string' || !dataset.name.trim()) fail(file, '测试集缺少名称');
      checkMetrics(dataset.metrics, `datasets.${dataset.name}.metrics`);
    }
  }
  return {
    schemaVersion: config.schemaVersion,
    rankingMetrics: config.rankingMetrics,
    modelFiles: files.map(file => './' + path.relative(dataDir, file).split(path.sep).join('/')),
  };
}
if (require.main === module) {
  const index = buildIndex();
  fs.writeFileSync(path.join(dataDir, 'index.json'), JSON.stringify(index, null, 2) + '\n');
  console.log(`模型索引已生成：${index.modelFiles.length} 个模型`);
}
module.exports = { buildIndex, dataDir };
