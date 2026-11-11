const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync('dist/app.js', 'utf8');
const elements = new Map();
const context = vm.createContext({
  URL,
  location: { href: 'http://localhost:8000/' },
  document: {
    querySelector(selector) {
      if (selector === '.architecture img') return null;
      if (!elements.has(selector)) elements.set(selector, {});
      return elements.get(selector);
    },
  },
});
const preamble = source.slice(0, source.indexOf("let task='asr'"));
const normalizers = source.slice(source.indexOf('function normalizeMetrics'), source.indexOf('function renderDatasetList'));
const profile = source.slice(source.indexOf('function assetUrl'), source.indexOf('function comparisonWidth'));
vm.runInContext(preamble + normalizers + profile, context);
const { buildIndex, dataDir } = require('../scripts/build-data-index.cjs');
const index = buildIndex();
const payload = { ...index, models: index.modelFiles.map(file => JSON.parse(fs.readFileSync(path.join(dataDir, file), 'utf8'))) };
context.payload = payload;
vm.runInContext('DATA=normalizePayload(payload);renderModelDetails(DATA.asr[0],"asr")', context);
assert.match(elements.get('#modelIntroduction').innerHTML, /模型文件大小/);
assert.match(elements.get('#modelIntroduction').innerHTML, /待补充/);
assert.match(elements.get('#detailHead').innerHTML, /CER/);
assert.match(elements.get('#detailHead').innerHTML, /WER/);
context.custom = { task: 'tts', name: 'New model', metrics: { mos: 4, pesq: 3 }, datasets: [{name:'New dataset',metrics:{pesq:3}}], introduction: { summary: '<script>bad</script>', specifications: {'自定义规格':'42M'}, architecture: {image:'javascript:alert(1)'}, links:[{label:'bad',url:'javascript:alert(1)'}] } };
vm.runInContext('renderModelDetails(normalizeModel(custom,"tts"),"tts")', context);
assert.match(elements.get('#detailHead').innerHTML, /PESQ/);
assert.match(elements.get('#detailKpis').innerHTML, /PESQ/);
assert.match(elements.get('#modelIntroduction').innerHTML, /42M/);
assert.match(elements.get('#modelIntroduction').innerHTML, /&lt;script&gt;/);
assert.doesNotMatch(elements.get('#modelIntroduction').innerHTML, /javascript:/);
console.log('model profile: ok (normalization, custom metrics, specifications, escaping, image/link URLs)');
