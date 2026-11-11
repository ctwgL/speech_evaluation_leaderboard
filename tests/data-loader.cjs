const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { buildIndex, dataDir } = require('../scripts/build-data-index.cjs');
const source = fs.readFileSync('dist/app.js', 'utf8');
const loader = source.slice(source.indexOf('async function readDataJson'), source.indexOf('async function loadFixedData'));
const requests = [];
let missing = '';
const context = vm.createContext({
  URL,
  location: { href: 'https://example.com/speech_evaluation_leaderboard/' },
  fetch: async url => {
    const pathname = new URL(url).pathname;
    requests.push(pathname);
    const relative = pathname.split('/data/')[1];
    if (relative === missing) return { ok: false, status: 404 };
    return { ok: true, json: async () => JSON.parse(fs.readFileSync(path.join(dataDir, relative), 'utf8')) };
  },
});
vm.runInContext(loader, context);
(async () => {
  const payload = await vm.runInContext('loadModelPayload()', context);
  assert.equal(payload.models.length, buildIndex().modelFiles.length);
  assert.equal(requests.length, payload.models.length + 1);
  assert.ok(requests.every(url => url.startsWith('/speech_evaluation_leaderboard/data/')));
  missing = buildIndex().modelFiles[0].slice(2);
  await assert.rejects(vm.runInContext('loadModelPayload()', context), error => error.message.includes(missing) && error.message.includes('404'));
  console.log('data loader: ok (all models, Pages subpath, missing file diagnostics)');
})().catch(error => { console.error(error); process.exitCode = 1; });
