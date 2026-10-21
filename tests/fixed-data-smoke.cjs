const fs = require('fs');

const file = 'dist/data/benchmark-results.json';
const payload = JSON.parse(fs.readFileSync(file, 'utf8'));
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

console.log(`fixed data smoke test: ok (${asr.length} ASR, ${tts.length} TTS)`);
