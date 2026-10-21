const fs = require('fs');
const vm = require('vm');

const source = fs.readFileSync('dist/app.js', 'utf8');
const definitions = source.split("\ndocument.querySelectorAll('[data-task]')")[0];
const context = {
  console,
  document: { querySelector: () => ({}) },
  Set,
  Map,
  Object,
  Array,
  Number,
  String,
  Math,
};

vm.createContext(context);
vm.runInContext(`${definitions}\nglobalThis.__parser = { normalizePayload, parseCSV, metricArrow };`, context);

const jsonPayload = JSON.parse(fs.readFileSync('examples/benchmark-results.example.json', 'utf8'));
const jsonData = context.__parser.normalizePayload(jsonPayload);
if (jsonData.asr.length !== 2 || jsonData.tts.length !== 2) throw new Error('JSON model count mismatch');
if (jsonData.asr[0].name !== 'Example-ASR-A') throw new Error('WER must sort ascending');
if (jsonData.tts[0].name !== 'Example-TTS-A') throw new Error('MOS must sort descending');

const csvPayload = context.__parser.parseCSV(fs.readFileSync('examples/benchmark-results.example.csv', 'utf8'));
const csvData = context.__parser.normalizePayload(csvPayload);
if (csvData.asr.length !== 1 || csvData.tts.length !== 1) throw new Error('CSV model count mismatch');
if (context.__parser.metricArrow('wer') !== '↓' || context.__parser.metricArrow('cer') !== '↓') throw new Error('WER/CER direction mismatch');
if (context.__parser.metricArrow('mos') !== '↑' || context.__parser.metricArrow('sim') !== '↑') throw new Error('MOS/SIM direction mismatch');

console.log('parser smoke test: ok');
