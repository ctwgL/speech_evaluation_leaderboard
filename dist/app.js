let DATA = {
  asr: [
    {name:'Paraformer-Large',org:'FunASR',kind:'open',logo:'PF',score:4.12,rtf:0.06,cost:'免费',trend:[18,16,17,12,11,8],radar:[94,92,86,88,90]},
    {name:'Seed-ASR',org:'ByteDance',kind:'api',logo:'SE',score:4.46,rtf:0.09,cost:'¥1.20',trend:[20,18,15,16,12,9],radar:[93,95,90,84,91]},
    {name:'Whisper Large-v3',org:'OpenAI',kind:'open',logo:'WV',score:5.01,rtf:0.18,cost:'免费',trend:[19,18,17,14,13,10],radar:[91,90,94,78,88]},
    {name:'SenseVoice-Large',org:'Alibaba',kind:'open',logo:'SV',score:5.37,rtf:0.07,cost:'免费',trend:[21,19,18,16,12,11],radar:[89,87,91,92,86]},
    {name:'Google Chirp 3',org:'Google Cloud',kind:'api',logo:'GC',score:5.62,rtf:0.11,cost:'¥1.44',trend:[17,18,15,15,13,12],radar:[88,92,93,80,89]},
    {name:'Qwen-Audio ASR',org:'Qwen',kind:'open',logo:'QA',score:6.08,rtf:0.13,cost:'免费',trend:[22,20,17,18,15,13],radar:[85,86,88,90,84]},
    {name:'Azure Speech',org:'Microsoft',kind:'api',logo:'AS',score:6.31,rtf:0.10,cost:'¥1.05',trend:[23,21,19,17,15,14],radar:[84,88,91,86,83]},
    {name:'NVIDIA Canary',org:'NVIDIA',kind:'open',logo:'NC',score:6.55,rtf:0.15,cost:'免费',trend:[24,21,20,18,16,15],radar:[83,86,89,82,85]},
    {name:'Deepgram Nova-3',org:'Deepgram',kind:'api',logo:'DN',score:6.78,rtf:0.08,cost:'¥1.32',trend:[22,20,19,18,17,16],radar:[82,87,88,91,82]},
    {name:'Moonshine Base',org:'Useful Sensors',kind:'open',logo:'MB',score:7.14,rtf:0.05,cost:'免费',trend:[25,23,21,20,18,17],radar:[79,82,81,96,87]}
  ],
  tts: [
    {name:'CosyVoice 2',org:'FunAudioLLM',kind:'open',logo:'CV',score:4.63,rtf:0.18,cost:'免费',trend:[10,12,14,16,18,21],radar:[93,91,88,86,92]},
    {name:'Eleven v3',org:'ElevenLabs',kind:'api',logo:'EL',score:4.58,rtf:0.21,cost:'¥2.10',trend:[11,13,15,17,19,22],radar:[96,92,90,82,88]},
    {name:'Seed-TTS',org:'ByteDance',kind:'api',logo:'ST',score:4.51,rtf:0.16,cost:'¥1.68',trend:[9,11,15,16,20,21],radar:[94,93,89,88,90]},
    {name:'Fish Speech 1.5',org:'Fish Audio',kind:'open',logo:'FS',score:4.42,rtf:0.22,cost:'免费',trend:[10,12,13,17,18,20],radar:[90,89,92,91,85]},
    {name:'GPT-SoVITS',org:'RVC-Boss',kind:'open',logo:'GS',score:4.28,rtf:0.26,cost:'免费',trend:[8,10,12,14,17,19],radar:[86,88,90,93,82]},
    {name:'Azure Neural TTS',org:'Microsoft',kind:'api',logo:'AZ',score:4.19,rtf:0.14,cost:'¥1.05',trend:[12,13,15,16,17,18],radar:[89,90,86,84,87]},
    {name:'MiniMax Speech-02',org:'MiniMax',kind:'api',logo:'MM',score:4.12,rtf:0.17,cost:'¥1.26',trend:[10,12,13,15,16,17],radar:[88,86,84,87,85]},
    {name:'F5-TTS',org:'SWivid',kind:'open',logo:'F5',score:4.06,rtf:0.24,cost:'免费',trend:[8,9,11,13,15,16],radar:[84,85,87,83,86]},
    {name:'XTTS-v2',org:'Coqui',kind:'open',logo:'XT',score:3.98,rtf:0.28,cost:'免费',trend:[7,9,10,12,14,15],radar:[82,84,86,80,84]},
    {name:'Google Cloud TTS',org:'Google Cloud',kind:'api',logo:'GT',score:3.91,rtf:0.13,cost:'¥1.18',trend:[9,10,11,12,13,14],radar:[81,83,85,89,82]}
  ]
};

let rankingMetric={asr:'wer',tts:'mos'};
const LOWER_IS_BETTER=new Set(['wer','cer','rtf','latency','latency_ms','cost','memory','params']);
const METRIC_UNITS={wer:'%',cer:'%',mos:'/ 5',sim:'',rtf:'×',accuracy:'%',pesq:'',stoi:'',sisdr:' dB',sdr:' dB'};
const metricLabel=key=>String(key||'score').toUpperCase();
const metricArrow=key=>LOWER_IS_BETTER.has(String(key).toLowerCase())?'↓':'↑';
const metricUnit=key=>METRIC_UNITS[String(key).toLowerCase()]??'';
const metricValue=(model,key)=>Number(model.metrics?.[key]??model.score);
const sortModels=(models,key)=>[...models].sort((a,b)=>LOWER_IS_BETTER.has(key)?metricValue(a,key)-metricValue(b,key):metricValue(b,key)-metricValue(a,key));
const formatMetric=(value,key)=>`${Number(value).toFixed(key==='sim'?3:2)}${metricUnit(key)}`;
const esc=value=>String(value??'').replace(/[&<>"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));

let task = 'asr';
const selected = new Set();
const body = document.querySelector('#leaderboardBody');
const typeFilter = document.querySelector('#typeFilter');
const datasetFilter = document.querySelector('#datasetFilter');

function sparkline(values, kind) {
  const max=Math.max(...values), min=Math.min(...values);
  const pts=values.map((v,i)=>`${i*12},${19-((v-min)/(max-min||1))*15}`).join(' ');
  return `<svg class="spark" viewBox="0 0 64 22"><polyline points="${pts}" fill="none" stroke="${kind==='api'?'#9b8cff':'#61f2d0'}" stroke-width="1.3"/><circle cx="60" cy="${pts.split(' ').at(-1).split(',')[1]}" r="2" fill="${kind==='api'?'#9b8cff':'#61f2d0'}"/></svg>`;
}

function renderTable() {
  const aggregateMetric=rankingMetric[task],dataset=datasetFilter.value;
  const entries=DATA[task].map(model=>{
    if(dataset==='综合评测')return {model,key:aggregateMetric,value:metricValue(model,aggregateMetric)};
    const result=(model.datasets||[]).find(d=>d.name===dataset);if(!result)return null;
    const key=result.metrics?.[aggregateMetric]!=null?aggregateMetric:(task==='asr'?(result.metrics?.wer!=null?'wer':'cer'):(result.metrics?.mos!=null?'mos':'sim'));
    return {model,key,value:Number(result.metrics?.[key])};
  }).filter(Boolean).filter(x=>typeFilter.value==='all'||x.model.kind===typeFilter.value);
  const key=entries[0]?.key||aggregateMetric;entries.sort((a,b)=>LOWER_IS_BETTER.has(key)?a.value-b.value:b.value-a.value);
  document.querySelector('#primaryMetric').textContent=`${metricLabel(key)} ${metricArrow(key)}`;
  if(!entries.length){body.innerHTML='<tr><td colspan="7" class="empty-state">当前筛选条件下无模型数据</td></tr>';return}
  body.innerHTML=entries.map(({model:m,value,key},i)=>`<tr data-model="${esc(m.name)}" tabindex="0" aria-label="查看 ${esc(m.name)} 的各测试集指标">
    <td><span class="rank ${i<3?'top':''}">${String(i+1).padStart(2,'0')}</span></td>
    <td><div class="model"><span class="model-logo">${esc(m.logo)}</span><span><b>${esc(m.name)}</b><small><i class="type-dot ${m.kind}"></i>${esc(m.org)}</small></span></div></td>
    <td><span class="score ${i===0?'best':''}">${formatMetric(value,key)}</span></td>
    <td>${Number(m.rtf||0).toFixed(2)}×</td><td class="cost ${m.cost==='免费'?'free':''}">${esc(m.cost)}</td><td>${sparkline(m.trend,m.kind)}</td>
    <td><input class="compare-check" type="checkbox" aria-label="将 ${esc(m.name)} 加入对比" data-name="${esc(m.name)}" ${selected.has(m.name)?'checked':''}></td>
  </tr>`).join('');
  body.querySelectorAll('.compare-check').forEach(el=>el.addEventListener('change',onCompare));
  body.querySelectorAll('tr[data-model]').forEach(row=>{
    row.addEventListener('click',e=>{if(!e.target.classList.contains('compare-check'))openModel(row.dataset.model)});
    row.addEventListener('keydown',e=>{if(e.key==='Enter')openModel(row.dataset.model)});
  });
}

function setTask(next) {
  task=next;
  document.querySelector('#taskTitle').textContent=task==='asr'?'语音识别':'语音合成';
  const metric=rankingMetric[task];
  document.querySelectorAll('[data-task]').forEach(b=>{
    b.classList.toggle(b.classList.contains('task-switch')?'active-task':'active',b.dataset.task===task);
  });
  updateDatasetFilter();renderTable();renderTop10();
}

function updateDatasetFilter(){
  const current=datasetFilter.value,datasets=[...new Set(DATA[task].flatMap(m=>(m.datasets||[]).map(d=>d.name)))];
  datasetFilter.innerHTML='<option>综合评测</option>'+datasets.map(name=>`<option>${esc(name)}</option>`).join('');
  if(datasets.includes(current))datasetFilter.value=current;
}

function onCompare(e){
  const name=e.target.dataset.name;
  if(e.target.checked){if(selected.size>=3){e.target.checked=false;showToast('最多同时对比 3 个模型');return}selected.add(name);showToast(`${name} 已加入对比`)}else selected.delete(name);
  document.querySelector('#compareCount').textContent=selected.size;
}

function renderTop10(){
  const metric=rankingMetric[task];
  const models=sortModels(DATA[task],metric).slice(0,10);
  const values=models.map(m=>LOWER_IS_BETTER.has(metric)?1/Math.max(metricValue(m,metric),.0001):metricValue(m,metric));
  const max=Math.max(...values),min=Math.min(...values);
  document.querySelector('#top10Task').textContent=task.toUpperCase();
  document.querySelector('#rankMetric').textContent=`${metricLabel(metric)} ${metricArrow(metric)}`;
  document.querySelector('#top10Chart').innerHTML=models.map((m,i)=>{
    const width=42+(values[i]-min)/(max-min||1)*58;
    const value=formatMetric(metricValue(m,metric),metric);
    return `<button class="top10-row" data-name="${esc(m.name)}" aria-label="查看 ${esc(m.name)} 指标"><span class="top10-rank">${String(i+1).padStart(2,'0')}</span><span class="top10-main"><span class="top10-label"><b>${esc(m.name)}</b><em>${value}</em></span><span class="top10-track"><i style="width:${width}%"></i></span></span></button>`;
  }).join('');
  document.querySelectorAll('.top10-row').forEach(row=>row.addEventListener('click',()=>openModel(row.dataset.name)));
}

function showToast(message){const t=document.querySelector('#toast');t.textContent=message;t.classList.add('show');clearTimeout(showToast.timer);showToast.timer=setTimeout(()=>t.classList.remove('show'),2200)}

const ASR_DATASETS=[
  ['AISHELL-1','中文 · 普通话','178h','cer',0],
  ['AISHELL-2 iOS','中文 · 远场','1000h','cer',1.14],
  ['WenetSpeech Test_Net','中文 · 多场景','23h','cer',2.38],
  ['LibriSpeech test-clean','英文 · 朗读','5.4h','wer',-1.55],
  ['LibriSpeech test-other','英文 · 复杂','5.1h','wer',1.92],
  ['CHiME-4 Real','英文 · 噪声','2.3h','wer',4.76]
];
const TTS_DATASETS=[
  ['SeedTTS-Eval ZH','中文 · 零样本','2,000 条','zh',0],
  ['SeedTTS-Eval EN','英文 · 零样本','2,000 条','en',-.12],
  ['LibriTTS test-clean','英文 · 朗读','4.8h','en',-.18],
  ['AISHELL-3 Test','中文 · 多说话人','5.1h','zh',-.24],
  ['Emilia-ZH Test','中文 · 对话','6.2h','zh',-.31],
  ['Common Voice EN','英文 · 多口音','7.6h','en',-.36]
];

function findModel(name){for(const group of Object.keys(DATA)){const model=DATA[group].find(m=>m.name===name);if(model)return {model,group}}}
function openModel(name){
  const found=findModel(name);if(!found)return;
  const {model,group}=found;
  document.querySelector('#drawerLogo').textContent=model.logo;
  document.querySelector('#drawerTask').textContent=`${group.toUpperCase()} / MODEL METRICS`;
  document.querySelector('#drawerName').textContent=model.name;
  document.querySelector('#drawerOrg').textContent=`${model.org} · ${model.kind==='open'?'开源模型':'商业 API'}`;
  renderModelDetails(model,group);
  const drawer=document.querySelector('#modelDrawer'),backdrop=document.querySelector('#drawerBackdrop');
  backdrop.hidden=false;requestAnimationFrame(()=>{drawer.classList.add('open');backdrop.classList.add('open')});
  drawer.setAttribute('aria-hidden','false');document.body.classList.add('body-locked');
}
function closeModel(){
  const drawer=document.querySelector('#modelDrawer'),backdrop=document.querySelector('#drawerBackdrop');
  drawer.classList.remove('open');backdrop.classList.remove('open');drawer.setAttribute('aria-hidden','true');document.body.classList.remove('body-locked');
  setTimeout(()=>backdrop.hidden=true,260);
}
function renderModelDetails(model,group){
  const primary=rankingMetric[group],datasetTotal=model.datasets?.length||6;
  const kpis=[[`综合 ${metricLabel(primary)}`,metricValue(model,primary).toFixed(primary==='sim'?3:2),metricUnit(primary)],['实时率',Number(model.rtf||0).toFixed(2),'×'],['每小时成本',model.cost,''],['测试集',datasetTotal,'SETS']];
  document.querySelector('#detailKpis').innerHTML=kpis.map(x=>`<article class="detail-kpi"><span>${x[0]}</span><strong>${x[1]} <small>${x[2]}</small></strong></article>`).join('');
  if(group==='asr'){
    document.querySelector('#detailHead').innerHTML=`<tr><th>测试集</th><th>语言 / 场景</th><th>${metricLabel(primary)} ${metricArrow(primary)}</th><th>RTF ↓</th><th>音频规模</th></tr>`;
    if(model.datasets?.length){
      document.querySelector('#detailBody').innerHTML=model.datasets.map(d=>{const key=d.metrics?.[primary]!=null?primary:(d.metrics?.wer!=null?'wer':'cer');return `<tr><td class="dataset-name">${esc(d.name)}<small>${metricLabel(key)}</small></td><td>${esc(d.scenario||d.language||'—')}</td><td class="metric-good">${formatMetric(d.metrics?.[key]??0,key)}</td><td>${formatMetric(d.metrics?.rtf??model.rtf,'rtf')}</td><td>${esc(d.size||d.samples||'—')}</td></tr>`}).join('');
    }else{
      document.querySelector('#detailBody').innerHTML=ASR_DATASETS.map((d,i)=>{const error=Math.max(1.4,model.score+d[4]+((model.logo.charCodeAt(0)+i)%5)*.11);const rtf=model.rtf*(1+i*.08);return `<tr><td class="dataset-name">${d[0]}<small>${d[3].toUpperCase()}</small></td><td>${d[1]}</td><td class="metric-good">${error.toFixed(2)}%</td><td>${rtf.toFixed(2)}×</td><td>${d[2]}</td></tr>`}).join('');
    }
  }else{
    document.querySelector('#detailHead').innerHTML='<tr><th>测试集</th><th>语言 / 场景</th><th>MOS ↑</th><th>WER ↓</th><th>SIM ↑</th></tr>';
    if(model.datasets?.length){
      document.querySelector('#detailBody').innerHTML=model.datasets.map(d=>`<tr><td class="dataset-name">${esc(d.name)}<small>${esc(d.size||d.samples||'—')}</small></td><td>${esc(d.scenario||d.language||'—')}</td><td class="metric-good">${formatMetric(d.metrics?.mos??0,'mos')}</td><td>${formatMetric(d.metrics?.wer??0,'wer')}</td><td>${formatMetric(d.metrics?.sim??0,'sim')}</td></tr>`).join('');
    }else{
      document.querySelector('#detailBody').innerHTML=TTS_DATASETS.map((d,i)=>{const mos=Math.max(3.4,model.score+d[4]-((model.logo.charCodeAt(0)+i)%4)*.025);const wer=2.18+i*.47+(4.65-model.score)*1.3;const sim=.82+(model.score-4)*.10-i*.009;return `<tr><td class="dataset-name">${d[0]}<small>${d[2]}</small></td><td>${d[1]}</td><td class="metric-good">${mos.toFixed(2)}</td><td>${wer.toFixed(2)}%</td><td>${sim.toFixed(3)}</td></tr>`}).join('');
    }
  }
}

function normalizeMetrics(metrics={}){
  return Object.fromEntries(Object.entries(metrics).map(([key,value])=>[key.toLowerCase(),Number(value)]).filter(([,value])=>Number.isFinite(value)));
}
function normalizeModel(raw,group){
  const metrics=normalizeMetrics(raw.metrics);
  const primary=rankingMetric[group];
  if(metrics[primary]==null&&Number.isFinite(Number(raw.score)))metrics[primary]=Number(raw.score);
  if(metrics.rtf==null&&Number.isFinite(Number(raw.rtf)))metrics.rtf=Number(raw.rtf);
  const score=Number(metrics[primary]);
  if(!raw.name||!Number.isFinite(score))return null;
  const datasets=(raw.datasets||[]).map(d=>({...d,name:d.name||d.dataset||'Unnamed',metrics:normalizeMetrics(d.metrics)}));
  const initials=String(raw.name).split(/[\s-_]+/).map(x=>x[0]).join('').slice(0,2).toUpperCase();
  const trend=Array.isArray(raw.trend)&&raw.trend.length>1?raw.trend.map(Number):[score*1.08,score*1.05,score*1.03,score*1.02,score*1.01,score];
  return {name:String(raw.name),org:String(raw.org||raw.provider||'—'),kind:String(raw.kind||raw.type||'open').toLowerCase()==='api'?'api':'open',logo:String(raw.logo||initials),score,metrics,rtf:Number(metrics.rtf||0),cost:raw.cost??'—',trend,datasets};
}
function normalizePayload(payload){
  rankingMetric={asr:String(payload.rankingMetrics?.asr||payload.ranking_metrics?.asr||'wer').toLowerCase(),tts:String(payload.rankingMetrics?.tts||payload.ranking_metrics?.tts||'mos').toLowerCase()};
  const groups={asr:[],tts:[]};
  if(Array.isArray(payload.models)){
    payload.models.forEach(raw=>{const group=String(raw.task||raw.group||'').toLowerCase();if(groups[group]){const model=normalizeModel(raw,group);if(model)groups[group].push(model)}});
  }else{
    ['asr','tts'].forEach(group=>(payload[group]||[]).forEach(raw=>{const model=normalizeModel(raw,group);if(model)groups[group].push(model)}));
  }
  if(!groups.asr.length&&!groups.tts.length)throw new Error('文件中没有可用的 ASR/TTS 模型记录');
  groups.asr=sortModels(groups.asr,rankingMetric.asr);groups.tts=sortModels(groups.tts,rankingMetric.tts);
  return groups;
}
function parseCSV(text){
  const parseLine=line=>{const cells=[];let value='',quoted=false;for(let i=0;i<line.length;i++){const ch=line[i];if(ch==='"'&&line[i+1]==='"'&&quoted){value+='"';i++}else if(ch==='"'){quoted=!quoted}else if(ch===','&&!quoted){cells.push(value.trim());value=''}else value+=ch}cells.push(value.trim());return cells};
  const lines=text.replace(/^\uFEFF/,'').split(/\r?\n/).filter(line=>line.trim());
  if(lines.length<2)throw new Error('CSV 至少需要表头和一行数据');
  const headers=parseLine(lines[0]).map(x=>x.toLowerCase());
  const rows=lines.slice(1).map(line=>Object.fromEntries(headers.map((h,i)=>[h,parseLine(line)[i]??''])));
  const groups=new Map(),metricKeys=['wer','cer','mos','sim','rtf','accuracy','pesq','stoi','sisdr','sdr','latency','latency_ms'];
  rows.forEach(row=>{
    const taskName=(row.task||'').toLowerCase(),name=row.name||row.model;if(!['asr','tts'].includes(taskName)||!name)return;
    if(row.ranking_metric)rankingMetric[taskName]=row.ranking_metric.toLowerCase();
    const key=`${taskName}::${name}`;
    if(!groups.has(key))groups.set(key,{task:taskName,name,org:row.org||row.provider,kind:row.kind||row.type,logo:row.logo,cost:row.cost,metrics:{},datasets:[]});
    const model=groups.get(key),metrics={};metricKeys.forEach(k=>{if(row[k]!==''&&Number.isFinite(Number(row[k])))metrics[k]=Number(row[k])});
    if(row.dataset){model.datasets.push({name:row.dataset,scenario:row.scenario||row.language,size:row.size||row.samples,metrics})}else Object.assign(model.metrics,metrics);
  });
  const models=[...groups.values()];
  models.forEach(model=>{const primary=rankingMetric[model.task];if(model.metrics[primary]==null){const vals=model.datasets.map(d=>d.metrics[primary]).filter(Number.isFinite);if(vals.length)model.metrics[primary]=vals.reduce((a,b)=>a+b,0)/vals.length}if(model.metrics.rtf==null){const vals=model.datasets.map(d=>d.metrics.rtf).filter(Number.isFinite);if(vals.length)model.metrics.rtf=vals.reduce((a,b)=>a+b,0)/vals.length}});
  return {rankingMetrics:{...rankingMetric},models};
}
function renderDatasetList(){
  const counts=new Map();[...DATA.asr,...DATA.tts].forEach(model=>(model.datasets||[]).forEach(d=>{const old=counts.get(d.name)||{name:d.name,scenario:d.scenario||d.language||'测试集',size:d.size||d.samples||'—',count:0};old.count++;counts.set(d.name,old)}));
  const datasets=counts.size?[...counts.values()].sort((a,b)=>b.count-a.count).slice(0,3):[
    {name:'AISHELL-1',scenario:'中文普通话',size:'178h',count:DATA.asr.length},{name:'LibriSpeech',scenario:'英文朗读',size:'960h',count:DATA.asr.length},{name:'SeedTTS-Eval',scenario:'零样本合成',size:'4,000 条',count:DATA.tts.length}
  ];
  document.querySelector('#datasetList').innerHTML=datasets.map((d,i)=>`<button><span class="dataset-icon ${i===1?'violet':i===2?'orange':''}">${i===0?'01':i===1?'02':'03'}</span><span><b>${esc(d.name)}</b><small>${esc(d.size)} · ${esc(d.scenario)}</small></span><em>${d.count}</em></button>`).join('');
}
function updateSummary(){
  const asr=sortModels(DATA.asr,rankingMetric.asr)[0],tts=sortModels(DATA.tts,rankingMetric.tts)[0],all=[...DATA.asr,...DATA.tts];
  document.querySelector('#asrBestDirection').textContent=`${metricLabel(rankingMetric.asr)} ${metricArrow(rankingMetric.asr)}`;
  document.querySelector('#asrBestValue').textContent=asr?formatMetric(metricValue(asr,rankingMetric.asr),rankingMetric.asr):'—';
  document.querySelector('#asrBestUnit').textContent=metricLabel(rankingMetric.asr);document.querySelector('#asrBestModel').textContent=asr?.name||'无数据';
  document.querySelector('#ttsBestDirection').textContent=`${metricLabel(rankingMetric.tts)} ${metricArrow(rankingMetric.tts)}`;
  document.querySelector('#ttsBestValue').textContent=tts?formatMetric(metricValue(tts,rankingMetric.tts),rankingMetric.tts):'—';
  document.querySelector('#ttsBestUnit').textContent=metricLabel(rankingMetric.tts);document.querySelector('#ttsBestModel').textContent=tts?.name||'无数据';
  const fastest=[...all].filter(m=>Number(m.rtf)>0).sort((a,b)=>a.rtf-b.rtf)[0];document.querySelector('#rtfBestValue').textContent=fastest?formatMetric(fastest.rtf,'rtf'):'—';document.querySelector('#rtfBestModel').textContent=fastest?.name||'无数据';
  const asrSets=new Set(DATA.asr.flatMap(m=>(m.datasets||[]).map(d=>d.name))),ttsSets=new Set(DATA.tts.flatMap(m=>(m.datasets||[]).map(d=>d.name)));
  const asrCount=DATA.asr.length?(asrSets.size||6):0,ttsCount=DATA.tts.length?(ttsSets.size||6):0;document.querySelector('#datasetCount').textContent=asrCount+ttsCount;document.querySelector('#datasetSplit').innerHTML=`ASR ${asrCount} <span>·</span> TTS ${ttsCount}`;
}
function refreshAll(){
  if(!DATA[task].length)task=DATA.asr.length?'asr':'tts';
  setTask(task);updateSummary();renderDatasetList();
}
async function loadResultFile(file){
  const loader=document.querySelector('.file-loader'),badge=document.querySelector('#dataSource');loader.classList.add('loading');badge.textContent='解析中…';badge.className='source-badge';
  try{
    const text=await file.text();const payload=file.name.toLowerCase().endsWith('.csv')?parseCSV(text):JSON.parse(text);DATA=normalizePayload(payload);selected.clear();document.querySelector('#compareCount').textContent='0';refreshAll();badge.textContent=file.name;badge.title=file.name;badge.classList.add('loaded');document.querySelector('#sourceSummary').innerHTML=`数据源：${esc(file.name)} · <i></i> 已解析 ${DATA.asr.length+DATA.tts.length} 个模型`;showToast(`已加载 ${DATA.asr.length+DATA.tts.length} 个模型`);
  }catch(error){badge.textContent='文件解析失败';badge.classList.add('error');showToast(error.message||'文件解析失败')}
  finally{loader.classList.remove('loading');document.querySelector('#resultFile').value=''}
}

document.querySelectorAll('[data-task]').forEach(b=>b.addEventListener('click',()=>setTask(b.dataset.task)));
typeFilter.addEventListener('change',renderTable);
datasetFilter.addEventListener('change',renderTable);
document.querySelector('#resultFile').addEventListener('change',e=>{if(e.target.files?.[0])loadResultFile(e.target.files[0])});
document.querySelector('#showAll').addEventListener('click',()=>showToast('完整榜单将在数据接入后开放'));
document.querySelector('#mobileMenu').addEventListener('click',()=>document.querySelector('#sidebar').classList.toggle('open'));

const overlay=document.querySelector('#commandOverlay'),input=document.querySelector('#commandInput'),results=document.querySelector('#commandResults');
function openSearch(){overlay.hidden=false;input.value='';fillResults('');setTimeout(()=>input.focus(),0)}
function closeSearch(){overlay.hidden=true}
function fillResults(q){const items=[...DATA.asr,...DATA.tts].filter(m=>m.name.toLowerCase().includes(q.toLowerCase())).slice(0,6);results.innerHTML=items.length?items.map(m=>`<button class="command-result" data-name="${m.name}"><span>${m.name}</span><small>${m.org} · ${m.kind==='open'?'开源':'API'}</small></button>`).join(''):'<button class="command-result"><span>未找到匹配结果</span></button>';results.querySelectorAll('[data-name]').forEach(b=>b.addEventListener('click',()=>{closeSearch();openModel(b.dataset.name)}))}
document.querySelector('#searchTrigger').addEventListener('click',openSearch);input.addEventListener('input',()=>fillResults(input.value));overlay.addEventListener('click',e=>{if(e.target===overlay)closeSearch()});document.querySelector('#drawerClose').addEventListener('click',closeModel);document.querySelector('#drawerBackdrop').addEventListener('click',closeModel);document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openSearch()}if(e.key==='Escape'){closeSearch();closeModel()}});

refreshAll();
