const DATA = {
  asr: [
    {name:'Paraformer-Large',org:'FunASR',kind:'open',logo:'PF',score:4.12,rtf:0.06,cost:'免费',trend:[18,16,17,12,11,8],radar:[94,92,86,88,90]},
    {name:'Seed-ASR',org:'ByteDance',kind:'api',logo:'SE',score:4.46,rtf:0.09,cost:'¥1.20',trend:[20,18,15,16,12,9],radar:[93,95,90,84,91]},
    {name:'Whisper Large-v3',org:'OpenAI',kind:'open',logo:'WV',score:5.01,rtf:0.18,cost:'免费',trend:[19,18,17,14,13,10],radar:[91,90,94,78,88]},
    {name:'SenseVoice-Large',org:'Alibaba',kind:'open',logo:'SV',score:5.37,rtf:0.07,cost:'免费',trend:[21,19,18,16,12,11],radar:[89,87,91,92,86]},
    {name:'Google Chirp 3',org:'Google Cloud',kind:'api',logo:'GC',score:5.62,rtf:0.11,cost:'¥1.44',trend:[17,18,15,15,13,12],radar:[88,92,93,80,89]},
    {name:'Qwen-Audio ASR',org:'Qwen',kind:'open',logo:'QA',score:6.08,rtf:0.13,cost:'免费',trend:[22,20,17,18,15,13],radar:[85,86,88,90,84]}
  ],
  tts: [
    {name:'CosyVoice 2',org:'FunAudioLLM',kind:'open',logo:'CV',score:4.63,rtf:0.18,cost:'免费',trend:[10,12,14,16,18,21],radar:[93,91,88,86,92]},
    {name:'Eleven v3',org:'ElevenLabs',kind:'api',logo:'EL',score:4.58,rtf:0.21,cost:'¥2.10',trend:[11,13,15,17,19,22],radar:[96,92,90,82,88]},
    {name:'Seed-TTS',org:'ByteDance',kind:'api',logo:'ST',score:4.51,rtf:0.16,cost:'¥1.68',trend:[9,11,15,16,20,21],radar:[94,93,89,88,90]},
    {name:'Fish Speech 1.5',org:'Fish Audio',kind:'open',logo:'FS',score:4.42,rtf:0.22,cost:'免费',trend:[10,12,13,17,18,20],radar:[90,89,92,91,85]},
    {name:'GPT-SoVITS',org:'RVC-Boss',kind:'open',logo:'GS',score:4.28,rtf:0.26,cost:'免费',trend:[8,10,12,14,17,19],radar:[86,88,90,93,82]},
    {name:'Azure Neural TTS',org:'Microsoft',kind:'api',logo:'AZ',score:4.19,rtf:0.14,cost:'¥1.05',trend:[12,13,15,16,17,18],radar:[89,90,86,84,87]}
  ]
};

let task = 'asr';
const selected = new Set();
const body = document.querySelector('#leaderboardBody');
const typeFilter = document.querySelector('#typeFilter');

function sparkline(values, kind) {
  const max=Math.max(...values), min=Math.min(...values);
  const pts=values.map((v,i)=>`${i*12},${19-((v-min)/(max-min||1))*15}`).join(' ');
  return `<svg class="spark" viewBox="0 0 64 22"><polyline points="${pts}" fill="none" stroke="${kind==='api'?'#9b8cff':'#61f2d0'}" stroke-width="1.3"/><circle cx="60" cy="${pts.split(' ').at(-1).split(',')[1]}" r="2" fill="${kind==='api'?'#9b8cff':'#61f2d0'}"/></svg>`;
}

function renderTable() {
  const filtered=DATA[task].filter(m=>typeFilter.value==='all'||m.kind===typeFilter.value);
  body.innerHTML=filtered.map((m,i)=>`<tr data-model="${m.name}" tabindex="0" aria-label="查看 ${m.name} 的各测试集指标">
    <td><span class="rank ${i<3?'top':''}">${String(i+1).padStart(2,'0')}</span></td>
    <td><div class="model"><span class="model-logo">${m.logo}</span><span><b>${m.name}</b><small><i class="type-dot ${m.kind}"></i>${m.org}</small></span></div></td>
    <td><span class="score ${i===0?'best':''}">${m.score.toFixed(2)}${task==='asr'?'%':' / 5'}</span></td>
    <td>${m.rtf.toFixed(2)}×</td><td class="cost ${m.cost==='免费'?'free':''}">${m.cost}</td><td>${sparkline(m.trend,m.kind)}</td>
    <td><input class="compare-check" type="checkbox" aria-label="将 ${m.name} 加入对比" data-name="${m.name}" ${selected.has(m.name)?'checked':''}></td>
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
  document.querySelector('#primaryMetric').textContent=task==='asr'?'WER ↓':'MOS ↑';
  document.querySelectorAll('[data-task]').forEach(b=>{
    b.classList.toggle(b.classList.contains('task-switch')?'active-task':'active',b.dataset.task===task);
  });
  renderTable(); drawRadar(DATA[task][0]);
}

function onCompare(e){
  const name=e.target.dataset.name;
  if(e.target.checked){if(selected.size>=3){e.target.checked=false;showToast('最多同时对比 3 个模型');return}selected.add(name);showToast(`${name} 已加入对比`)}else selected.delete(name);
  document.querySelector('#compareCount').textContent=selected.size;
}

function polygonPoints(values,cx=160,cy=122,r=82){return values.map((v,i)=>{const a=-Math.PI/2+i*Math.PI*2/5,rr=r*v/100;return `${cx+Math.cos(a)*rr},${cy+Math.sin(a)*rr}`}).join(' ')}
function drawRadar(model){
  const labels=['准确性','鲁棒性','多语种','效率','易用性'];
  const grids=[.25,.5,.75,1].map(s=>`<polygon class="radar-grid" points="${polygonPoints([s*100,s*100,s*100,s*100,s*100])}"/>`).join('');
  const axes=labels.map((_,i)=>{const p=polygonPoints([100,100,100,100,100]).split(' ')[i];return `<line class="radar-axis" x1="160" y1="122" x2="${p.split(',')[0]}" y2="${p.split(',')[1]}"/>`}).join('');
  const text=labels.map((l,i)=>{const a=-Math.PI/2+i*Math.PI*2/5;return `<text x="${160+Math.cos(a)*105}" y="${125+Math.sin(a)*101}" text-anchor="middle">${l}</text>`}).join('');
  const points=polygonPoints(model.radar);
  const dots=points.split(' ').map(p=>`<circle class="radar-point" cx="${p.split(',')[0]}" cy="${p.split(',')[1]}" r="2.4"/>`).join('');
  document.querySelector('#radar').innerHTML=grids+axes+`<polygon class="radar-shape" points="${points}"/>`+dots+text;
  document.querySelector('#radarModel').textContent=model.name;
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
  const {model,group}=found; drawRadar(model);
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
  const kpis=group==='asr'
    ?[['综合 WER',model.score.toFixed(2),'%'],['实时率',model.rtf.toFixed(2),'×'],['每小时成本',model.cost,''],['测试集','6','SETS']]
    :[['综合 MOS',model.score.toFixed(2),'/ 5'],['实时率',model.rtf.toFixed(2),'×'],['每小时成本',model.cost,''],['测试集','6','SETS']];
  document.querySelector('#detailKpis').innerHTML=kpis.map(x=>`<article class="detail-kpi"><span>${x[0]}</span><strong>${x[1]} <small>${x[2]}</small></strong></article>`).join('');
  if(group==='asr'){
    document.querySelector('#detailHead').innerHTML='<tr><th>测试集</th><th>语言 / 场景</th><th>错误率 ↓</th><th>RTF ↓</th><th>音频规模</th></tr>';
    document.querySelector('#detailBody').innerHTML=ASR_DATASETS.map((d,i)=>{const error=Math.max(1.4,model.score+d[4]+((model.logo.charCodeAt(0)+i)%5)*.11);const rtf=model.rtf*(1+i*.08);return `<tr><td class="dataset-name">${d[0]}<small>${d[3].toUpperCase()}</small></td><td>${d[1]}</td><td class="metric-good">${error.toFixed(2)}%</td><td>${rtf.toFixed(2)}×</td><td>${d[2]}</td></tr>`}).join('');
  }else{
    document.querySelector('#detailHead').innerHTML='<tr><th>测试集</th><th>语言 / 场景</th><th>MOS ↑</th><th>WER ↓</th><th>SIM ↑</th></tr>';
    document.querySelector('#detailBody').innerHTML=TTS_DATASETS.map((d,i)=>{const mos=Math.max(3.4,model.score+d[4]-((model.logo.charCodeAt(0)+i)%4)*.025);const wer=2.18+i*.47+(4.65-model.score)*1.3;const sim=.82+(model.score-4)*.10-i*.009;return `<tr><td class="dataset-name">${d[0]}<small>${d[2]}</small></td><td>${d[1]}</td><td class="metric-good">${mos.toFixed(2)}</td><td>${wer.toFixed(2)}%</td><td>${sim.toFixed(3)}</td></tr>`}).join('');
  }
  const labels=group==='asr'?['准确率','噪声鲁棒性','多语种','推理效率','部署易用性']:['自然度','说话人相似度','可懂度','推理效率','韵律稳定性'];
  document.querySelector('#profileBars').innerHTML=labels.map((label,i)=>`<div class="profile-row"><span>${label}</span><div class="profile-track"><i style="--score:${model.radar[i]}%"></i></div><b>${model.radar[i]}</b></div>`).join('');
}

document.querySelectorAll('[data-task]').forEach(b=>b.addEventListener('click',()=>setTask(b.dataset.task)));
typeFilter.addEventListener('change',renderTable);
document.querySelector('#datasetFilter').addEventListener('change',e=>showToast(`已切换至：${e.target.value}`));
document.querySelector('#showAll').addEventListener('click',()=>showToast('完整榜单将在数据接入后开放'));
document.querySelector('#themeToggle').addEventListener('click',()=>{const root=document.documentElement;root.dataset.theme=root.dataset.theme==='dark'?'light':'dark'});
document.querySelector('#mobileMenu').addEventListener('click',()=>document.querySelector('#sidebar').classList.toggle('open'));

const overlay=document.querySelector('#commandOverlay'),input=document.querySelector('#commandInput'),results=document.querySelector('#commandResults');
function openSearch(){overlay.hidden=false;input.value='';fillResults('');setTimeout(()=>input.focus(),0)}
function closeSearch(){overlay.hidden=true}
function fillResults(q){const items=[...DATA.asr,...DATA.tts].filter(m=>m.name.toLowerCase().includes(q.toLowerCase())).slice(0,6);results.innerHTML=items.length?items.map(m=>`<button class="command-result" data-name="${m.name}"><span>${m.name}</span><small>${m.org} · ${m.kind==='open'?'开源':'API'}</small></button>`).join(''):'<button class="command-result"><span>未找到匹配结果</span></button>';results.querySelectorAll('[data-name]').forEach(b=>b.addEventListener('click',()=>{closeSearch();openModel(b.dataset.name)}))}
document.querySelector('#searchTrigger').addEventListener('click',openSearch);input.addEventListener('input',()=>fillResults(input.value));overlay.addEventListener('click',e=>{if(e.target===overlay)closeSearch()});document.querySelector('#drawerClose').addEventListener('click',closeModel);document.querySelector('#drawerBackdrop').addEventListener('click',closeModel);document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openSearch()}if(e.key==='Escape'){closeSearch();closeModel()}});

renderTable();drawRadar(DATA.asr[0]);
