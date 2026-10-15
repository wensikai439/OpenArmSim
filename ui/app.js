import {createAgentUI} from './agent.js?v=platform-24';
import {createSceneUI} from './scene.js?v=platform-25';
const $ = (id) => document.getElementById(id);
document.querySelectorAll('[data-demo-prompt]').forEach(button => {
  button.addEventListener('click', () => {
    const input = $('chat-input');
    input.value = button.dataset.demoPrompt;
    input.dispatchEvent(new Event('input', {bubbles:true}));
    input.focus();
  });
});
const labels = {operator:'正面观察',overview:'全景',head:'头部 · 模型相机',left:'左腕 · 模型相机',right:'右腕 · 模型相机'};
const statusLabels = {success:'仿真任务成功',primitive_completed:'技能执行完成 · 完整目标另行复查',invalid:'物理状态异常',cancelled:'已停止',error:'执行失败',reference_exhausted:'轨迹已播完，任务未成功',budget_exhausted:'步数预算耗尽，任务未成功',truncated:'达到时长上限'};
let state, camera='operator', imageToken='', sensorToken='', frames=[], frameIndex=0, shownRun='', toastTimer;
let lastAgentEvent=0, learningId='', learningMetrics=[], calibrationRefreshKey='';
let alignment=null, alignmentRows=[], alignmentRow=null, modelConfig=null;
const frontView={mode:'front',azimuth:180,elevation:-24,distance:1.25,lookat:[.43,0,1.20]};
let operatorView=structuredClone(frontView), viewerLoaded=false, viewSending=false, viewPending=null, drag=null, suppressClick=false;
async function api(path, body) {
  const response = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
function toast(message) { $('toast').textContent=message; $('toast').hidden=false; clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,6000); }
function message(text, role='assistant') {
  const element=document.createElement('article');element.className=`message ${role}`;
  const label=document.createElement('span');label.className='message-label';label.textContent=role==='user'?'你':'OpenArm Agent';
  const body=document.createElement('p');body.className='message-body';body.textContent=text;
  element.append(label,body);$('messages').append(element);$('messages').scrollTop=$('messages').scrollHeight;
}
message('描述任务后，模型会制定阶段、调用工具，并在运行后复查。过程与记忆都会保留。');
const agentUI=createAgentUI({api,toast,message});
const sceneUI=createSceneUI({api,toast});
function renderModel(config, editing=false){
  modelConfig=config;
  $('model-summary').hidden=!config.configured||editing;
  $('model-form').hidden=config.configured&&!editing;
  $('model-current-name').textContent=config.model;
  $('model-current-base').textContent=config.base_url;
  $('model-current-mode').textContent=config.vision?'文字 + 三路相机':'文字';
  $('model-storage-note').textContent=config.storage==='system_keyring'?'密钥保存在本机系统密钥环，重启后自动恢复。':'当前密钥只在服务内存，保存配置后会使用本机存储。';
  $('model-base').value=config.base_url;$('model-name').value=config.model;$('model-vision').checked=config.vision;
  $('model-key').value='';$('model-key').required=!config.configured;
  $('model-key').placeholder=config.configured?'留空，继续使用已保存的密钥':'输入密钥';
  $('model-key-hint').textContent=config.configured?'留空保留现有密钥':'';
  $('model-cancel').hidden=!config.configured;
  $('model-result').textContent=config.storage_error||'';
}
function modelBusy(busy){for(const id of ['model-save','model-test','model-clear','model-edit','model-test-saved','model-cancel'])$(id).disabled=busy;}
$('planner-mode').addEventListener('click',async()=>{
  try{renderModel(await api('/api/model'));$('model-dialog').showModal();}catch(error){toast(error.message);}
});
$('model-edit').addEventListener('click',()=>{renderModel(modelConfig,true);$('model-name').focus();});
$('model-cancel').addEventListener('click',()=>renderModel(modelConfig));
$('model-close').addEventListener('click',()=>{$('model-key').value='';$('model-dialog').close();});
$('model-dialog').addEventListener('close',()=>{$('model-key').value='';});
$('model-base').addEventListener('input',()=>{$('model-key').required=!modelConfig?.configured||$('model-base').value.trim().replace(/\/$/,'')!==modelConfig.base_url;});
$('model-form').addEventListener('submit',async event=>{
  event.preventDefault();const test=event.submitter?.dataset.test==='1';modelBusy(true);
  try{
    const values={base_url:$('model-base').value,model:$('model-name').value,vision:$('model-vision').checked};
    if($('model-key').value.trim())values.api_key=$('model-key').value;
    renderModel(await api('/api/model',values));
    $('model-result').textContent='已保存。之后可直接使用，无需重复输入。';
    if(test){$('model-result').textContent='正在测试连接…';const result=await api('/api/model/test',{});$('model-result').textContent=`连接成功 · ${result.latency_seconds} 秒。`;}
    toast('API 配置已保存');
  }catch(error){$('model-key').value='';$('model-result').textContent=error.message;}
  finally{modelBusy(false);}
});
$('model-test-saved').addEventListener('click',async()=>{
  modelBusy(true);$('model-result').textContent='正在测试已保存的连接…';
  try{const result=await api('/api/model/test',{});$('model-result').textContent=`连接成功 · ${result.latency_seconds} 秒。`;}
  catch(error){$('model-result').textContent=error.message;}finally{modelBusy(false);}
});
$('model-clear').addEventListener('click',async()=>{
  modelBusy(true);try{renderModel(await api('/api/model/clear',{}));$('model-result').textContent='配置和密钥已删除。';}catch(error){$('model-result').textContent=error.message;}finally{modelBusy(false);}
});
$('tamp-button').addEventListener('click',async()=>{
  $('tamp-button').disabled=true;
  try{const task=$('tamp-task').value;const observed=await api('/api/tools',{name:'observe',arguments:{}});await api('/api/tools',{name:'run_tamp',arguments:{observation_id:observed.observation_id,task}});message(`已直接启动 TAMP：${$('tamp-task').selectedOptions[0].textContent}。规划器使用仿真场景信息，每个操作都会检查前置条件、碰撞和接触；这次测试没有调用 LLM。`,'note');}
  catch(error){toast(error.message);$('tamp-button').disabled=false;}
});
async function command(name, args={}) {
  try {return await api('/api/command',{command:name,arguments:args});} catch(error){toast(error.message);throw error;}
}
function setCamera(next) {
  camera=next;imageToken='';
  const free=next==='operator'&&operatorView.mode==='free';
  $('camera-label').textContent=next==='operator'?(free?'自由视角':'正面观察'):labels[next];
  $('view-kind').textContent=['head','left','right'].includes(next)?'模型的固定输入':'操作者视角';
  $('view-select').value=next==='operator'?operatorView.mode:next;
  $('orbit-tools').hidden=!free;$('orbit-hint').hidden=!free;$('scene-viewport').classList.toggle('orbit-active',free);
  $('overview-button').hidden=['head','left','right'].includes(next);
  document.querySelectorAll('[data-camera]').forEach(el=>el.classList.toggle('selected',el.dataset.camera===camera));
  if(state)updateImages(state);
}
function updateImages(data) {
  const token=`${data.observation_id}:${camera}:${camera==='operator'?data.viewer?.revision:''}`;
  if(token!==imageToken){imageToken=token;$('main-camera').src=`/api/frame?camera=${camera}&v=${encodeURIComponent(data.observation_id)}&view=${data.viewer?.revision??0}`;}
  if(sensorToken!==data.observation_id){sensorToken=data.observation_id;for(const name of ['head','left','right'])$(`thumb-${name}`).src=`/api/frame?camera=${name}&v=${encodeURIComponent(data.observation_id)}`;}
}
async function sendView(){
  if(viewSending)return;viewSending=true;
  try{while(viewPending){const value=viewPending;viewPending=null;await api('/api/view',value);}}
  catch(error){viewPending=null;toast(error.message);}finally{viewSending=false;}
}
function queueView(){viewPending=structuredClone(operatorView);sendView();}
function chooseView(mode){
  viewerLoaded=true;
  if(mode==='front'){operatorView=structuredClone(frontView);queueView();setCamera('operator');}
  else if(mode==='free'){operatorView.mode='free';queueView();setCamera('operator');}
  else setCamera(mode);
}
const clamp=(value,min,max)=>Math.min(max,Math.max(min,value));
function moveOrbit(dx,dy,pan=false){
  if(pan){const a=operatorView.azimuth*Math.PI/180,scale=operatorView.distance*.0015;
    operatorView.lookat[0]=clamp(operatorView.lookat[0]-Math.sin(a)*dx*scale,-.5,1.5);
    operatorView.lookat[1]=clamp(operatorView.lookat[1]+Math.cos(a)*dx*scale,-1.2,1.2);
    operatorView.lookat[2]=clamp(operatorView.lookat[2]+dy*scale,.3,2);
  }else{operatorView.azimuth=((operatorView.azimuth-dx*.4+540)%360)-180;operatorView.elevation=clamp(operatorView.elevation-dy*.3,-85,35);}
  queueView();
}
function zoomOrbit(amount){operatorView.distance=clamp(operatorView.distance*Math.exp(amount),.65,3.5);queueView();}
$('main-camera').addEventListener('load',()=>{$('loading').hidden=true;});
$('main-camera').draggable=false;
document.querySelectorAll('[data-camera]').forEach(el=>el.addEventListener('click',()=>setCamera(el.dataset.camera)));
$('view-select').addEventListener('change',()=>chooseView($('view-select').value));
$('overview-button').addEventListener('click',()=>chooseView('front'));
$('scene-viewport').addEventListener('pointerdown',event=>{
  if(camera!=='operator'||operatorView.mode!=='free'||event.button!==0)return;
  drag={x:event.clientX,y:event.clientY,moved:0};suppressClick=false;$('scene-viewport').setPointerCapture(event.pointerId);
});
$('scene-viewport').addEventListener('pointermove',event=>{
  if(!drag)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;drag.moved+=Math.abs(dx)+Math.abs(dy);
  drag.x=event.clientX;drag.y=event.clientY;if(drag.moved>3){suppressClick=true;moveOrbit(dx,dy,event.shiftKey);}
});
for(const name of ['pointerup','pointercancel','lostpointercapture'])$('scene-viewport').addEventListener(name,()=>{drag=null;});
$('scene-viewport').addEventListener('wheel',event=>{if(camera==='operator'&&operatorView.mode==='free'){event.preventDefault();zoomOrbit(clamp(event.deltaY,-150,150)*.0015);}},{passive:false});
$('scene-viewport').addEventListener('keydown',event=>{
  if(camera!=='operator'||operatorView.mode!=='free')return;
  const direction={ArrowLeft:[-25,0],ArrowRight:[25,0],ArrowUp:[0,-20],ArrowDown:[0,20]}[event.key];
  if(direction){event.preventDefault();moveOrbit(...direction);}
  else if(['+','-','0'].includes(event.key)){event.preventDefault();event.key==='0'?chooseView('front'):zoomOrbit(event.key==='+'?-.12:.12);}
});
document.querySelectorAll('[data-orbit]').forEach(button=>button.addEventListener('click',()=>{
  const action=button.dataset.orbit;if(action==='left'||action==='right')moveOrbit(action==='left'?-30:30,0);else zoomOrbit(action==='in'?-.12:.12);
}));
function updateFrames(next) {
  if(JSON.stringify(next)===JSON.stringify(frames))return;frames=next;
  $('keyframes').replaceChildren();
  if(!frames.length){const el=document.createElement('div');el.className='empty';el.textContent='开始一轮实验，留下可核验的过程。';$('keyframes').append(el);return;}
  frames.forEach((frame,index)=>{
    const button=document.createElement('button');button.className='frame-card';button.title=`${frame.label} · ${frame.sim_time.toFixed(2)} s`;
    if(frame.outcome)button.dataset.outcome=frame.outcome;
    const img=document.createElement('img');img.src=frame.url;img.alt=frame.label;img.loading='lazy';
    const text=document.createElement('span');text.textContent=`${frame.label} · ${frame.sim_time.toFixed(1)}s`;
    button.append(img,text);button.addEventListener('click',()=>showFrame(index));$('keyframes').append(button);
  });
}
function showImage(url,title,meta) {
  $('frame-large').classList.remove('zoomed');$('frame-large').style.width='';$('frame-scale').textContent='放大 2×';$('frame-large').src=url;$('frame-title').textContent=title;$('frame-meta').textContent=meta;
  if(!$('frame-dialog').open)$('frame-dialog').showModal();
}
function showFrame(index) {if(!frames.length)return;frameIndex=(index+frames.length)%frames.length;const f=frames[frameIndex];showImage(f.url,f.label,`${f.sim_time.toFixed(3)} s · ${f.camera}`);$('frame-prev').disabled=false;$('frame-next').disabled=false;}
function renderAlignment(){
  if(!alignment?.samples.length)return;
  alignmentRows=alignment.samples.filter(r=>r.episode===Number($('alignment-episode').value));
  const index=Math.min(Number($('alignment-frame').value),alignmentRows.length-1);
  $('alignment-frame').max=alignmentRows.length-1;$('alignment-frame').value=index;
  $('alignment-phase').replaceChildren(...alignmentRows.map((r,i)=>{
    const option=document.createElement('option');option.value=i;
    option.textContent=`${(r.command_landmarks||[]).map(s=>s.label).join(' / ')||'过程采样'} · ${r.frame} 帧`;return option;
  }));$('alignment-phase').value=index;
  alignmentRow=alignmentRows[index];const row=alignmentRow,cam=$('alignment-camera').value;
  for(const kind of ['real','geometry','dynamics']){
    const img=$(`alignment-${kind}`),url=row.images[cam].urls[kind];
    if(url)img.src=url;else img.removeAttribute('src');img.hidden=!url;$(`alignment-${kind}-button`).disabled=!url;
  }
  $('alignment-dynamics-note').textContent=row.dynamics_available?'从初态积分真实指令；关节位置由物理产生。':row.dynamics_unavailable_reason;
  $('alignment-prev').disabled=index===0;$('alignment-next').disabled=index===alignmentRows.length-1;
  $('alignment-time').textContent=`第 ${row.frame} 帧 · 记录 ${row.recorded_seconds.toFixed(3)} s`;
  $('alignment-note').textContent=cam==='head'?'头部原图有明显鱼眼畸变，模型仍使用透视相机。先核对投影与坐标，再判断动作响应。':'左右腕相机按原始记录对应；夹爪反馈到开口的映射仍是候选值。点击画面可放大。';
  $('alignment-metrics').textContent=row.dynamics_available?`动作响应差：14 关节 RMSE ${row.joint_rmse_rad.toFixed(4)} rad · 左 / 右末端 ${(row.ee_body_model_delta_m.left*1000).toFixed(1)} / ${(row.ee_body_model_delta_m.right*1000).toFixed(1)} mm（模型计算） · 时钟偏差 ${(row.clock_error_seconds*1000).toFixed(3)} ms`:'物理回放已终止，本帧只提供真实记录与几何对照。';
  const sim=row.simulation;
  $('alignment-contact').textContent=sim?`仿真已达：${sim.reached_stages.at(-1)} · 当前夹持判定：左${sim.held.left?'是':'否'} / 右${sim.held.right?'是':'否'} · 篮内：${sim.inside_basket?'是':'否'} · 篮底支撑：${sim.supported_by_basket?'是':'否'}。${sim.failure_reasons.join('；')} 记录时刻按夹爪指令选择，真机是否抓稳需查看画面。`:'';
}
$('alignment-button').addEventListener('click',async()=>{
  $('alignment-dialog').showModal();
  alignment=null;alignmentRows=[];alignmentRow=null;$('alignment-grid').hidden=true;
  $('alignment-note').textContent='读取原始帧和模型对照…';
  for(const id of ['alignment-contact','alignment-metrics','alignment-time'])$(id).textContent='';
  for(const id of ['alignment-episode','alignment-camera','alignment-phase','alignment-frame','alignment-prev','alignment-next'])$(id).disabled=true;
  try{
    alignment=await api('/api/alignment');const available=alignment.samples.length>0;
    $('alignment-grid').hidden=!available;
    $('alignment-version').textContent=available?(alignment.matches_current_model?'当前配置':'历史配置'):'尚无对照';
    $('alignment-report').hidden=!available;if(available)$('alignment-report').href=alignment.report_url;
    for(const id of ['alignment-episode','alignment-camera','alignment-phase','alignment-frame','alignment-prev','alignment-next'])$(id).disabled=!available;
    if(available)renderAlignment();else{$('alignment-note').textContent='尚未生成经过原始帧核验的对照。';$('alignment-time').textContent='';$('alignment-metrics').textContent='';}
  }catch(error){$('alignment-note').textContent=error.message;$('alignment-grid').hidden=true;toast(error.message);}
});
$('alignment-episode').addEventListener('change',()=>{$('alignment-frame').value=0;renderAlignment();});
$('alignment-camera').addEventListener('change',renderAlignment);
$('alignment-phase').addEventListener('change',()=>{$('alignment-frame').value=$('alignment-phase').value;renderAlignment();});
$('alignment-frame').addEventListener('input',renderAlignment);
for(const [id,delta] of [['alignment-prev',-1],['alignment-next',1]])$(id).addEventListener('click',()=>{$('alignment-frame').value=Number($('alignment-frame').value)+delta;renderAlignment();});
for(const kind of ['real','geometry','dynamics'])$(`alignment-${kind}-button`).addEventListener('click',()=>{
  if(!alignmentRow)return;const cam=$('alignment-camera').value,url=alignmentRow.images[cam].urls[kind];if(!url)return;
  const name={real:'真机原始画面',geometry:'记录关节位置 · 几何对照',dynamics:'指令动力学回放'}[kind];
  showImage(url,name,`轨迹 ${alignmentRow.episode+1} · ${labels[cam]} · 第 ${alignmentRow.frame} 帧 · ${alignmentRow.recorded_seconds.toFixed(3)} s`);
  $('frame-prev').disabled=true;$('frame-next').disabled=true;
});
$('frame-prev').addEventListener('click',()=>showFrame(frameIndex-1));$('frame-next').addEventListener('click',()=>showFrame(frameIndex+1));
$('frame-scale').addEventListener('click',()=>{const img=$('frame-large'),width=img.getBoundingClientRect().width,on=img.classList.toggle('zoomed');img.style.width=on?`${width*2}px`:'';$('frame-scale').textContent=on?'适应窗口':'放大 2×';});
document.querySelectorAll('.close-dialog').forEach(button=>button.addEventListener('click',()=>button.closest('dialog').close()));
$('frame-dialog').addEventListener('keydown',event=>{if(event.key==='ArrowLeft'&&!$('frame-prev').disabled){event.preventDefault();showFrame(frameIndex-1);}if(event.key==='ArrowRight'&&!$('frame-next').disabled){event.preventDefault();showFrame(frameIndex+1);}});
for(const id of ['calibration-button','experiment-button'])$(id).addEventListener('click',()=>{$('details-dialog').showModal();loadCalibration().catch(error=>toast(error.message));});
async function loadCalibration(){
  const data=await api('/api/calibration');
  const clocks=data.acquisition?.episodes;
  if(clocks){const rows=Object.values(clocks);$('acquisition-summary').textContent=`三条原始轨迹已逐帧核对；平均记录频率 ${Math.min(...rows.map(r=>r.average_hz)).toFixed(2)}–${Math.max(...rows.map(r=>r.average_hz)).toFixed(2)} Hz。辨识使用原始记录时钟，不再按帧号 ÷ 30；曝光与 CAN 时钟仍待核验。头部视频实际为 1280×960，旧元数据的 720 高度有误。`;}
  const statuses={candidate:'候选',missing:'缺少测量',ported:'其他部署记录',software_match:'软件比对吻合',unverified:'待核验'};
  $('calibration-evidence').replaceChildren(...data.parameters.map(p=>{
    const row=document.createElement('details');row.className='evidence-row';
    const summary=document.createElement('summary'),label=document.createElement('strong'),badge=document.createElement('span');
    label.textContent=p.label;badge.className='badge';badge.textContent=statuses[p.status]||p.status;summary.append(label,badge);
    const text=document.createElement('p');text.textContent=p.evidence;
    const needed=document.createElement('p');needed.className='evidence-needed';needed.textContent='待补：'+p.needed;
    row.append(summary,text,needed);if(p.value!==null&&typeof p.value!=='object'&&p.value!==undefined){const value=document.createElement('p');value.textContent=`运行值：${p.value}${p.unit?' '+p.unit:''}`;row.append(value);}return row;
  }));
  const result=$('calibration-study-result');result.replaceChildren();
  if(data.latest_study){const s=data.latest_study,p=document.createElement('p');
    p.textContent=`开发验证误差比 ${(s.holdout_error_ratio*100).toFixed(1)}%（当前模型为 100%）${s.at_parameter_bound?' · 参数触及搜索边界':''}。${s.matches_current_model?'':'模型已变化，此报告属于旧配置。'}候选尚未应用。`;
    const a=document.createElement('a');a.textContent='查看辨识报告 ↗';a.href=s.result_url;a.target='_blank';a.rel='noopener';result.append(p,a);
    if(s.plot_url){const plot=document.createElement('a');plot.textContent='查看对比曲线 ↗';plot.href=s.plot_url;plot.target='_blank';plot.rel='noopener';result.append(plot);}
  }
  const wrist=$('wrist-study-result');wrist.replaceChildren();
  if(data.latest_wrist){const s=data.latest_wrist,p=document.createElement('p');
    p.textContent=Object.entries(s.development).map(([side,v])=>`${side==='left'?'左腕':'右腕'} ${v.scored_frames} 帧：轮廓 IoU ${v.baseline_iou?.toFixed(2)??'—'} → ${v.candidate_iou?.toFixed(2)??'—'}${v.at_parameter_bound?'（参数接近边界）':''}`).join('；')+`。${s.excluded.length} 帧因遮挡或分割歧义排除。${s.matches_current_model?'':'当前模型已变化。'}候选未应用。`;
    const a=document.createElement('a');a.textContent='查看原始图与候选投影 ↗';a.href=s.report_url;a.target='_blank';a.rel='noopener';wrist.append(p,a);
  }
  const gripper=$('gripper-study-result');gripper.replaceChildren();
  if(data.latest_gripper){const s=data.latest_gripper,p=document.createElement('p'),r=s.aggregate_rmse_rad;
    p.textContent=`相同记录时钟下的电机角度误差：左 ${r.legacy[0].toFixed(3)} → ${r.shared_motor_gain[0].toFixed(3)} rad；右 ${r.legacy[1].toFixed(3)} → ${r.shared_motor_gain[1].toFixed(3)} rad。${s.matches_current_model?'':'当前模型已变化，此报告属于历史配置。'}真实开口与传动仍待标定。`;
    const a=document.createElement('a');a.textContent='查看夹爪曲线与覆盖范围 ↗';a.href=s.report_url;a.target='_blank';a.rel='noopener';gripper.append(p,a);
  }else{gripper.textContent='尚无完成的夹爪对照实验。';}
}
$('fit-wrist').addEventListener('click',async()=>{
  $('fit-wrist').disabled=true;
  try{const job=await api('/api/tools',{role:'research',name:'fit_wrist_camera',arguments:{}});toast('腕部投影实验已启动');message('腕部相机候选实验：'+job.job_id+'。固定拟合与开发验证分组，不改变运行参数。','note');}
  catch(error){toast(error.message);}finally{$('fit-wrist').disabled=false;}
});
$('fit-controller').addEventListener('click',async()=>{
  $('fit-controller').disabled=true;
  try{const job=await api('/api/tools',{role:'research',name:'fit_controller',arguments:{}});toast('右臂辨识已启动，候选参数不会自动应用');message('已启动右臂控制响应辨识：'+job.job_id+'。两条轨迹用于拟合，另一条用于开发验证；不代表全新测试集。','note');}
  catch(error){toast(error.message);}finally{$('fit-controller').disabled=false;}
});
$('timing-study').addEventListener('click',async()=>{
  $('timing-study').disabled=true;
  try{const job=await api('/api/tools',{role:'research',name:'evaluate_timing',arguments:{}});toast('延迟对照已启动');message('已启动延迟对照：'+job.job_id+'。三条记录轨迹 × 三种等待时长，不调用真实 VLA，不改变标定参数。','note');}
  catch(error){toast(error.message);}finally{$('timing-study').disabled=false;}
});
$('capture-button').addEventListener('click',()=>command('capture',{camera}).then(()=>toast('关键帧已保存')).catch(()=>{}));
$('reset-button').addEventListener('click',()=>command('reset',{episode:Number($('episode-select').value)}).catch(()=>{}));
$('replay-button').addEventListener('click',()=>command('replay',{episode:Number($('episode-select').value),record_images:$('record-images').checked}).then(()=>message('按原始记录时间回放真机指令，保留不等间隔。接触结果由物理引擎产生；当前仍使用候选参数。','note')).catch(()=>{}));
$('stop-button').addEventListener('click',()=> (state?.agent?.busy?api('/api/tools',{role:'execute',name:'stop',arguments:{}}):command('stop')).then(()=>toast('已请求停止；不会应用后续模型动作')).catch(error=>toast(error.message)));
$('stop-learning').addEventListener('click',()=>api('/api/tools',{role:'research',name:'stop_job',arguments:{job_id:learningId}}).then(()=>toast('已请求停止本次训练任务')).catch(error=>toast(error.message)));
function renderLearning(job){
  $('learning-card').hidden=!job;if(!job)return;
  if(learningId!==job.job_id){learningId=job.job_id;learningMetrics=[];}
  const names={export:'数据导出与校验',train:'ACT 模型训练',evaluate:'独立 episode 预测评估',calibrate:'右臂响应辨识',wrist:'腕部相机投影'};
  const statuses={starting:'准备中',running:'运行中',stopping:'正在停止',completed:'已完成',failed:'未通过',cancelled:'已停止',timed_out:'达到时间预算',unknown:'进程状态待核实'};
  $('learning-title').textContent=job.kind==='timing'?'推理延迟对照':names[job.kind]||job.kind;$('learning-status').textContent=statuses[job.status]||job.status;
  if(job.metrics)learningMetrics=job.metrics;
  const metric=job.latest_metric;if(metric&&!learningMetrics.some(m=>m.step===metric.step))learningMetrics.push(metric);
  $('learning-detail').textContent=(metric?`第 ${metric.step} / ${job.steps} 步 · loss ${metric.loss.toFixed(4)} · `:'')+(job.scope==='pipeline_validation_only'?'仅验证训练链路，尚未训练出任务能力。':job.kind==='evaluate'?'只比较动作预测误差，不代表任务成功率。':'结果仍需独立评测，不自动认定可迁移。');
  if(job.offline_metrics)$('learning-detail').textContent=`${job.offline_metrics.frames} 帧 · 关节 RMSE ${job.offline_metrics.joint_rmse_rad.toFixed(4)} rad。离线预测误差，不是任务成功率。`;
  if(job.kind==='timing')$('learning-detail').textContent=`已完成 ${job.timing_progress?.completed??0} / 9 组 · 0 / 100 / 300 ms 等待。记录动作的物理对照，不是真实 VLA 测试。`;
  if(job.kind==='calibrate'){
    const p=job.calibration_progress,r=job.calibration_result;
    $('learning-detail').textContent=r?`开发验证误差为当前模型的 ${(r.holdout_error_ratio*100).toFixed(1)}%${r.at_parameter_bound?' · 参数触及边界':''}。仅前 4 秒响应，候选未应用。`:`${p?`已比较 ${p.completed} / ${p.total} 组延迟候选。`:'正在拟合控制响应。'}验证轨迹不参与本次参数选择。`;
  }
  if(job.kind==='wrist'){
    const phase={prepare:'核验参考帧',fit:'拟合几何投影',refine:'检查原生遮挡',masks:'核验开发帧',verify:'生成原图对照'}[job.wrist_progress?.phase];
    $('learning-detail').textContent=job.wrist_result?'24 组原图与候选投影已保存。夹爪和相机仍待独立标定，候选未应用。':`${phase||'准备腕部相机实验'}；固定分组，候选不覆盖运行参数。`;
  }
  $('learning-plot').hidden=!job.plot_url;if(job.plot_url)$('learning-plot').href=job.plot_url;
  const active=['starting','running','stopping'].includes(job.status);$('stop-learning').disabled=!active;$('learning-result').hidden=job.status!=='completed';$('learning-result').href=job.result_url;
  if(job.wrist_result)$('learning-result').href=job.wrist_result.report_url;
  $('loss-chart').replaceChildren();if(learningMetrics.length){const ys=learningMetrics.map(m=>m.loss),lo=Math.min(...ys),hi=Math.max(...ys),line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('points',learningMetrics.map((m,i)=>`${8+i*384/Math.max(1,learningMetrics.length-1)},${48-(m.loss-lo)*38/Math.max(hi-lo,.000001)}`).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke','#78916f');line.setAttribute('stroke-width','2');$('loss-chart').append(line);}
  $('loss-chart').style.display=learningMetrics.length?'':'none';
}
function render(data) {
  state=data;const running=data.state==='running';$('connection').textContent=`${running?'实验运行中':'仿真已连接'} · OpenArm ${data.robot_model||'版本待确认'}`;
  sceneUI.render(data);
  if(!viewerLoaded&&data.viewer){const {revision,...view}=data.viewer;operatorView=view;viewerLoaded=true;setCamera('operator');}
  $('fit-controller').disabled=running||['starting','running','stopping','unknown'].includes(data.learning_job?.status);
  $('timing-study').disabled=$('fit-controller').disabled;
  $('fit-wrist').disabled=$('fit-controller').disabled;
  $('planner-mode').textContent=data.llm?'API 配置':'接入模型';
  $('tamp-button').disabled=running||data.agent?.busy||['starting','running','stopping','unknown'].includes(data.learning_job?.status);
  $('tamp-task').disabled=$('tamp-button').disabled;
  $('sim-clock').textContent=`${data.observation.sim_time.toFixed(2)} s`;
  $('physics-status').textContent=running?(data.job?(data.job.kind==='replay'?'原始记录时钟 · 1 ms 物理':'30 Hz 控制 · 服务端物理'):'模型等待 · 保持上一控制目标'):'已暂停 · 原生 MuJoCo';
  const evaluation=data.diagnostics.evaluation;
  $('contact-status').textContent=`最大穿入 ${(evaluation.max_penetration_m*1000).toFixed(1)} mm`;
  for(const id of ['reset-button','replay-button','capture-button','episode-select'])$(id).disabled=running||data.agent?.busy;
  $('stop-button').disabled=!running&&!data.agent?.busy;$('progress-bar').style.width=`${data.job?(data.job.progress||0)*100:0}%`;
  const events=new Map(data.diagnostics.events.map(e=>[e.stage,e]));
  const modelPlan=agentUI.render(data);
  if(!modelPlan){
  const run=data.job||data.last_run,gripper=(run?.task||run?.metrics?.tamp?.task)==='set_gripper',cartesian=['move_relative','move_to_pose'].includes(run?.task||run?.metrics?.tamp?.task)||gripper;
  if(cartesian){const done=!!data.last_run?.metrics?.tamp?.primitive_success&&!data.job;const li=document.createElement('li');li.className=done?'done':'';const dot=document.createElement('span');dot.className='stage-dot';dot.textContent=done?'✓':'1';li.append(dot,document.createTextNode(gripper?'夹爪开合':'末端到位'));$('stage-list').replaceChildren(li);$('stage-count').textContent=`${done?1:0} / 1`;}
  else{$('stage-count').textContent=`${events.size} / ${data.stages.length}`;
  $('stage-list').replaceChildren(...data.stages.map((name,index)=>{const li=document.createElement('li');li.className=events.has(index)?'done':'';const dot=document.createElement('span');dot.className='stage-dot';dot.textContent=events.has(index)?'✓':index+1;li.append(dot,document.createTextNode(name));if(events.has(index))li.title=events.get(index).evidence;return li;}));}
  }
  if(running){$('result-line').textContent=!data.job?`模型决策等待 · 保持上一控制目标 · ${data.observation.sim_time.toFixed(1)} s`:data.job.kind==='tamp'?`运动规划 · ${data.job.phase||'准备规划'} · 使用仿真场景信息 · ${data.observation.sim_time.toFixed(1)} s`:`${data.job.kind==='replay'?'真机指令回放 · 原始时间':data.job.timing_mode==='measured_delay'?'VLA 执行 · 计入推理等待':'VLA 执行 · 离线暂停'} · ${(data.job.progress*100).toFixed(0)}% · 结果尚未判定`;$('result-line').classList.remove('error');}
  else if(data.last_run){const run=data.last_run;$('result-line').textContent=`${statusLabels[run.status]||run.status}${run.error?' · '+run.error:''}`;$('result-line').classList.toggle('error',!['success','primitive_completed'].includes(run.status));if((run.run_id||run.error)!==shownRun){shownRun=run.run_id||run.error;const feedback=run.metrics?.tamp;const diagnostic=feedback?.planning_diagnostics;const extra=diagnostic?`\n规划阶段：${diagnostic.arm==='left'?'左手':'右手'} / ${diagnostic.phase}。${diagnostic.blocked_pairs?.length?'碰撞对象：'+diagnostic.blocked_pairs[0].geoms.join(' ↔ '):'最小位置误差：'+((diagnostic.best_ik?.position_error_m||0)*1000).toFixed(1)+' mm'}。`:'';if(!run.initiator||run.initiator.type!=='llm')message($('result-line').textContent+'。'+extra+(run.run_id?'状态、动作及结果已保存到本次实验目录。':''),'note');}}
  else{$('result-line').textContent='阶段由接触与稳定性证据确认，播放到某个时间点不会自动判定完成。';}
  const assessment=data.agent?.plan;
  if(!running&&data.last_run&&assessment?.execution_id===data.last_run.execution_id){
    const verdict={success:'成功',failure:'失败',uncertain:'证据不足'}[assessment.verdict]||(data.agent.busy?'复查中':'尚未提交');
    const feedback=data.last_run.metrics?.tamp,arms={left:'左手',right:'右手'};
    const objectGoal=assessment.goal_assessment;
    const precision=objectGoal?.confirmed&&Number.isFinite(objectGoal.position_error_m)&&Number.isFinite(objectGoal.tolerance_m)?` · 误差 ${(objectGoal.position_error_m*1000).toFixed(1)} mm / 允许 ${(objectGoal.tolerance_m*1000).toFixed(1)} mm`:'';
    const goalCount=objectGoal?.objects?.length||0,confirmedCount=objectGoal?.objects?.filter(g=>g.confirmed).length||0;
    const taskAcceptance=goalCount?objectGoal.confirmed?`${confirmedCount}/${goalCount} 个物体到达原目标${precision.replace('误差','最大误差')} · 均已松手并稳定支撑`:`${confirmedCount}/${goalCount} 个物体到达原目标 · 整项任务尚未确认`:null;
    const acceptance=taskAcceptance||(objectGoal?.confirmed?`物体到达锁定目标${precision} · 已松手并稳定支撑`:feedback?.verification_source==='robot_gripper_encoder_settling'&&feedback.primitive_success?'夹爪开合已完成（不代表夹持）':feedback?.primitive_success&&feedback.verification_source==='native_support_release_and_stability'?`${feedback.surface==='table'?'桌面目标':'篮内'}放置已确认 · 已松手并稳定支撑`:feedback?.primitive_success&&feedback.verification_source==='native_receiver_contact_then_donor_release'?`交接已确认：${arms[feedback.from_arm]} → ${arms[feedback.to_arm]}独立持物`:feedback?.primitive_success&&feedback.verification_source==='native_two_finger_contact_and_lift'?`${arms[feedback.held_by]}夹持与抬升已确认`:data.last_run.status==='success'?'篮内稳定验收通过':statusLabels[data.last_run.status]||data.last_run.status);
    $('result-line').textContent=`${acceptance} · 模型复查：${verdict}`;
  }
  const feedback=data.last_run?.metrics?.tamp;
  if(data.decision_timing?.status==='running'&&data.agent?.busy){$('result-line').textContent='模型正在决策 · 机械臂保持当前目标 · 物理继续运行';$('result-line').classList.remove('error');}
  const sliding=data.decision_timing?.root===assessment?.turn_id&&Object.values(data.decision_timing?.grasp_stability?.arms||{}).some(a=>a.status==='sliding'||a.status==='contact_lost');
  if(!running&&sliding){$('result-line').textContent='持物状态不稳定 · 已记录滑动反馈，等待重新决策';$('result-line').classList.add('error');}
  if(data.decision_timing?.task_failed&&data.decision_timing.root===assessment?.turn_id){const label={budget_exhausted:'等待期间到达任务预算',physics_invalid:'等待期间物理失效',physics_error:'等待期间物理错误',cancelled:'任务已停止'}[data.decision_timing.status]||'等待阶段未通过';$('result-line').textContent=label+' · 后续动作已禁止';$('result-line').classList.add('error');}
  const reversed=feedback?.from_arm==='left'&&feedback?.to_arm==='right';
  const goalFrames=data.agent?.goal_keyframes;
  let displayFrames=goalFrames?.length?goalFrames:(data.keyframes||[]);
  if(goalFrames?.length){const epoch=data.observation_id.split(':')[0];displayFrames=[...goalFrames,...(data.keyframes||[]).filter(f=>f.manual&&f.observation_id?.split(':')[0]===epoch)];}
  if(reversed)displayFrames=displayFrames.filter(f=>!['右手抬起','右手夹持'].includes(f.label)).map(f=>({...f,label:f.label==='双手交接'?'双手夹持确认':f.label.startsWith('结束')&&feedback.primitive_success?'右手独立持物':f.label}));
  if(feedback?.verification_source==='native_support_release_and_stability'&&feedback.primitive_success)
    displayFrames=displayFrames.map(f=>({...f,label:f.label.startsWith('结束')?'放置已稳定':f.label}));
  if(feedback?.primitive_success&&feedback.verification_source==='robot_forward_kinematics')
    displayFrames=displayFrames.map(f=>({...f,label:f.label.startsWith('结束')?'末端到位':f.label}));
  if(feedback?.primitive_success&&feedback.verification_source==='robot_gripper_encoder_settling')
    displayFrames=displayFrames.map(f=>({...f,label:f.label.startsWith('结束')?'夹爪已调整':f.label}));
  updateImages(data);updateFrames(displayFrames);
  renderLearning(data.learning_job);
  const completedCalibration=['calibrate','wrist'].includes(data.learning_job?.kind)&&data.learning_job.status==='completed'?data.learning_job.job_id:'';
  if($('details-dialog').open&&completedCalibration&&completedCalibration!==calibrationRefreshKey){calibrationRefreshKey=completedCalibration;loadCalibration().catch(error=>toast(error.message));}
  $('calibration-gaps').replaceChildren(...data.calibration_gaps.map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
  $('run-list').replaceChildren(...data.runs.map(name=>{const a=document.createElement('a');a.className='run-item';a.textContent=name;a.href=`/runs/${encodeURIComponent(name)}/${name.startsWith('handover-')?'episode':'summary'}.json`;a.target='_blank';a.rel='noopener';return a;}));
}
async function poll(){try{render(await api('/api/status'));}catch(error){$('connection').textContent='连接中断';$('loading').hidden=false;$('loading').textContent='连接中断；页面重连不会重新启动实验';}finally{setTimeout(poll,document.hidden?2000:camera==='operator'&&operatorView.mode==='free'?150:500);}}
poll();
