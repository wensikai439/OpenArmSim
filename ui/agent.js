const $ = id => document.getElementById(id);
const names = {set_gripper:'夹爪开合',observe:'观察场景',inspect_manipulation:'检查操作条件',prepare_simulation:'准备新仿真任务',set_plan:'制定阶段计划',pick:'抓取',handover:'双手交接',place:'放置',relocate:'连续搬到目标',move_to_pose:'移动到目标位姿',move_relative:'末端相对移动',run_policy:'调用学习策略',list_skills:'读取技能能力',run_tamp:'TAMP 基线',run_pi05_vla:'VLA 基线',get_run_status:'读取执行反馈',finish:'提交模型复查',recall_memory:'检索相关经验',remember_scene:'更新场景记忆'};
names.inspect_scene='读取场景目录';names.resolve_scene_target='解析相对物体目标';
names.resolve_placement_transit='计算运输位姿';
names.select_manipulation_object='选择操作对象';
const statuses = {running:'进行中',waiting:'等待执行',completed:'已结束',error:'失败',interrupted:'已中断',cancelled:'已停止',budget_exhausted:'预算耗尽'};
const marks = {done:'✓',running:'·',blocked:'!',uncertain:'?',failure:'!'};
const outcomes = {success:'技能成功',failure:'执行失败',rejected:'调用被拒绝',cancelled:'已取消',unverified:'结果未验证'};
const checks = {observation:'已读取观测',right_grasp:'右手夹持证据',right_lift:'右手抬起证据',handover:'双手交接证据',left_grasp:'右手打开后左手独立持瓶',left_release_at_basket:'左手在篮中释放瓶子',basket:'稳定入篮证据',tool_completed:'工具完成，不代表完整任务成功',visual_review:'模型视觉复查，与物理评分分开'};
checks.placement_confirmed='本次放置的目标、支撑、脱离夹爪与稳定';checks.motion_reached='绑定该次运动的末端位姿反馈';checks.grasp_confirmed='本次抓取的双指接触与抬升';checks.handover_confirmed='本次交接后接收手独立持物';
const skillHelp={relocate:'选择物体目标和机械臂，连续完成抓取、可选交接或携物移动、放置与撤离；每步由原生反馈确认。持瓶期间无需等待模型回复。仍只支持已声明的直立瓶子，持续夹持与实物标定未通过。',set_gripper:'独立开合任一夹爪，支持部分开口。验证指令与编码器稳定；闭合不等于夹住物体。放下持瓶时优先使用放置技能。',inspect_manipulation:'读取当前持瓶状态和放置区域，供 TAMP 判断操作条件。属于仿真几何反馈。',move_to_pose:'指定末端位置和朝向，由规划器检查可达性与碰撞。',move_relative:'从当前末端位姿移动，保持朝向。位移总长度最多 15 cm。',pick:'选择左手或右手，规划直立瓶子的抓取位姿，并验证实际夹持与抬升。',handover:'选择供物手和接收手，支持两个方向；接收手抓稳后才释放供物手。',place:'选择持瓶的手，放到指定桌面位置或篮子。模型先锁定任务起点的目标位置；换手和重规划不会改变目标。验证松手、支撑和稳定。',run_policy:'调用已配置的 OpenArm 策略。当前模型能力是完整交接任务。'};
skillHelp.inspect_scene='读取命名物体、当前位姿与候选几何。来源是仿真模型，不能当作相机检测或已标定的实物。';
skillHelp.resolve_scene_target='按物体原点、外形中心或顶部，以及物体／世界坐标偏移解析末端目标。只生成目标；运动工具仍须检查可达性、碰撞和实际位姿。篮子的顶部指外形边缘。';
skillHelp.resolve_placement_transit='根据当前持物与夹爪关系，计算原放置目标上方的末端位姿。只生成目标，不启动动作；规划运动并得到新反馈后，仍须执行放置和稳定检查。';
skillHelp.select_manipulation_object='任务开始先选择已注册对象，再观察和规划。选择只改变任务身份，不移动物体；夹爪接触物体或已开始动作后不能更换。';
function node(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;}

// Render a small, safe Markdown subset using text nodes, never model HTML.
export function richText(text){
  const root=node('div','rich-text');
  function inline(parent,line){
    for(const part of line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)){
      if(part.startsWith('**')&&part.endsWith('**'))parent.append(node('strong','',part.slice(2,-2)));
      else if(part.startsWith('`')&&part.endsWith('`'))parent.append(node('code','',part.slice(1,-1)));
      else parent.append(document.createTextNode(part));
    }
  }
  let list=null;
  for(const line of String(text).split('\n')){
    if(!line.trim()){list=null;continue;}
    const bullet=line.match(/^\s*(?:[-*]|\d+[.)])\s+(.+)$/);
    if(bullet){if(!list){list=node('ul');root.append(list);}const item=node('li');inline(item,bullet[1]);list.append(item);}
    else{list=null;const p=node('p');inline(p,line.replace(/^#{1,4}\s+/,''));root.append(p);}
  }
  return root;
}

export function createAgentUI({api,toast,message}){
  let revision='',fetching=false,modelBusy=false,submitting=false,local={turns:[],memory:{notes:[],episodes:[]}},planToken='',selectedBackend='tamp',skillCatalogue=null,memoryObject=null,skillsRequest=0;
  const cards=new Map();
  function role(){return $('agent-role').value;}
  function shortDetail(event){
    const d=event.detail,r=d.result||{};
    if(event.kind==='api')return `${d.model||''}${d.usage?.total_tokens?` · ${d.usage.total_tokens} tokens`:''}`;
    if(event.kind==='memory')return `${d.notes||0} 条偏好 · ${d.knowledge_items||0} 条关键经验`; 
    if(event.kind==='observation')return d.input||'读取中';
    if(d.result?.error)return r.error;
    if(event.name==='set_plan')return `${r.steps?.length||d.arguments?.steps?.length||0} 个阶段 · 模型生成`;
    if(event.name==='finish')return ({success:'模型判断：成功',failure:'模型判断：失败',uncertain:'模型判断：证据不足'}[r.reported_verdict]||'判断已记录')+(r.report_evidence?.decision_revalidation?.historical_evidence?' · 引用历史观测':'');
    if(event.name==='run_tamp')return d.arguments?.task||r.state||'';
    if(event.name==='run_pi05_vla')return 'OpenArm 原子策略';
    if(event.name==='relocate')return '连续搬运 · 每步验证后再继续';
    if(event.name==='inspect_scene')return `${r.objects?.length||0} 个命名参考 · 仿真几何`;
    if(event.name==='resolve_scene_target')return `${r.reference?.label||r.reference?.object_id||'物体'} · 目标已解析，尚未运动`;
    if(event.name==='resolve_placement_transit')return '当前持物 → 原放置目标上方 · 尚未运动';
    if(event.name==='select_manipulation_object')return `${r.object_id||'物体'} · 已选择，物理状态保留`;
    if(['pick','handover','place','relocate','move_relative','move_to_pose','set_gripper','run_policy'].includes(event.name))return '单项技能 · 结束后重新观察';
    if(event.name==='recall_memory')return r.title?'按需读取经验详情':`${r.knowledge?.index?.length||0} 条关键经验 · 不包含原始轨迹`; 
    if(event.kind==='execution')return '动作结束 · 重新观察与决策';
    return r.observation_id?'三路相机与本体状态':r.state||'';
  }
  function renderTrace(group,turn){
    const old=new Map([...group.querySelectorAll('.trace-item')].map(el=>[el.dataset.event,el]));
    for(const event of turn.trace){
      const key=String(event.id);let row=old.get(key);
      if(!row){row=node('details','trace-item');row.dataset.event=key;row.append(node('summary'),node('pre','trace-json'));group.append(row);}
      const label=names[event.name]||event.name;
      const icon=node('span',`trace-icon ${event.status}`,event.status==='completed'?'✓':event.status==='running'?'·':'!');
      const body=node('span','trace-description');body.append(node('strong','',label),node('small','',shortDetail(event)));
      const elapsed=event.ended?`${Math.max(0,event.ended-event.started).toFixed(1)}s`:'进行中';
      row.querySelector('summary').replaceChildren(icon,body,node('span','trace-time',elapsed));
      row.querySelector('pre').textContent=JSON.stringify(event.detail,null,2);
      row.dataset.status=event.status;
    }
  }
  function renderTurns(){
    const container=$('messages'),atBottom=container.scrollHeight-container.scrollTop-container.clientHeight<70;
    const seen=new Set();
    // A user goal is one conversation card. Automatic skill continuations
    // append to its public tool trace; they are not new human messages.
    const groups=new Map(),all=new Map(local.turns.map(t=>[t.id,t]));
    for(const item of local.turns){
      let root=item;const visited=new Set();
      while(root.parent&&all.has(root.parent)&&!visited.has(root.id)){
        visited.add(root.id);root=all.get(root.parent);
      }
      if(root.parent&&!all.has(root.parent))continue; // truncated archive child has no visible user-goal context
      if(!groups.has(root.id))groups.set(root.id,{...root,trace:[]});
      const group=groups.get(root.id);group.trace.push(...item.trace);
      group.answer=item.answer;group.status=item.status;
    }
    for(const turn of groups.values()){
      seen.add(turn.id);let card=cards.get(turn.id);
      if(!card){
        card=node('article','agent-turn');card.dataset.turn=turn.id;
        const user=node('div','turn-user');user.append(node('span','message-label','你'),node('p','turn-prompt'));
        const head=node('div','turn-heading');head.append(node('span','agent-avatar','◈'),node('strong','turn-model'),node('span','turn-status'));
        const trace=node('details','trace-group');trace.append(node('summary','trace-summary'),node('div','trace-list'));
        trace.querySelector('summary').addEventListener('click',()=>{trace.dataset.touched='1';});
        card.append(user,head,trace,node('div','turn-answer'));
        cards.set(turn.id,card);container.append(card);
      }
      card.querySelector('.turn-user').hidden=!!turn.review||!!turn.parent;
      card.querySelector('.turn-prompt').textContent=turn.prompt;
      card.querySelector('.turn-model').textContent=`${turn.model}${turn.review?' · 运行后复查':''}`;
      card.querySelector('.turn-status').textContent=statuses[turn.status]||turn.status;
      card.dataset.status=turn.status;
      const trace=card.querySelector('.trace-group'),count=turn.trace.filter(e=>e.kind==='api').length;
      const tools=turn.trace.filter(e=>e.kind==='tool').length;
      trace.querySelector('.trace-summary').textContent=`${turn.status==='running'?'正在处理':'调用记录'} · ${count} 次 API · ${tools} 次工具`;
      if(!trace.dataset.initialized){trace.open=turn.status==='running';trace.dataset.initialized='1';}
      if(turn.status!=='running'&&!trace.dataset.touched)trace.open=false;
      renderTrace(trace.querySelector('.trace-list'),turn);
      const answer=card.querySelector('.turn-answer');
      const answerText=turn.answer;
      if(answer.dataset.value!==answerText){
        answer.dataset.value=answerText;answer.replaceChildren();
        if(answerText.length>360){const full=node('details','answer-expand');full.append(node('summary','',answerText.split('\n')[0].slice(0,110).replace(/[*`]/g,'')+' · 展开详细说明'),richText(answerText));answer.append(full);}
        else answer.append(richText(answerText));
      }
    }
    for(const [id,card] of cards){if(!seen.has(id)){card.remove();cards.delete(id);}}
    $('memory-count').textContent=String(local.memory.notes.filter(n=>!n.stale).length+(local.memory.knowledge?.lessons.length||0)+(local.memory.knowledge?.recipes.length||0));
    if(atBottom)container.scrollTop=container.scrollHeight;
    if($('memory-dialog').open)renderMemory();
  }
  function renderMemory(){
    $('memory-role-label').textContent=role()==='execute'?'执行与观察':'实验与训练';
    $('memory-list').replaceChildren();
    const q=$('memory-search').value.trim().toLowerCase();
    const outcome=$('memory-outcome').value;
    const currentOnly=$('memory-context').value==='current';
    const knowledge=local.memory.knowledge||{lessons:[],recipes:[]};
    const bank=[...local.memory.notes.filter(n=>!n.stale).map(n=>({...n,kind:'note'})),...knowledge.lessons.map(e=>({...e,kind:'lesson',receipt:{outcome:e.outcome,evidence_kind:e.evidence_kind,feedback:{}}})),...knowledge.recipes.map(r=>({...r,kind:'recipe'}))];
    const items=bank.filter(item=>(!currentOnly||!memoryObject||item.kind==='note'||!item.conditions?.object_id||item.conditions.object_id===memoryObject)&&(!q||JSON.stringify(item).toLowerCase().includes(q))&&(outcome==='all'||(item.receipt?.outcome||(item.kind==='recipe'?'success':'unverified'))===outcome));
    $('memory-capabilities').replaceChildren();
    const protocols=[...new Set(items.filter(item=>item.kind==='lesson').map(item=>item.protocol_id).filter(Boolean))];
    for(const id of protocols){const text=knowledge.protocols?.[id];if(text){const label={relocate:'搬运共同规则',object_motion:'运动原则',object_manipulation:'操作原则'}[id]||'共同规则';$('memory-capabilities').append(node('strong','',label),node('p','',text));}}
    $('memory-working').replaceChildren();
    // Scene beliefs remain available to the harness as current context; this
    // panel shows durable methods and preferences only.
    $('memory-working').hidden=true;
    if(!items.length){$('memory-list').append(node('p','memory-empty',q||outcome!=='all'?'没有符合筛选条件的记忆。':'还没有保存的记忆。可靠的经验会在验证后提炼；单次试探和调用日志不进入记忆。也可以在下方添加偏好。'));return;}
    for(const item of items){
      const card=node('article','memory-item');
      const header=node('div','memory-heading');
      const label=item.kind==='lesson'?(item.outcome==='success'?(item.evidence_kind==='simulator_oracle'?'仿真中验证的方法':'已验证的方法'):'失败教训'):item.kind==='recipe'?'可复用技能配方 · 仿真验证':['episode','experience','lesson'].includes(item.kind)?outcomes[item.receipt?.outcome]||'执行经历':item.scope==='preference'?'你的偏好':item.stale?'场景已变 · 笔记过期':'模型场景笔记 · 待验证';
      header.append(node('span','',label));
      const remove=node('button','quiet','删除');remove.type='button';remove.addEventListener('click',async()=>{try{await api('/api/memory/delete',{role:role(),id:item.id});await refresh(true);}catch(e){toast(e.message);}});
      if(item.kind==='note'||item.kind==='lesson')header.append(remove);card.append(header,node('p','',item.kind==='lesson'?item.title:item.kind==='recipe'?'抓取 → 双手交接 → 放入容器':item.text||item.goal));
      if(item.kind==='lesson'){
        card.append(node('p','',item.summary),node('small','',item.support));
        const details=node('details','memory-evidence');details.append(node('summary','','适用范围与实验依据'));
        const c=item.conditions||{}, labels=[];
        if(c.object_id)labels.push(({bottle:'瓶子',block:'长方体',can:'圆柱'})[c.object_id]||c.object_id);
        if(c.arm)labels.push(c.arm==='left'?'左手':'右手');
        if(c.from_arm&&c.to_arm)labels.push(`${c.from_arm==='left'?'左手':'右手'} → ${c.to_arm==='left'?'左手':'右手'}`);
        if(c.payload_state)labels.push(({empty:'空手',single_hold:'单手持物',dual_hold:'双手持物'})[c.payload_state]||c.payload_state);
        if(c.start_orientation==='non_upright')labels.push('倒卧／倾斜起点');
        if(c.next_handover)labels.push('为后续交接选择抓取');
        if(c.grasp_intent==='table_placement_lookahead')labels.push('为原桌面目标选择抓取');
        if(c.requested_hold_seconds)labels.push(`持稳请求 ${c.requested_hold_seconds} 秒`);
        if(c.hold_validation)labels.push(c.hold_validation==='all_full_windows'?'连续窗口审核':'末段窗口审核');
        if(c.surface)labels.push(c.surface==='table'?'桌面放置':'篮内放置');
        if(labels.length)details.append(node('p','',`适用条件：${labels.join(' · ')}`));
        details.append(node('p','',item.evidence_scope||'仅覆盖已测条件。'));
        const timing=Object.entries(item.timing_conditions||{}).map(([mode,count])=>`${mode==='measured_delay'?'含模型等待':mode==='paused'?'规划时暂停':mode} ${count} 次`).join(' · ');
        if(timing)details.append(node('small','',timing));
        const planning=Object.entries(item.planner_timing_conditions||{}).map(([condition,count])=>`${condition==='legacy_unmodeled'?'旧记录未计底层规划':condition.endsWith(':measured_delay')?'已计底层规划等待':'底层规划时暂停'} ${count} 次`).join(' · ');
        if(planning)details.append(node('p','',planning));
        for(const source of item.sources||[]){const frame=source.evidence?.camera_keyframes?.[0];const url=source.run_id?'/runs/'+source.run_id+'/summary.json':frame?.sensor_urls?.head;if(url&&/^\/runs\/[A-Za-z0-9_./-]+$/.test(url)){const link=node('a','',source.run_id||'相机证据');link.href=url;link.target='_blank';link.rel='noopener';const row=node('p','');row.append(link);details.append(row);}}
        card.append(details);
      }
      if(item.kind==='episode')card.append(node('small','',`${names[item.tool]||item.tool} · ${item.model} · ${item.verdict?`模型判断 ${item.verdict}`:item.ended?'已执行，尚无模型判断':'未记录结束'}${item.run_id?' · '+item.run_id:''}`));
      else if(item.kind==='experience')card.append(node('small','',`${names[item.tool]||item.tool} · ${item.receipt.evidence_kind==='contract_guard'?'动作未启动':'已记录执行反馈'}`));
      else if(item.kind==='recipe')card.append(node('small','',`${(item.body?.steps||[]).map(s=>names[s.skill]||s.skill).join(' → ')} · 复用时重新定位并规划`));
      else if(item.kind==='note')card.append(node('small','',item.scope==='scene'?'仅在同一次场景内供模型参考；重置后过期。':'后续对话会读取，可随时修改或删除。'));
      if(item.receipt&&item.kind!=='lesson'){const f=item.receipt.feedback;card.append(node('p','memory-feedback',`${item.receipt.evidence_kind==='simulator_oracle'?'仿真特权反馈':item.receipt.evidence_kind==='contract_guard'?'工具约束反馈':'执行反馈'} · ${f.failure_code||f.status||'已记录'}${f.message?' · '+f.message:''}`));const e=item.receipt.evidence||{};if(e.before?.observation_id&&e.after?.observation_id)card.append(node('small','',`观测 ${e.before.observation_id} → ${e.after.observation_id}`));if(e.camera_keyframes?.length){const details=node('details','memory-evidence');details.append(node('summary','','查看操作前后相机证据'));for(const [index,frame] of e.camera_keyframes.entries()){const links=node('p','');links.append(document.createTextNode(index===0?'操作前：':'操作后：'));for(const camera of ['head','left','right']){const url=frame.sensor_urls?.[camera];if(url&&/^\/runs\/[A-Za-z0-9_./-]+$/.test(url)){const a=node('a','',({head:'头部',left:'左腕',right:'右腕'})[camera]);a.href=url;a.target='_blank';a.rel='noopener';links.append(a);}}details.append(links);}card.append(details);}if(item.receipt.compatible===false)card.append(node('small','memory-stale','其他版本的历史记录，不供当前任务检索。'));}
      if(item.kind==='note'&&item.scope==='preference'){
        const edit=node('button','memory-edit','修改');edit.type='button';edit.addEventListener('click',()=>{$('memory-note').value=item.text;$('memory-note').dataset.replaces=item.id;$('memory-note').focus();});card.append(edit);
      }
      $('memory-list').append(card);
    }
  }
  async function refresh(force=false){
    if(fetching)return;fetching=true;const selected=role();
    try{const data=await api('/api/agent?role='+selected);if(selected!==role())return;local=data;revision=data.revision;renderTurns();if($('memory-dialog').open)renderMemory();}
    catch(e){if(force)toast(e.message);}finally{fetching=false;}
  }
  $('chat-form').addEventListener('submit',async event=>{
    event.preventDefault();const text=$('chat-input').value.trim();if(!text||submitting||modelBusy)return;
    submitting=true;$('send-button').disabled=true;$('agent-role').disabled=true;$('chat-input').value='';
    $('agent-activity').textContent='正在提交给模型…';
    try{const result=await api('/api/chat',{message:text,role:role()});if(result.mode==='not_configured'){if(!$('chat-input').value)$('chat-input').value=text;message(result.message,'note');}}
    catch(e){if(!$('chat-input').value)$('chat-input').value=text;toast(e.message);}finally{submitting=false;await refresh(true);$('send-button').disabled=modelBusy;$('agent-role').disabled=modelBusy;}
  });
  $('agent-role').addEventListener('change',()=>{revision='';$('memory-note').value='';delete $('memory-note').dataset.replaces;for(const card of cards.values())card.remove();cards.clear();refresh(true);});
  $('memory-button').addEventListener('click',async()=>{renderMemory();$('memory-dialog').showModal();await refresh(true);if($('memory-dialog').open)renderMemory();});
  $('memory-search').addEventListener('input',renderMemory);
  $('memory-outcome').addEventListener('change',renderMemory);
  $('memory-context').addEventListener('change',renderMemory);
  function selectBackend(id){
    selectedBackend=id;const data=skillCatalogue,backend=data.adapters.find(b=>b.id===id);
    for(const b of $('backend-list').children){const selected=b.dataset.backend===id;b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1;}
    $('backend-panel').setAttribute('aria-labelledby','backend-'+id);
    $('backend-title').textContent=id==='pi05'?'π0.5':id==='curobo'?'cuRobo':id.toUpperCase();
    const scope={tamp:'左右手抓取、双向交接、桌面定点放置、入篮、夹爪开合与末端位姿规划。已知仿真几何；保留交接演示，真机未连接。',curobo:'空手或所选手独立持瓶时，规划末端位置与朝向；GPU 求解后复核原生碰撞几何与接触。双手／另一手持物尚未开放，抓取和放置由 TAMP 完成。',pi05:'调用匹配的 OpenArm π0.5 策略；技能范围由权重声明。',act:'调用匹配的 OpenArm ACT 策略；从图像与关节状态生成动作块。'};
    const objectMode=Boolean(data.object_manipulation_scope);
    if(objectMode)scope.tamp='选择已注册的直立瓶子、长方体或圆柱，规划抓取与目标放置，并验证实际接触、松手和稳定。新形状的交接尚未开放；属于未标定的开发场景。';
    if(data.object_relay_evidence)scope.tamp='选择已注册的直立物体，规划抓取、双臂交接与目标放置。交接搜索双手姿态和接近路径，确认接收手抓稳后才松开。已测长方体左→右、圆柱右→左；仍是未标定的开发场景。';
    if(data.multi_object_tasks)scope.tamp+=' 支持在同一任务中顺序操作多个物体：先锁定各自目标，放下并稳定后再选择下一对象。所有动作共用原时间预算。';
    if(data.oriented_grasp_scope)scope.tamp+=' 倒卧／倾斜抓取仍为实验功能；倒卧长方体和瓶子已完成仿真开发任务，倾斜圆柱尚未通过。';
    if(data.bottle_directed_scope)scope.tamp+=' 瓶子左→右交接会搜索障碍物绕行，并让空出的供物手退出后续运输区域；目前只验证了固定起点的仿真任务。';
    if(data.object_gpu_scope)scope.curobo='空手或所选手独立持瓶子、长方体、圆柱时，规划末端位置与朝向。携物外形进入GPU碰撞检查，再复核原生路径与实际接触。双手／另一手持物尚未开放；抓取和放置由TAMP完成。';
    const objectHelp=objectMode?{pick:'抓取当前所选的直立瓶子、长方体或圆柱。先检查几何、开口和可达性，再验证实际双指接触与抬升。',place:'将当前持物的手放到锁定的原目标。验证整个物体外形、支撑、松手、撤离与连续稳定。',inspect_manipulation:'检查当前所选对象的持握、支撑区域与近期滑动，来源是仿真几何。'}:{};
    if(data.object_relay_evidence)objectHelp.handover='从当前物体外形搜索供物手朝向、接收抓取和双臂路径，再验证实际交接与独立持物。有后续桌面目标时会检查放置姿态；预测可行仍须实际执行验证。圆柱交接前的抓取会考虑接收手所需空间。';
    if(data.bottle_directed_scope)objectHelp.handover+=' 瓶子左→右会先规划避让，再让空手退开。API 可要求交接后独立持物 2–10 秒，每个完整稳定窗口均需通过。';
    if(data.multi_object_tasks)objectHelp.select_manipulation_object='任务开始选择已注册对象。多物体计划中，只有当前物体按原目标放下、松手并稳定后，才能选下一声明对象。选择不移动物体，也不重置时间、预算或目标。';
    if(data.oriented_grasp_scope)objectHelp.pick='根据当前注册物体的外形与姿态选择抓取点和接近方向，规划路径后核验双指接触与实际抬升。倒卧／倾斜对象仍是实验范围；有原桌面目标时会预检查后续放置，预测可行仍须实际执行验证。';
    $('skills-scope').textContent=scope[id]+(backend.validation_note?' '+backend.validation_note:'')+(!backend.ready?' 当前未接入：'+backend.reason:'');
    $('skills-list').replaceChildren();
    const skills=data.skills.filter(s=>(s.backends||(['move_relative','move_to_pose'].includes(s.id)?['tamp','curobo']:s.id==='run_policy'?['pi05','act']:['tamp'])).includes(id));
    for(const skill of skills.filter(s=>s.kind!=='scene_context')){const item=node('details','skill-item');item.append(node('summary','',names[skill.id]||skill.id),node('p','muted',objectHelp[skill.id]||skillHelp[skill.id]||skill.description));$('skills-list').append(item);}
    const contextSkills=skills.filter(s=>s.kind==='scene_context');
    if(contextSkills.length){const item=node('details','skill-item');item.append(node('summary','','场景与目标'));for(const skill of contextSkills){const paragraph=node('p','muted');paragraph.append(node('strong','',names[skill.id]||skill.id),document.createTextNode(' · '+(objectHelp[skill.id]||skillHelp[skill.id]||skill.description)));item.append(paragraph);}$('skills-list').append(item);}
    $('tamp-demo').hidden=id!=='tamp';$('motion-form').hidden=!['tamp','curobo'].includes(id);$('motion-run').disabled=!backend.ready;$('motion-result').textContent='';
  }
  $('skills-button').addEventListener('click',async()=>{
    const request=++skillsRequest;
    $('skills-dialog').showModal();
    $('backend-list').replaceChildren();$('skills-list').replaceChildren();
    $('skills-scope').textContent='正在加载技能…';$('tamp-demo').hidden=true;$('motion-form').hidden=true;
    try{
      const data=await api('/api/skills');if(request!==skillsRequest)return;$('backend-list').replaceChildren();$('skills-list').replaceChildren();
      skillCatalogue=data;
      for(const backend of data.adapters){const ready=backend.callable??backend.ready;const item=node('button',`backend ${ready?'ready':'unavailable'}`);item.type='button';item.id='backend-'+backend.id;item.dataset.backend=backend.id;item.setAttribute('role','tab');item.setAttribute('aria-controls','backend-panel');item.append(node('strong','',backend.id==='pi05'?'π0.5':backend.id==='curobo'?'cuRobo':backend.id.toUpperCase()),node('small','',ready?'可调用':'未接入'));item.title=backend.reason||backend.validation_note||(backend.performance_verified?'任务效果已验证':'接口可用不代表任务效果已验证');item.addEventListener('click',()=>selectBackend(backend.id));item.addEventListener('keydown',e=>{const ids=data.adapters.map(b=>b.id),index=ids.indexOf(backend.id);const next=e.key==='ArrowRight'?ids[(index+1)%ids.length]:e.key==='ArrowLeft'?ids[(index+ids.length-1)%ids.length]:e.key==='Home'?ids[0]:e.key==='End'?ids.at(-1):null;if(next){e.preventDefault();selectBackend(next);$('backend-'+next).focus();}});$('backend-list').append(item);}
      selectBackend(selectedBackend);
    }catch(e){if(request===skillsRequest){$('skills-scope').textContent='加载失败，请关闭后重试。';toast(e.message);}}
  });
  $('skills-dialog').addEventListener('close',()=>{skillsRequest++;});
  $('motion-form').addEventListener('submit',async event=>{
    event.preventDefault();const button=$('motion-run');button.disabled=true;
    try{const delta=['motion-x','motion-y','motion-z'].map(id=>Number($(id).value)/100);if(!delta.every(Number.isFinite)||Math.hypot(...delta)>.15)throw new Error('位移总长度须在 15 cm 内');const obs=await api('/api/tools',{name:'observe',arguments:{}});const result=await api('/api/tools',{name:'move_relative',arguments:{observation_id:obs.observation_id,arm:$('motion-arm').value,delta_m:delta,frame:'world',speed_scale:.5,planner:selectedBackend}});$('motion-result').textContent='已提交规划 · 执行反馈见场景与运行阶段';message('已提交末端相对移动，执行编号 '+result.execution_id,'note');}
    catch(e){$('motion-result').textContent=e.message;}finally{button.disabled=!skillCatalogue.adapters.find(b=>b.id===selectedBackend).ready;}
  });
  $('memory-form').addEventListener('submit',async event=>{
    event.preventDefault();const text=$('memory-note').value.trim();if(!text)return;
    try{
      const id=$('memory-note').dataset.replaces;
      await api('/api/memory',{role:role(),text,...(id?{id}:{})});
      $('memory-note').value='';delete $('memory-note').dataset.replaces;await refresh(true);
    }catch(e){toast(e.message);}
  });
  $('memory-clear').addEventListener('click',async()=>{try{await api('/api/memory/delete',{role:role()});await refresh(true);toast('当前职责的记忆已清除；实验原始记录保留');}catch(e){toast(e.message);}});
  return {
    render(data){
      const selected=data.diagnostics?.manipulation_scene?.object_id||data.agent?.plan?.placement_goal?.object_id||null;
      if(selected!==memoryObject){memoryObject=selected;$('memory-context').options[0].textContent=selected?`当前对象：${({bottle:'瓶子',block:'长方体',can:'圆柱'})[selected]||selected}`:'当前任务';if($('memory-dialog').open)renderMemory();}
      modelBusy=!!data.agent?.busy;
      $('send-button').disabled=submitting||modelBusy;$('agent-role').disabled=submitting||modelBusy;
      $('agent-activity').textContent=submitting&&!modelBusy?'正在提交给模型…':modelBusy?'模型正在处理 · 调用过程见下方':data.llm?`${data.llm_model} · API 已配置`:'尚未接入 API';
      $('agent-activity').classList.toggle('active',submitting||modelBusy);
      if(data.agent&&data.agent.revision!==revision)refresh();
      const plan=data.agent?.plan,token=JSON.stringify(plan);
      if(token!==planToken){planToken=token;if(plan){
        $('stage-list').style.gridTemplateColumns=`repeat(${plan.steps.length},minmax(82px,1fr))`;
        $('stage-list').replaceChildren(...plan.steps.map((step,i)=>{
          const li=node('li',step.status);li.title=checks[step.checkpoint]||step.checkpoint;
          li.append(node('span','stage-dot',marks[step.status]||String(i+1)),node('span','stage-title',step.title.replace(/\brelocate\b/g,'连续搬运')));return li;
        }));
      }}
      const run=data.job||data.last_run;
      $('stage-source').textContent=plan?`模型计划 · ${plan.model}${plan.timing_mode==='measured_delay'?' · 计入等待时间':''}${plan.object_goals?` · ${plan.object_goals.length}个目标已锁定`:plan.placement_goal?" · 目标已锁定":""}${plan.time_budget_s>40?` · ${plan.time_budget_s}秒仿真预算`:''}`:run?.kind==='replay'?'回放 · 未调用 API':run&&run.initiator?.type!=='llm'?'技能测试 · 未调用 API':'等待模型计划';
      $('stage-source').title=$('stage-source').textContent;
      if(plan){const done=plan.steps.filter(s=>s.status==='done').length;$('stage-count').textContent=`${done} / ${plan.steps.length}`;$('progress-bar').style.width=`${100*done/plan.steps.length}%`;}
      else if(!run){
        $('stage-list').style.gridTemplateColumns='1fr';$('stage-list').replaceChildren(node('li','plan-placeholder',modelBusy?'模型正在处理请求…':'发送一个任务，模型会在这里制定运行阶段。'));$('stage-count').textContent='待规划';
      }else $('stage-list').style.gridTemplateColumns='';
      return !!plan||!run;
    }
  };
}
