// Shared read-only presentation of task and execution facts.
export const toolNames = {set_gripper:'夹爪开合',observe:'观察场景',inspect_manipulation:'检查操作条件',prepare_simulation:'准备新仿真任务',set_plan:'制定阶段计划',pick:'抓取',handover:'双手交接',place:'放置',relocate:'连续搬到目标',move_to_pose:'移动到目标位姿',move_relative:'末端相对移动',run_policy:'调用学习策略',list_skills:'读取技能能力',run_tamp:'TAMP 基线',run_pi05_vla:'VLA 基线',get_run_status:'读取执行反馈',finish:'提交模型复查',recall_memory:'检索相关经验',remember_scene:'更新场景记忆'};
toolNames.inspect_scene='读取场景目录';toolNames.resolve_scene_target='解析相对物体目标';
toolNames.resolve_placement_transit='计算运输位姿';
toolNames.select_manipulation_object='选择操作对象';

export function duration(value) {
  if (!Number.isFinite(value) || value < 0) return '—';
  if (value < 60) return `${value.toFixed(value < 10 ? 1 : 0)} s`;
  return `${Math.floor(value / 60)}分 ${Math.floor(value % 60)}秒`;
}

// A call returning "started" only accepted an asynchronous job. Use its
// independently recorded receipt for the actual skill outcome, including history.
export function traceState(event) {
  const detail=event.detail || {}, result=detail.result || {}, execution=detail.execution;
  const value=(status,tone,label,description='',icon='pending')=>({status,tone,label,description,icon});
  if (execution) {
    const feedback=execution.feedback || {}, status=execution.status;
    if (status==='cancelled') return value('cancelled','neutral','已停止','','stop');
    if (status==='interrupted') return value('interrupted','neutral','已中断','','stop');
    if (['failed','rejected','failure'].includes(status) || feedback.failure_code || feedback.primitive_success===false) {
      const reason=feedback.failure_code==='placement_unreachable'?'未找到满足姿态约束的可达放置方案':feedback.message || feedback.failure_code || '执行未通过验证';
      return value('error','negative','执行失败',reason+(execution.native_commands===0?' · 未下发新动作':''),'error');
    }
    if (status==='succeeded' && feedback.primitive_success===true && ['simulator_oracle','sensor_execution'].includes(execution.evidence_kind))
      return value('completed','positive','技能完成','原生执行反馈已确认','check');
    if (status==='accepted' || status==='running') return value('running','active','正在规划／执行','已启动，等待执行反馈','spinner');
    return value('unverified','neutral','已结束，结果未验证','','pending');
  }
  if (result.state==='started' && result.execution_id)
    return value('unverified','neutral','已启动，结果未验证','尚无绑定的完成反馈','pending');
  if (event.status==='running') return value('running','active','调用中','','spinner');
  if (event.status==='error' || result.tool_failed) return value('error','negative','调用失败',result.message || result.error || '','error');
  // Model self-assessment and generic tool completion stay neutral.
  return value(event.status,'neutral',event.kind==='execution'?'执行已结束，结果未验证':'调用已返回','','pending');
}

export function outcome(status) {
  const labels = {success:'任务成功',primitive_completed:'技能完成',completed:'已结束',
    running:'运行中',starting:'启动中',waiting:'等待中',stopping:'正在停止',cancelled:'已停止',
    error:'失败',failure:'失败',invalid:'物理异常',interrupted:'已中断',
    budget_exhausted:'预算耗尽',reference_exhausted:'轨迹结束',truncated:'达到时限',
    incomplete:'缺少结果',unknown:'结果未知',unverified:'未验证',rejected:'已拒绝',captured:'已留帧'};
  const tone = ['success','primitive_completed'].includes(status) ? 'positive' :
    ['error','failure','invalid','rejected'].includes(status) ? 'negative' :
    ['running','waiting','starting','stopping'].includes(status) ? 'active' : 'neutral';
  const terminal = ['success','primitive_completed','completed','cancelled','error','failure',
    'invalid','interrupted','budget_exhausted','reference_exhausted','truncated','rejected','captured'].includes(status);
  return {label:labels[status] || '结果未知', tone, terminal};
}

export function runTitle(record) {
  const kinds = {tamp:'运动规划',policy:'策略执行',replay:'轨迹回放',snapshot:'手动留帧',
    train:'策略训练',evaluate:'策略评估',export:'数据导出',calibrate:'参数辨识',wrist:'相机拟合'};
  const title = toolNames[record.task] || kinds[record.kind] || record.kind || '执行记录';
  const arm = {left:'左臂',right:'右臂'}[record.parameters?.arm];
  return title + (arm ? ` · ${arm}` : '');
}

export function movement(parameters) {
  if (!Array.isArray(parameters?.delta_m)) return '';
  return parameters.delta_m.map((v,i) => Number.isFinite(v) && v !== 0 ?
    `${'XYZ'[i]} ${v > 0 ? '+' : ''}${Number((v * 100).toFixed(2))} cm` : '').filter(Boolean).join(' · ');
}

const artifacts = new Set(['manifest.json','summary.json','events.jsonl','transitions.jsonl','job.json']);
export function archiveURL(runId, filename) {
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(runId || '') || !artifacts.has(filename)) return null;
  return `/runs/${encodeURIComponent(runId)}/${filename}`;
}
export function mediaURL(url, runId) {
  const base = archiveURL(runId, 'manifest.json');
  if (typeof url !== 'string' || !base || !url.startsWith(base.slice(0, -13) + 'keyframes/')) return null;
  if (!/^\/runs\/[A-Za-z0-9_.-]+\/keyframes\/[A-Za-z0-9_-]+\.(jpg|png)$/i.test(url)) return null;
  return url;
}

export function groupTurns(turns = []) {
  const ordered = [...turns].sort((a,b) => (a.created || 0) - (b.created || 0));
  const all = new Map(ordered.map(t => [t.id,t])), groups = new Map();
  for (const item of ordered) {
    let root = item, truncated = false;
    const visited = new Set();
    while (root.parent) {
      if (visited.has(root.id)) { truncated = true; break; }
      visited.add(root.id);
      if (!all.has(root.parent)) { truncated = true; break; }
      root = all.get(root.parent);
    }
    const key = truncated ? `partial:${[...visited,root.id].sort()[0]}` : root.id;
    if (!groups.has(key)) groups.set(key, {...root, id:key, trace:[], members:[], truncated});
    const group = groups.get(key);
    group.members.push(item);
    group.trace.push(...(item.trace || []).map(event => ({...event, turn_id:item.id})));
    if (item.answer) group.answer = item.answer;
    group.status = truncated ? 'unknown' : item.status;
  }
  for (const group of groups.values()) {
    group.trace.sort((a,b) => (a.started || 0) - (b.started || 0));
    const seen = new Set();
    group.trace = group.trace.filter((event,index) => {
      const key = `${event.turn_id}:${event.id ?? index}`;
      if (seen.has(key)) return false;
      seen.add(key); return true;
    });
    group.runIds = [...new Set(group.trace.flatMap(event => {
      const detail = event.detail || {}, result = detail.result || {};
      return [detail.run_id,result.run_id,result.evidence?.run_id,detail.execution?.run_id].filter(id => archiveURL(id,'manifest.json'));
    }))];
    const finish=[...group.trace].reverse().find(event=>event.kind==='tool' && event.name==='finish');
    if (group.status==='completed' && finish?.detail?.result?.reported_verdict==='failure') group.status='failure';
  }
  return [...groups.values()];
}
