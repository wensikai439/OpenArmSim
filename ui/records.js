import {groupTurns, toolNames, outcome, traceState, runTitle, movement, duration, archiveURL, mediaURL} from './activity.js?v=execution-1';
import {richText} from './agent.js?v=execution-1';

const node = (tag, cls, text) => {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
};
function stamp(seconds, runId = '') {
  let date = Number.isFinite(seconds) ? new Date(seconds * 1000) : null;
  const match = runId.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})-/);
  if (!date && match) date = new Date(`${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${match[6]}`);
  return date && Number.isFinite(date.getTime()) ? new Intl.DateTimeFormat('zh-CN', {
    month:'numeric', day:'numeric', hour:'2-digit', minute:'2-digit'}).format(date) : '时间未记录';
}
function badge(status) {
  const value = outcome(status);
  return node('span', `record-outcome ${value.tone}`, value.label);
}
function rawDetails(value, title = '原始数据') {
  const details = node('details', 'record-raw');
  details.append(node('summary', '', title), node('pre', '', JSON.stringify(value, null, 2)));
  return details;
}
function fileLink(runId, filename, title) {
  const url = archiveURL(runId, filename);
  if (!url) return null;
  const link = node('a', 'record-file', `${title} ↗`);
  link.href = url; link.target = '_blank'; link.rel = 'noopener';
  return link;
}
async function optionalJSON(url) {
  if (!url) return null;
  const response = await fetch(url);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`读取记录失败 (${response.status})`);
  return response.json();
}
async function eventLog(runId) {
  const response = await fetch(archiveURL(runId, 'events.jsonl'));
  if (response.status === 404) return {events:[], available:false};
  if (!response.ok) throw new Error(`读取事件失败 (${response.status})`);
  const reader = response.body.getReader(), parts = [];
  let bytes = 0;
  try {
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      bytes += value.length;
      if (bytes > 2 * 1024 * 1024) {
        await reader.cancel();
        return {events:[], available:true, warning:'事件文件较大，请打开原始事件文件查看。'};
      }
      parts.push(value);
    }
  } finally { reader.releaseLock(); }
  const buffer = new Uint8Array(bytes);
  let offset = 0;
  for (const part of parts) { buffer.set(part, offset); offset += part.length; }
  const events = []; let invalid = 0;
  for (const line of new TextDecoder().decode(buffer).split('\n').filter(line => line.trim())) {
    try {
      const event = JSON.parse(line);
      if (event && typeof event === 'object' && !Array.isArray(event)) events.push(event);
      else invalid++;
    } catch { invalid++; }
  }
  return {events, available:true, warning:invalid ? `${invalid} 条事件无法解析，原文件仍可查看。` : ''};
}

// Reads existing archives and the current owner's journal. No execution requests.
export function createRecordsUI({api, getRole, onImage}) {
  const $ = id => document.getElementById(id);
  const dialog = $('records-dialog');
  let entries = [], selected = '', fetching = false, generation = 0, detailRequest = 0;
  let sourceToken = '', reloadTimer, pendingReload = false;
  const cache = new Map();

  function empty(target, text) { target.replaceChildren(node('p', 'record-empty', text)); }
  function notice(text) { $('records-notice').textContent = text; $('records-notice').hidden = !text; }
  function visibleEntries() {
    const query = $('records-search').value.trim().toLowerCase(), filter = $('records-filter').value;
    return entries.filter(item => !item.detailOnly).filter(item => {
      const status = outcome(item.status);
      const matches = filter === 'all' || (filter === 'ended' ? status.terminal : status.tone === filter);
      return matches && `${item.title} ${item.run_id || ''} ${item.kind || ''} ${status.label} ${movement(item.parameters)}`.toLowerCase().includes(query);
    });
  }
  function renderList(refresh = false) {
    const rows = visibleEntries(), list = $('records-list');
    list.replaceChildren();
    $('records-count').textContent = `最近 ${entries.filter(item => !item.detailOnly).length} 条`;
    if (!rows.length) { selected = ''; detailRequest++; empty(list, entries.length ? '没有匹配的记录' : '暂无实验记录'); empty($('record-detail'), '选择一条记录查看'); return; }
    for (const item of rows) {
      const wrap = node('div', 'record-list-item'); wrap.setAttribute('role', 'listitem');
      const button = node('button', 'record-row'); button.type = 'button'; button.dataset.record = item.id;
      button.setAttribute('aria-current', String(item.id === selected));
      button.tabIndex = item.id === selected ? 0 : -1;
      const icon = node('span', `record-row-icon ${outcome(item.status).tone}`, item.type === 'task' ? '↳' : '◇');
      const body = node('span', 'record-row-body');
      body.append(node('strong', '', item.title), node('span', 'record-row-meta',
        `${stamp(item.started_at, item.run_id)} · ${outcome(item.status).label}`));
      button.append(icon, body);
      button.addEventListener('click', () => select(item.id)); wrap.append(button); list.append(wrap);
    }
    if (!rows.some(item => item.id === selected)) select(rows[0].id);
    else if (refresh) select(selected);
  }
  function heading(item, subtitle) {
    const head = node('header', 'record-detail-heading'), title = node('div');
    title.append(node('h3', '', item.title), node('p', '', subtitle));
    head.append(title, badge(item.status)); return head;
  }
  function metricRow(items) {
    const row = node('dl', 'record-metrics');
    for (const [label, value] of items) {
      const cell = node('div'); cell.append(node('dt', '', label), node('dd', '', value)); row.append(cell);
    }
    return row;
  }
  function traceList(trace) {
    const list = node('ol', 'record-timeline');
    // API/memory bookkeeping stays in raw data; the first view follows actions.
    const actions = trace.filter(event => ['tool','execution'].includes(event.kind) || ['error','interrupted'].includes(event.status));
    for (const event of actions) {
      const detail = event.detail || {}, result = detail.result || {};
      const row = node('li'), block = node('details', 'record-action'), summary = node('summary');
      const name = toolNames[event.name] || (event.kind === 'execution' ? '执行反馈' : event.name);
      const state = traceState(event);
      summary.append(node('span', `record-action-dot ${state.tone}`), node('strong', '', name || '工具调用'),
        node('span', 'record-action-time', Number.isFinite(detail.execution?.wall_seconds) ? duration(detail.execution.wall_seconds) : event.ended ? duration(Math.max(0,event.ended - event.started)) : state.label));
      block.append(summary);
      const description = detail.execution || result.state==='started' ? [state.label,state.description].filter(Boolean).join(' · ') : result.error || result.feedback?.message || movement(detail.arguments) ||
        (result.reported_verdict ? `模型复查：${({success:'成功',failure:'失败',uncertain:'证据不足'})[result.reported_verdict] || result.reported_verdict}` : '');
      if (description) block.append(node('p', 'record-action-note', String(description)));
      block.append(node('pre', '', JSON.stringify(detail,null,2)));
      row.append(block); list.append(row);
    }
    if (!actions.length) list.append(node('li', 'record-empty', '尚无动作调用'));
    return list;
  }
  async function loadArchive(item) {
    const stored = cache.get(item.run_id);
    if (stored && !['active','unknown'].includes(outcome(item.status).tone) && stored.summary) return stored;
    const [manifest,summary,log] = await Promise.all([
      optionalJSON(archiveURL(item.run_id,'manifest.json')),
      optionalJSON(archiveURL(item.run_id,'summary.json')), eventLog(item.run_id),
    ]);
    if (manifest?.run_id && manifest.run_id !== item.run_id) throw new Error('记录标识不匹配，未合并数据');
    const value = {manifest,summary,...log};
    if (cache.size >= 40) cache.delete(cache.keys().next().value);
    cache.set(item.run_id,value); return value;
  }
  function renderNative(item, value) {
    const {manifest, summary, events} = value, target = $('record-detail');
    const native = manifest?.details?.tamp || {}, measured = summary?.metrics?.tamp || {};
    const parameters = native.parameters || item.parameters;
    const actual = {...item, task:native.task || measured.task || measured.skill || item.task, parameters,
      status:summary?.status || manifest?.status || item.status};
    actual.title = runTitle(actual);
    const planner = parameters?.planner === 'curobo' ? 'cuRobo' :
      ({tamp:'TAMP',replay:'轨迹回放',snapshot:'手动留帧',policy:'策略执行'})[manifest?.kind || item.kind] || item.kind || '';
    target.replaceChildren(heading(actual, `${stamp(manifest?.started_at_unix || item.started_at, item.run_id)} · ${planner}`));
    target.append(metricRow([['执行耗时',duration(summary?.wall_seconds)],
      ['仿真时长',duration(summary?.duration_sim_seconds)],['控制步数',Number.isFinite(summary?.transitions) ? String(summary.transitions) : '—']]));
    if (movement(parameters)) target.append(node('p', 'record-intent', movement(parameters)));
    if (summary?.error) target.append(node('p','record-error',String(summary.error)));
    if (!summary) target.append(node('p','record-inline-note','结果文件暂不可用'));
    if (value.warning) target.append(node('p','record-inline-note',value.warning));
    const frames = events.filter(event => event.type === 'keyframe');
    if (frames.length) {
      target.append(node('h4','record-section-title','关键帧'));
      const strip = node('div','record-frames'); let rejected = 0;
      for (const frame of frames) {
        const url = mediaURL(frame.url,item.run_id);
        if (!url) { rejected++; continue; }
        const label = frame.label?.startsWith('结束 · ') ? outcome(actual.status).label : frame.label || '关键帧';
        const button = node('button','record-frame'); button.type = 'button';
        const image = node('img'); image.src = url; image.alt = label; image.loading = 'lazy';
        button.append(image,node('span','',label));
        button.addEventListener('click',() => onImage(url,label,`${duration(frame.sim_time)} · ${frame.camera || ''}`));
        strip.append(button);
      }
      target.append(strip);
      if (rejected) target.append(node('p','record-inline-note',`${rejected} 张图像的路径与本次记录不匹配，未展示。`));
    }
    const milestones = summary?.events || [];
    if (milestones.length > 1) {
      target.append(node('h4','record-section-title','执行节点'));
      const list = node('ol','record-milestones');
      for (const event of milestones) list.append(node('li','',`${event.name || '已记录节点'} · ${duration(event.time)}`));
      target.append(list);
    }
    const evidence = native.scene_source === 'simulator_ground_truth' ? '仿真几何反馈' : native.verification_source;
    if (evidence) target.append(node('p','record-inline-note',evidence));
    const files = node('div','record-files');
    for (const [available,name,label] of [[manifest,'manifest.json','运行配置'],[summary,'summary.json','结果数据'],[value.available,'events.jsonl','事件记录'],[manifest?.format?.includes('native transitions'),'transitions.jsonl','控制轨迹']]) {
      if (available) { const link = fileLink(item.run_id,name,label); if (link) files.append(link); }
    }
    target.append(files,rawDetails({run_id:item.run_id,manifest,summary},'配置与诊断'));
  }
  function renderTask(item) {
    const target = $('record-detail'), group = item.group;
    target.replaceChildren(heading(item, `${stamp(group.created)} · ${group.model || '模型未记录'}`));
    if (group.truncated) target.append(node('p','record-inline-note','任务开头不在当前可见记录中，以下只展示已保留部分。'));
    target.append(metricRow([['工具调用',String(group.trace.filter(event => event.kind === 'tool').length)],
      ['模型调用',String(group.trace.filter(event => event.kind === 'api').length)],['关联执行',String(group.runIds.length)]]));
    if (group.answer) { const answer = node('div','record-answer'); answer.append(node('h4','record-section-title','模型回复'),richText(group.answer)); target.append(answer); }
    target.append(node('h4','record-section-title','动作过程'),traceList(group.trace));
    if (group.runIds.length) {
      target.append(node('h4','record-section-title','执行结果'));
      for (const runId of group.runIds) {
        const button = node('button','record-linked-run',runTitle(entries.find(row => row.run_id === runId) || {}) + ' ↗');
        button.type = 'button'; button.addEventListener('click',() => select(`run:${runId}`)); target.append(button);
      }
    }
    target.append(rawDetails(group,'完整调用记录'));
  }
  async function select(id) {
    selected = id; const request = ++detailRequest;
    for (const row of $('records-list').querySelectorAll('.record-row')) {
      row.setAttribute('aria-current',String(row.dataset.record === id)); row.tabIndex = row.dataset.record === id ? 0 : -1;
    }
    const item = entries.find(row => row.id === id);
    if (!item) return;
    $('record-detail').setAttribute('aria-busy','false');
    if (item.type === 'task') { renderTask(item); return; }
    empty($('record-detail'),'正在读取记录…'); $('record-detail').setAttribute('aria-busy','true');
    try {
      const value = await loadArchive(item);
      if (request === detailRequest && dialog.open) renderNative(item,value);
    } catch (error) { if (request === detailRequest) empty($('record-detail'),error.message); }
    finally { if (request === detailRequest) $('record-detail').setAttribute('aria-busy','false'); }
  }
  async function load() {
    if (fetching) { pendingReload = true; return; } fetching = true; const current = ++generation, role = getRole();
    $('records-refresh').disabled = true; notice('');
    try {
      const results = await Promise.allSettled([api('/api/experiments'),api('/api/agent?role='+role)]);
      if (current !== generation || !dialog.open || role !== getRole()) return;
      const catalogue = results[0].status === 'fulfilled' ? results[0].value.experiments : [];
      if (!Array.isArray(catalogue)) throw new Error('记录目录格式不正确，请刷新后重试');
      const groups = results[1].status === 'fulfilled' ? groupTurns(results[1].value.turns) : [];
      const runs = catalogue.filter(row => archiveURL(row.run_id,'manifest.json')).map(row =>
        ({...row,id:`run:${row.run_id}`,type:'run',title:runTitle(row)}));
      const linked = new Set(groups.flatMap(group => group.runIds));
      for (const id of linked) if (!runs.some(row => row.run_id === id)) runs.push({id:`run:${id}`,run_id:id,type:'run',title:'执行记录',status:'unknown',linked:true});
      const tasks = groups.map(group => ({id:`task:${group.id}`,type:'task',group,title:group.prompt || '任务记录',
        status:group.status,started_at:group.created,kind:'task'}));
      entries = [...tasks,...runs].sort((a,b) => (b.started_at || 0) - (a.started_at || 0) || (b.run_id || '').localeCompare(a.run_id || ''));
      for (const run of runs) run.linked = linked.has(run.run_id);
      // Linked executions remain addressable in details, but one goal occupies one list row.
      const all = entries; entries = entries.filter(row => !row.linked || row.id === selected);
      if (selected && !entries.some(row => row.id === selected)) selected = '';
      renderList(true);
      // Keep explicit children available without presenting them as new user tasks.
      for (const row of all) if (!entries.includes(row)) entries.push({...row,detailOnly:true});
      const failed = results.filter(result => result.status === 'rejected');
      notice(failed.length ? '部分记录暂不可用，可点击刷新重试。' : '');
    } catch (error) {
      if (current === generation && dialog.open) notice(error.message);
    } finally { fetching = false; $('records-refresh').disabled = false;
      if (pendingReload && dialog.open) { pendingReload = false; load(); }
    }
  }
  $('experiment-button').addEventListener('click',() => {
    dialog.showModal(); $('records-search').focus(); empty($('records-list'),'正在读取记录…'); load();
  });
  $('records-refresh').addEventListener('click',() => { cache.clear(); selected=''; load(); });
  $('records-search').addEventListener('input',() => renderList());
  $('records-filter').addEventListener('change',() => renderList());
  $('records-list').addEventListener('keydown',event => {
    const row = event.target.closest('.record-row'); if (!row) return;
    const rows = [...$('records-list').querySelectorAll('.record-row')], index = rows.indexOf(row);
    const next = {ArrowDown:Math.min(index+1,rows.length-1),ArrowUp:Math.max(0,index-1),Home:0,End:rows.length-1}[event.key];
    if (next === undefined || event.ctrlKey || event.metaKey || event.altKey) return;
    event.preventDefault(); rows[next].focus(); select(rows[next].dataset.record);
  });
  dialog.addEventListener('close',() => { generation++; detailRequest++; clearTimeout(reloadTimer); });
  return {
    render(data) {
      const token = `${data.agent?.revision}:${data.runs?.join(',')}:${data.last_run?.run_id}:${data.state}`;
      if (token === sourceToken) return;
      sourceToken = token;
      if (dialog.open) { clearTimeout(reloadTimer); reloadTimer = setTimeout(load,500); }
    },
  };
}
