import {useEffect,useRef,useState} from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {ArrowUp,Camera,CheckCircle2,ChevronDown,FileVideo,Loader2,MessageSquare,Plus,Settings2,Square,Terminal,X,Activity,Menu} from 'lucide-react';
import ManagementPanel from './ManagementPanel';
import type {ManagementSection} from './ManagementPanel';
import ModelPicker from './ModelPicker';
import './real-command.css';
type Data=Record<string,any>;
const names:Data={cam_middle:'全景相机',cam_left:'左腕相机',cam_right:'右腕相机'};
const time=(v:any)=>v?new Date(typeof v==='number'?v*1000:v).toLocaleTimeString('zh-CN',{hour12:false}):'—';
async function get(url:string){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw new Error('读取失败 '+r.status);return r.json()}
async function post(url:string,body:Data){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const v=await r.json();if(!r.ok)throw new Error(v.error||'请求失败');return v}
export default function RealCommand(){
 const [state,setState]=useState<Data>({cameras:{},archive:[]}),[core,setCore]=useState<Data>({sessions:[]});
 const [session,setSession]=useState(()=>localStorage.getItem('paos.real.session')||'console:real-'+crypto.randomUUID().slice(0,8));
 const [events,setEvents]=useState<Data[]>([]),[input,setInput]=useState(()=>sessionStorage.getItem('paos.real.draft.'+session)||''),[error,setError]=useState(''),[online,setOnline]=useState(false),[sending,setSending]=useState(false);
 const [execution,setExecution]=useState(false),[camera,setCamera]=useState('cam_middle'),[view,setView]=useState('chat'),[video,setVideo]=useState<Data|null>(null);
 const [modelOpen,setModelOpen]=useState(false),[management,setManagement]=useState<ManagementSection|null>(null),[navOpen,setNavOpen]=useState(false),[observeOpen,setObserveOpen]=useState(false),[streamOnline,setStreamOnline]=useState(false),[notice,setNotice]=useState('');
 const bottom=useRef<HTMLDivElement>(null),sessionRef=useRef(session),request=useRef<{id:string,content:string,motion:boolean,session:string}|null>(null);
 const current=state.current?.session===session?state.current:null;
 const activeSession=core.job?.session||(state.current?.phase==='running'?state.current.session:null);
 function setDraft(value:string){const key='paos.real.draft.'+sessionRef.current;if(value)sessionStorage.setItem(key,value);else sessionStorage.removeItem(key);setInput(value)}
 function selectSession(key:string){if(sending)return;sessionRef.current=key;setSession(key);setInput(sessionStorage.getItem('paos.real.draft.'+key)||'');setExecution(false);setError('');setNotice('');request.current=null;setView('chat');setNavOpen(false)}
 function selectView(value:string){setView(value);setNavOpen(false)}
 function mergeEvents(a:Data[],b:Data[]){return [...new Map([...a,...b].map(e=>[e.id,e])).values()].sort((a,b)=>a.id-b.id)}
 async function refresh(){const [s,c]=await Promise.all([get('/api/real-command/state'),get('/api/state')]);setState(s);setCore(c)}

 useEffect(()=>{sessionRef.current=session;localStorage.setItem('paos.real.session',session);setEvents([]);
  get('/api/session?key='+encodeURIComponent(session)).then(d=>{if(sessionRef.current===session)setEvents(old=>mergeEvents(d.events||[],old.filter(e=>e.session===session)))}).catch(e=>setError(e.message));
 },[session]);
 useEffect(()=>{let dead=false;async function poll(){try{const [s,c]=await Promise.all([get('/api/real-command/state'),get('/api/state')]);if(!dead){setState(s);setCore(c);setOnline(true)}}catch{if(!dead)setOnline(false)}}poll();const timer=setInterval(poll,1100);return()=>{dead=true;clearInterval(timer)}},[]);
 useEffect(()=>{if(!online)return;let stream:EventSource;let dead=false;get('/api/state').then(d=>{if(dead)return;stream=new EventSource('/api/events?after='+d.last_event_id);
   stream.onopen=()=>{setStreamOnline(true);const key=sessionRef.current;get('/api/session?key='+encodeURIComponent(key)).then(h=>{if(!dead&&key===sessionRef.current)setEvents(old=>mergeEvents(h.events||[],old.filter(e=>e.session===key)))}).catch(e=>{if(!dead)setError(e.message)})};
   stream.onerror=()=>setStreamOnline(false);stream.addEventListener('console',(e:MessageEvent)=>{const v=JSON.parse(e.data);if(v.session===sessionRef.current)setEvents(old=>old.some(x=>x.id===v.id)?old:[...old,v])})}).catch(e=>{if(!dead){setStreamOnline(false);setError(e.message)}});return()=>{dead=true;stream?.close()}},[online]);
 useEffect(()=>{bottom.current?.scrollIntoView({behavior:'smooth'})},[events.length]);
 async function send(){
  const original=input;const targetSession=session;const text=input.trim();if(!text||sending||state.busy||!online)return;
  setSending(true);setError('');setNotice('');
  if(!request.current||request.current.content!==text||request.current.motion!==execution||request.current.session!==session)request.current={id:crypto.randomUUID(),content:text,motion:execution,session};
  try{await post('/api/real-command/message',{content:text,session,request_id:request.current.id,allow_motion:execution});if(sessionStorage.getItem('paos.real.draft.'+targetSession)===original){sessionStorage.removeItem('paos.real.draft.'+targetSession);if(sessionRef.current===targetSession)setInput('')}request.current=null}
  catch(e:any){setError(e.message)}finally{setSending(false)}
 }
 async function control(action:string,body:Data={}){
  setSending(true);setError('');try{const result=await post('/api/real-command/'+action,body);if(action==='stop')setNotice(result.phase==='stopped'?'任务已停止。':result.requested?'已请求停止，等待任务结束回执。':'当前没有正在执行的任务。');await refresh()}catch(e:any){setError(e.message)}finally{setSending(false)}
 }
 function newChat(){selectSession('console:real-'+crypto.randomUUID().slice(0,8))}
 const sessions=[...new Set<string>([session,...Object.keys(sessionStorage).filter(k=>k.startsWith('paos.real.draft.')).map(k=>k.slice('paos.real.draft.'.length)),...(core.sessions||[]).map((s:Data)=>s.key).filter((s:string)=>s.startsWith('console:real-'))])];
 const cam=state.cameras?.[camera]||{};
 const live=online&&cam.fresh;
 const cameraOn=['active','activating'].includes(state.camera_health?.ActiveState);
 const cameraFailed=state.camera_health?.ActiveState==='failed';
 const busy=state.busy||sending;
 useEffect(()=>{const escape=(e:KeyboardEvent)=>{if(e.key==='Escape'&&!modelOpen&&!management){setVideo(null);setNavOpen(false);setObserveOpen(false)}};document.addEventListener('keydown',escape);return()=>document.removeEventListener('keydown',escape)},[modelOpen,management]);
 const count=(state.archive||[]).reduce((n:number,g:Data)=>n+(g.files?.length??g.count??0),0);

 return <div className="command-app">
  {navOpen&&<button className="command-nav-shade" aria-label="关闭导航" onClick={()=>setNavOpen(false)}/>}
  <aside className={"command-nav "+(navOpen?'nav-open':'')}>
   <button className="command-brand" onClick={()=>selectView('chat')}><span>P</span><div><b>PhyAgentOS</b><small>机器人 Agent 工作台</small></div></button>
   <button className="command-new" disabled={sending} onClick={newChat}><Plus size={17}/>新建任务</button>
   <div className="command-nav-label">工作区</div>
   <button className={view==='chat'?'nav-active':''} onClick={()=>selectView('chat')}><MessageSquare size={16}/>任务对话</button>
   <button className={view==='videos'?'nav-active':''} onClick={()=>selectView('videos')}><FileVideo size={16}/>实验录像<span className="command-count">{count}</span></button>
   <div className="command-nav-label">最近任务</div>
   <div className="command-sessions">{sessions.map(s=><button className={s===session?'nav-active':''} key={s} disabled={sending} onClick={()=>selectSession(s)}><MessageSquare size={14}/><span>{s.replace('console:real-','任务 ')}</span></button>)}</div>
   <button className="command-settings" onClick={()=>{setModelOpen(true);setNavOpen(false)}}><Settings2 size={16}/>模型设置</button>
   <button className="command-settings" onClick={()=>{setManagement('skills');setNavOpen(false)}}><Terminal size={16}/>技能与工作区</button>
   <a className="command-settings command-legacy" href="/?view=workspace"><Activity size={16}/>原有任务工作台</a>
   <div className="command-host"><i className={online?'is-on':''}/><div>4090-48<small>{online?'服务已连接':'服务连接中'}</small></div></div>
  </aside>
  <header className="command-header"><button className="command-menu-button" aria-label="打开导航" aria-expanded={navOpen} onClick={()=>setNavOpen(v=>!v)}><Menu size={19}/></button><div><b>{view==='videos'?'实验录像':'任务对话'}</b><span>PiPER 双臂 · 真机</span></div><button className="command-model" aria-label="切换模型" aria-haspopup="dialog" onClick={()=>setModelOpen(true)} title={state.model}>{state.model||'读取模型…'}<ChevronDown size={13}/></button><button className="command-observe-toggle" aria-label="打开现场观察" aria-expanded={observeOpen} onClick={()=>{setView('chat');setObserveOpen(true)}}><Camera size={18}/></button><span className={'command-badge '+(state.busy?'working':'')}>{!online?'连接中':state.busy?'处理中':'等待命令'}</span></header>
  {view==='videos'?<main className="command-videos"><h1>实验录像</h1><p>保留三轮任务的全部{count}份录像，成功与未完成过程分别记录。</p>{state.archive.map((g:Data)=><details className="command-video-group" key={g.id}><summary><FileVideo size={19}/><div><b>10月24日 · {g.title}</b><small>{g.result}</small></div><span>{g.files?.length??g.count} 份</span><ChevronDown size={15}/></summary><div className="command-video-list">{g.files?.map((f:Data)=><button key={f.path} onClick={()=>setVideo(f)}><FileVideo size={15}/><span>{f.title}</span><small>{(f.bytes/1024/1024).toFixed(1)} MB</small><span>播放 ↗</span></button>)}</div></details>)}</main>:<>
  <main className="command-conversation">
   <div className="command-chat-scroll">
    {!events.length?<div className="command-welcome"><span className="command-orb"><Terminal size={30}/></span><div className="command-eyebrow">YOUR EMBODIED AGENT</div><h1>输入目标，让机器人执行。</h1><p>Agent 观察现场、规划相对动作并调用技能。<br/>调用过程、现场画面和执行结果在这里展示。</p><div className="command-suggestions">
      <button onClick={()=>{setExecution(false);setDraft('/status')}}><Activity size={18}/><b>检查当前状态</b><small>不下发动作</small></button>
      <button onClick={()=>{setExecution(false);setDraft('观察三路现场画面，说明瓶子、夹爪和红桶的位置，不执行动作。')}}><Camera size={18}/><b>观察现场</b><small>读取当前画面</small></button>
      <button onClick={()=>{setExecution(true);setDraft('根据当前现场，用相对控制将黑框内的瓶子放入红桶。先观察，逐段执行，每段后检查；发生失败立即停止并报告。')}}><Terminal size={18}/><b>瓶子放入红桶</b><small>真机任务</small></button>
    </div></div>:<div className="command-messages">{events.filter(e=>['accepted','answer','progress','tool','tool_result','error','agent_cancelled','cancel_requested'].includes(e.type)).map(e=><article key={e.id} className={'command-message type-'+e.type}>
       {e.type==='accepted'?<div className="command-user">{e.content}</div>:e.type==='tool_result'?<details className="command-tool"><summary><CheckCircle2 size={14}/><b>{e.name}</b><span>调用回执</span><ChevronDown size={14}/></summary><pre>{(()=>{try{return JSON.stringify(JSON.parse(e.result),null,2)}catch{return e.result}})()}</pre></details>:e.type==='tool'?<div className="command-call"><Terminal size={14}/>{e.content}</div>:<div className="command-agent"><span className="command-agent-icon">{e.type==='error'?'!':'P'}</span><div><Markdown remarkPlugins={[remarkGfm]}>{e.content||''}</Markdown>{e.media?.filter((u:string)=>u.startsWith('/api/real-command/artifact?')).length>0&&<div className="command-evidence">{e.media.filter((u:string)=>u.startsWith('/api/real-command/artifact?')).map((u:string)=><a key={u} href={u} target="_blank" rel="noreferrer"><img src={u} alt="本次真实观察关键帧"/></a>)}</div>}<small>{time(e.at)} · {e.type==='answer'?'Agent 回复':'观察与进度'}</small></div></div>}
    </article>)}</div>}
    {state.busy&&<div className="command-thinking"><Loader2 size={15}/>{activeSession===session?'Agent 正在观察或执行':'工作台正在处理其他任务'}{activeSession&&activeSession!==session&&<button onClick={()=>selectSession(activeSession)}>返回执行中的任务</button>}</div>}<div ref={bottom}/>
   </div>
   <div className="command-compose">
    {(!online||!streamOnline)&&<div className="command-connection" role="status">{!online?'服务连接已断开，正在重连；命令暂不可发送。':'调用记录连接中，正在恢复实时更新。'}</div>}
    {notice&&<div className="command-result" role="status">{notice}</div>}
    {state.unresolved&&<div className="command-error">上次任务的执行结果需要核对，已阻止新的动作下发。</div>}
    {current&&current.phase!=='running'&&<div className="command-result">本次处理{current.phase==='completed'?'已结束':current.phase==='stopped'?'已停止':'遇到问题'} {current.allow_motion?' · 物体任务结果请查看观察记录':''}</div>}
    {error&&<div className="command-error">{error}<button aria-label="关闭错误" onClick={()=>setError('')}><X size={14}/></button></div>}
    <form onSubmit={e=>{e.preventDefault();send()}}><textarea aria-label="任务命令" placeholder={execution?'描述要执行的真机任务，例如：将黑框内的瓶子放入红桶…':'输入命令或描述要观察的内容…'} value={input} onChange={e=>setDraft(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.nativeEvent.isComposing){e.preventDefault();send()}}}/>
     <div className="command-compose-footer"><select aria-label="执行方式" value={execution?'motion':'observe'} disabled={busy} onChange={e=>setExecution(e.target.value==='motion')}><option value="observe">仅观察 · 不发送动作</option><option value="motion">真机执行 · 相对控制</option></select>{(core.job||state.invocation_id)?<button type="button" className="command-stop" onClick={()=>control('stop')} disabled={sending}><Square size={15}/>停止任务</button>:<button type="submit" className="command-send" aria-label="发送命令" disabled={!input.trim()||busy||!online||!!state.unresolved}><ArrowUp size={19}/></button>}</div>
    </form><p>Enter 发送 · Shift + Enter 换行<span>执行状态与物体结果分别记录</span></p>
   </div>
  </main>
  {observeOpen&&<button className="command-observation-shade" aria-label="关闭观察背景" onClick={()=>setObserveOpen(false)}/>}
  <aside className={"command-observation "+(observeOpen?'observation-open':'')}><div className="command-observe-head"><b><Camera size={17}/>现场观察</b><button className="command-observe-close" aria-label="关闭现场观察面板" onClick={()=>setObserveOpen(false)}><X size={18}/></button><span className={'command-live '+(live?'is-on':'')}>{live?'● LIVE':'未直播'}</span></div>
   <div className="command-camera-tabs">{Object.entries(names).map(([n,title])=><button className={camera===n?'selected':''} key={n} onClick={()=>setCamera(n)}>{String(title)}</button>)}</div>
   <div className="command-camera">{cam.saved_wall_s?<img src={'/api/real-command/frame/'+camera+'?t='+cam.saved_wall_s} alt={names[camera]}/>:<div><Camera size={29}/><span>尚未开启现场观察</span></div>}{!live&&cam.saved_wall_s&&<div className="command-camera-stale">历史帧 · 不作为当前执行依据</div>}</div>
   <div className="command-image-time"><span>{time(cam.saved_wall_s)}</span><span>{cam.age_s!=null?cam.age_s.toFixed(1)+'秒前':'等待图像'}</span></div>
   <button className="command-observe-button" disabled={busy||!online} onClick={()=>control('camera',{action:cameraOn?'stop':'start'})}><Camera size={16}/>{cameraOn?(cameraOn&&!live?'等待画面 · 点击停止':'关闭现场观察'):'开启现场观察'}</button>
   {cameraFailed&&<div className="command-error">观察进程已退出（{state.camera_health?.Result}，状态码{state.camera_health?.ExecMainStatus}），请检查相机占用与日志后重新开启。</div>}<div className="command-camera-note">开启观察只读取相机。选择“真机执行”后发送命令，才会允许本次任务下发动作。</div>
   <div className="command-facts"><h3>当前执行</h3><div><span>技能</span><b>PiPER 真机</b></div><div><span>控制方式</span><b>相对控制</b></div><div><span>当前任务</span><b>{state.busy?'处理中':'空闲'}</b></div><div><span>动作链路</span><b>{state.runtime_mode==='live'?'已连接':'未占用'}</b></div>{state.invocation_id&&<details><summary>当前调用</summary><code>{state.invocation_id}</code></details>}</div>
   <button className="command-history-link" onClick={()=>{selectView('videos');setObserveOpen(false)}}><FileVideo size={16}/>查看三轮实验录像<span>23 份</span></button>
  </aside></>}
  {video&&<div className="command-video-shade" onClick={()=>setVideo(null)}><section onClick={e=>e.stopPropagation()}><header><b>{video.title}</b><button aria-label="关闭视频" onClick={()=>setVideo(null)}><X size={20}/></button></header><video src={video.url} controls autoPlay/><a href={video.url} target="_blank" rel="noreferrer">单独打开录像 ↗</a></section></div>}
  {modelOpen&&<ModelPicker locked={!!state.busy||!!state.unresolved||sending} onClose={()=>setModelOpen(false)} onSaved={model=>{setState(s=>({...s,model}));refresh().catch(e=>setError(e.message))}} onAdvanced={()=>{setModelOpen(false);setManagement('settings')}}/>}
  {management&&<ManagementPanel commandContext initial={management} state={{...core,...state,agent_mode:'real',control_busy:busy||!!state.unresolved}} onClose={()=>setManagement(null)} onUpdated={()=>{refresh().catch(e=>setError(e.message))}}/>}
 </div>
}
