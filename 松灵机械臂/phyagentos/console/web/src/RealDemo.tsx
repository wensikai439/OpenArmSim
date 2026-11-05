import {useEffect,useState,useRef} from 'react';
import {Activity,Camera,ChevronDown,Play,Sparkles,Terminal,ArrowLeft,CheckCircle2,AlertTriangle} from 'lucide-react';
import './real-demo.css';
type Data=Record<string,any>;
const readonly=new URLSearchParams(location.search).get('source')==='readonly';
const artifact=(p:string)=>readonly?p:'/api/real-demo/artifact?path='+encodeURIComponent(p);
const frame=(n:string,t:any)=>readonly?n+'-latest.jpg?t='+t:'/api/real-demo/frame/'+n+'?t='+t;
const when=(t:number)=>new Date(t*1000).toLocaleTimeString('zh-CN',{hour12:false});
const names:Record<string,string>={cam_middle:'全景相机',cam_left:'左腕相机',cam_right:'右腕相机'};
const phase:Record<string,string>={running:'执行中',completed:'原生执行完成',failed:'执行失败',unknown:'结果待核查',pending:'准备中'};
export default function RealDemo(){
 const timeline=useRef<HTMLDivElement>(null);
 const [data,setData]=useState<Data>({cameras:{},events:[],stages:[]}),[error,setError]=useState(''),[busy,setBusy]=useState(false),[camera,setCamera]=useState('cam_right');
 useEffect(()=>{let dead=false;async function poll(){try{const r=await fetch(readonly?'demo-state.json?t='+Date.now():'/api/real-demo/state');if(!r.ok)throw new Error('演示状态读取失败');const d=await r.json();if(!dead){setData(d);setError('')}}catch(e:any){if(!dead)setError(e.message)}}poll();const timer=setInterval(poll,800);return()=>{dead=true;clearInterval(timer)}},[]);
 async function execute(){setBusy(true);try{const r=await fetch('/api/real-demo/execute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({plan_digest:data.plan_digest,request_id:crypto.randomUUID()})});const v=await r.json();if(!r.ok)throw new Error(v.error)}catch(e:any){setError(e.message)}finally{setBusy(false)}}
 useEffect(()=>{if(timeline.current)timeline.current.scrollTop=timeline.current.scrollHeight},[data.events.length]);
 const snapshotFresh=!!data.server_wall_s&&Date.now()/1000-data.server_wall_s<3;
 const live=snapshotFresh&&Object.values(data.cameras).every((c:any)=>c.fresh)&&Object.keys(data.cameras).length===3;
 const cam=data.cameras[camera]||{};
 return <div className="real-demo">
  <header className="demo-header"><a href="/" className="demo-brand"><span>P</span><b>PhyAgentOS<small>Agent 控制台</small></b></a><div className="demo-title"><span>真机任务演示</span><b>瓶子抓放</b></div><span className={'demo-live '+(live?'on':'')}>{live?'● 现场画面实时':'● 历史画面 · 当前未直播'}</span><a className="demo-back" href="/"><ArrowLeft size={15}/>工作台</a></header>
  <div className="demo-layout"><main className="demo-task"><div className="demo-kicker"><Sparkles size={16}/>目标与执行过程</div><h1>观察现场，调用技能，<br/>完成瓶子抓放。</h1><p className="demo-subtitle">{data.planner||'读取执行者…'}<br/>参考示教顺序，根据当前场景规划相对动作。</p>
   <div className="demo-goal"><span>本次目标</span><b>{data.goal}</b><small>{data.run}</small></div>
   {data.outcome&&<div className="demo-outcome">{data.outcome.status==='success'?<CheckCircle2 size={18}/>:<AlertTriangle size={18}/>}<div><b>{data.outcome.title}</b><p>{data.outcome.observation}</p>{data.outcome.replay_url&&<a href={data.outcome.replay_url} target="_blank">查看实际录像与关键帧 ↗</a>}</div></div>}
   <div className="demo-section"><h2><Terminal size={17}/>调用与观察 <span>{data.events.length} 条记录</span></h2><div className="demo-timeline" ref={timeline}>{data.events.length?data.events.map((e:Data,i:number)=><article key={i} className={'demo-event kind-'+e.kind}><div className="demo-event-icon">{e.kind==='observation'?<Camera size={15}/>:e.kind==='result'?(e.exit_code===0?<CheckCircle2 size={15}/>:<AlertTriangle size={15}/>):e.kind==='call'?<Terminal size={15}/>:<Sparkles size={15}/>}</div><div className="demo-event-body"><div><b>{e.title}</b><time>{when(e.at)}</time></div>{e.content&&<p>{e.content}</p>}{e.name&&<code>{e.name}</code>}{e.images&&<div className="demo-evidence">{e.images.map((p:string)=><a key={p} href={artifact(p)} target="_blank"><img src={artifact(p)} alt="本次观察关键帧"/></a>)}</div>}{(e.arguments||e.result||e.error)&&<details><summary>调用参数与原始回执 <ChevronDown size={13}/></summary><pre>{JSON.stringify(e.arguments||e.result||e.error,null,2)}</pre></details>}</div></article>):<p className="demo-muted">正在读取现场，尚未下发动作。</p>}</div></div>
   <div className="demo-plan"><div><span>当前准备动作</span><b>{data.plan?.title||'等待目标定位与动作规划'}</b><p>{data.plan?.description||'先确认瓶子与夹爪的相对位置。'}</p></div><button onClick={execute} disabled={readonly||!data.can_execute||busy}><Play size={16}/>{data.worker?.phase==='running'?'动作执行中':readonly?'由当前 Codex 下发':'执行当前准备动作'}</button>{data.plan&&<details><summary>相对动作参数</summary><pre>{JSON.stringify(data.plan,null,2)}</pre></details>}{error&&<p className="demo-error">{error}</p>}</div>
  </main><aside className="demo-observe"><div className="demo-observe-heading"><h2><Camera size={17}/>现场观察</h2><span>{cam.saved_wall_s?when(cam.saved_wall_s):'等待相机'}</span></div><div className="demo-camera-tabs">{Object.keys(names).map(n=><button key={n} className={camera===n?'active':''} onClick={()=>setCamera(n)}>{names[n]}</button>)}</div><div className="demo-main-camera">{cam.saved_wall_s?<img src={frame(camera,cam.saved_wall_s)} alt={names[camera]+(live?'实时画面':'历史画面')}/>:<div>等待真实相机图像</div>}{(!snapshotFresh||!cam.fresh)&&<span className="demo-stale">画面已过期 · 禁止下发</span>}</div><div className="demo-camera-mini">{Object.keys(names).filter(n=>n!==camera).map(n=><button key={n} onClick={()=>setCamera(n)}><img src={frame(n,data.cameras[n]?.saved_wall_s)} alt={names[n]}/><span>{names[n]}</span></button>)}</div>
   <div className="demo-status"><h2><Activity size={16}/>执行状态</h2><div><span>当前动作</span><b>{phase[data.worker?.phase]||'等待准备'}</b></div><div><span>已留存动作</span><b>{data.stages.length}</b></div><div><span>控制方式</span><b>启动反馈 + 相对增量</b></div><p>原生执行完成与物体任务成功分别记录。观察结论来自当前 Codex 对真实画面的判断。</p><details><summary>执行链与技能回执</summary><p>{data.chain}</p>{data.stages.map((s:Data)=><div key={s.name} className="demo-stage"><b>{s.name}</b><small>{phase[s.phase]||s.phase} · {s.invocation_id||'等待回执'}</small>{s.result_file&&<a href={artifact(s.result_file)} target="_blank">原始回执 ↗</a>}</div>)}</details></div>
  </aside></div>
 </div>
}
