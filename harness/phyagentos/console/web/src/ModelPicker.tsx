import {useEffect,useRef,useState} from 'react';
import {Check,ChevronDown,Loader2,Settings2,X} from 'lucide-react';
type Data=Record<string,any>;
export default function ModelPicker({locked,onClose,onSaved,onAdvanced}:{locked:boolean;onClose:()=>void;onSaved:(model:string)=>void;onAdvanced:()=>void}){
 const [data,setData]=useState<Data>({}),[model,setModel]=useState(''),[provider,setProvider]=useState('');
 const [loading,setLoading]=useState(true),[saving,setSaving]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState('');
 const panel=useRef<HTMLElement>(null);
 useEffect(()=>{let dead=false;fetch('/api/manage/settings',{cache:'no-store'}).then(async r=>{const d=await r.json();if(!r.ok)throw new Error(d.error||'无法读取模型配置');return d}).then(d=>{if(!dead){setData(d);setModel(d.defaults.model);setProvider(d.active_provider||d.defaults.provider)}}).catch(e=>{if(!dead)setError(e.message)}).finally(()=>{if(!dead)setLoading(false)});return()=>{dead=true}},[]);
 useEffect(()=>{const previous=document.activeElement as HTMLElement;panel.current?.focus();return()=>previous?.focus()},[]);
 const providers=Object.entries<Data>(data.providers||{}).filter(([,v])=>v.enabled!==false&&(v.api_key_configured||v.api_base));
 const models=[...new Set<string>([data.defaults?.model,...Object.values<Data>(data.modes?.models||{}).map(v=>v.model)].filter(Boolean))];
 async function save(){
  if(saving||locked||!model.trim())return;setSaving(true);setError('');setNotice('');
  try{const r=await fetch('/api/manage/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'switch_model',provider,model:model.trim()})});const d=await r.json();if(!r.ok)throw new Error(d.error||'模型切换失败');setData(d);setNotice('已生效：'+d.active_model+'。下一条命令使用此模型。');onSaved(d.active_model)}
  catch(e:any){setError(e.message)}finally{setSaving(false)}
 }
 return <div className="command-modal-shade" onMouseDown={e=>{if(e.target===e.currentTarget&&!saving)onClose()}}>
  <section className="command-model-dialog" role="dialog" aria-modal="true" aria-labelledby="model-title" ref={panel} tabIndex={-1} onKeyDown={e=>{
   if(e.key==='Escape'&&!saving){e.stopPropagation();onClose()}
   if(e.key==='Tab'){const items=panel.current?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),select:not(:disabled),[tabindex="0"]');if(!items?.length)return;const first=items[0],last=items[items.length-1];if(e.shiftKey&&(document.activeElement===first||document.activeElement===panel.current)){e.preventDefault();last.focus()}else if(!e.shiftKey&&(document.activeElement===last||document.activeElement===panel.current)){e.preventDefault();first.focus()}}
  }}>
   <header><div><small>AGENT MODEL</small><h2 id="model-title">切换模型</h2></div><button aria-label="关闭模型面板" disabled={saving} onClick={onClose}><X size={20}/></button></header>
   <p>当前生效 <b>{data.active_model||'读取中…'}</b><br/>保存后留在当前任务，草稿与观察画面保留。</p>
   {loading?<div className="command-model-loading"><Loader2 size={18}/>读取配置…</div>:<form onSubmit={e=>{e.preventDefault();save()}}>
    <label>模型服务<select aria-label="模型服务" value={provider} disabled={saving||locked} onChange={e=>{setProvider(e.target.value);setNotice('')}}>{providers.map(([name])=><option key={name} value={name}>{name}</option>)}</select></label>
    <label>模型 ID<input aria-label="模型 ID" value={model} list="command-model-ids" disabled={saving||locked} onChange={e=>{setModel(e.target.value);setNotice('')}} spellCheck={false}/><datalist id="command-model-ids">{models.map(m=><option key={m} value={m}/>)}</datalist></label>
    <small className="command-model-hint">可选用已配置的模型，或填写当前服务支持的模型 ID。服务地址与密钥沿用现有配置。</small>
    {locked&&<div className="command-error">任务执行中或结果待核对，结束后才能切换模型。</div>}
    {error&&<div className="command-error" role="alert">{error}</div>}
    {notice&&<div className="command-model-notice" role="status"><Check size={17}/>{notice}</div>}
    <footer><button type="button" disabled={saving} onClick={onAdvanced}><Settings2 size={15}/>高级设置</button><div><button type="button" disabled={saving} onClick={onClose}>{notice?'完成':'取消'}</button><button className="command-primary" type="submit" disabled={loading||saving||locked||!provider||!model.trim()}>{saving?<Loader2 size={15}/>:<ChevronDown size={15}/>}保存并切换</button></div></footer>
   </form>}
   {loading&&error&&<div className="command-error" role="alert">{error}</div>}
  </section>
 </div>
}
