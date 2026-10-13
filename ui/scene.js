const $=id=>document.getElementById(id);
const icons={
  cylinder:'<ellipse cx="24" cy="12" rx="10" ry="4"/><path d="M14 12v24c0 5 20 5 20 0V12M14 36c0 5 20 5 20 0"/>',
  'short-cylinder':'<ellipse cx="24" cy="20" rx="12" ry="4"/><path d="M12 20v13c0 6 24 6 24 0V20M12 33c0 6 24 6 24 0"/>',
  block:'<path d="m12 17 16-7 10 7v23l-16 6-10-7Zm0 0 10 7 16-7M22 24v22"/>',
  cup:'<path d="M11 17h22v18c0 10-22 10-22 0ZM33 20h4c9 0 9 14 0 14h-4"/><path d="M15 10v3m7-5v5m7-3v3"/>',
  book:'<path d="m9 21 24-9 7 15-24 9ZM9 21v9l7 10 24-9v-4M16 36v4M13 29l18-7"/>',
};
function icon(name){
  const node=document.createElementNS('http://www.w3.org/2000/svg','svg');
  node.setAttribute('viewBox','0 0 48 48');node.setAttribute('aria-hidden','true');
  node.innerHTML=icons[name]||icons.block;return node;
}
export function createSceneUI({api,toast}){
  let latest=null,anchor='',draft=new Set(),busy=false;
  const button=$('scene-library-button'),dialog=$('scene-dialog'),apply=$('scene-apply');
  function update(){
    const changed=latest&&JSON.stringify([...draft].sort())!==JSON.stringify([...latest.selected].sort());
    apply.disabled=busy||!changed||!latest?.editable;
    $('scene-count').textContent=`${draft.size} 件可选物品`;
  }
  function cards(){
    $('scene-items').replaceChildren(...latest.items.map(item=>{
      const card=document.createElement('button');card.type='button';card.className='scene-item';
      card.setAttribute('aria-label',item.label);card.setAttribute('aria-pressed',String(draft.has(item.object_id)));
      const drawing=icon(item.icon),name=document.createElement('strong'),description=document.createElement('small');
      name.textContent=item.label;description.textContent=item.description;
      const check=document.createElement('span');check.className='scene-check';check.textContent='✓';check.setAttribute('aria-hidden','true');
      card.append(drawing,name,description,check);card.disabled=busy||!latest.editable;
      card.addEventListener('click',()=>{draft.has(item.object_id)?draft.delete(item.object_id):draft.add(item.object_id);card.setAttribute('aria-pressed',String(draft.has(item.object_id)));update();});
      return card;
    }));update();
  }
  button.addEventListener('click',()=>{
    if(!latest)return;anchor=latest.observation_id;draft=new Set(latest.selected);$('scene-result').textContent='';cards();dialog.showModal();
  });
  $('scene-none').addEventListener('click',()=>{draft.clear();cards();});
  apply.addEventListener('click',async()=>{
    busy=true;cards();$('scene-result').textContent='正在生成场景…';
    try{
      await api('/api/scene/library',{object_ids:[...draft],observation_id:anchor});
      dialog.close();toast('物品已应用，新场景已准备好');
    }catch(error){$('scene-result').textContent=error.message;toast(error.message);}
    finally{busy=false;cards();}
  });
  return {render(data){
    latest=data.scene_library||null;button.hidden=!latest;
    if(latest){latest={...latest,editable:latest.editable&&!data.agent?.busy&&!Object.values(data.diagnostics?.evaluation?.touching||{}).some(Boolean)};button.disabled=!latest.editable;}
    if(dialog.open)update();
  }};
}
