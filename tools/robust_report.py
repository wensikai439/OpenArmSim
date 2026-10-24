"""Compact evidence page, without changing the main workbench UI."""
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/20261024-robust'


def load(path):return json.loads(path.read_text())


def score(folder):
    rows=load(folder/'results.json');passed=0
    for row in rows:
        r=load(ROOT/row['trial']/'result.json')
        points=r.get('body_relay',{}).get('first_native_grasp_contacts',{}).get('right',{}).get('contacts',[])
        passed+=bool(row['success'] and points and all(p['geom']=='bottle_body' and 0<=p['body_fraction']<=.5 for p in points))
    return {'passed':passed,'total':len(rows),'rows':rows}


def main():
    rows=[];summary={}
    comparisons=[('第一轮开发','development/baseline','development/lower-half'),
        ('第一轮留出 · 16个新位置','heldout/baseline','heldout/lower-half'),
        ('第二轮留出 · 8个新位置','refine/heldout/lower-half','refine/heldout/free-yaw'),
        ('历史32条件回归 · 已用于开发','heldout/baseline','heldout/final-regression')]
    for label,old,new in comparisons:
        a,b=score(OUT/old),score(OUT/new)
        summary[label]={'baseline':{k:v for k,v in a.items() if k!='rows'},'candidate':{k:v for k,v in b.items() if k!='rows'}}
        rows.append(f'<tr><td>{label}</td><td>{a["passed"]}/{a["total"]}</td><td><b>{b["passed"]}/{b["total"]}</b></td></tr>')
    final=score(OUT/'heldout/final-regression');fresh=score(OUT/'refine/heldout/free-yaw')
    all_final=final['rows']+fresh['rows']
    max_drop=max(r['left_drop_mm'] for r in all_final)
    trial_count=sum(1 for p in OUT.rglob('result.json'))
    summary.update(native_trial_count=trial_count,max_final_left_drop_mm=max_drop,
        neural_weights_trained=False,development_api_requests=1,development_api_tokens=3620,
        real_motion_service_connected=False,transfer_ready=False,
        heldout_interpretation='both methods tied in each independent heldout cohort; no demonstrated paired heldout success-rate increase',
        original_round1_failures_reused_as_development=[2207,2215])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    cases=[]
    for row in all_final:
        label='历史回归' if row in final['rows'] else '第二轮新位置'
        state='通过' if row['success'] else '失败'
        failure=(row.get('failure') or {}).get('code')
        recovered=('恢复：'+failure) if failure and row['success'] else failure or '—'
        cases.append(f'<tr><td>{label}</td><td>{row["seed"]}</td><td>{row["group"]}</td><td class="ok">{state}</td><td>{row["left_drop_mm"]:.2f} mm</td><td>{html.escape(recovered)}</td><td><a href="/{row["trial"]}/media/final-head.jpg">关键帧</a></td></tr>')
    source='https://github.com/NVlabs/ASPIRE'
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>瓶身技能 · 泛化复验</title><style>
body{margin:0;background:#f5f3ed;color:#292a25;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:30px;letter-spacing:-1px;margin-bottom:8px}h2{font-size:20px;margin-top:0}a{color:#487661}.sub{color:#66685f}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}.card,section{background:#fffefa;border:1px solid #e3ded3;border-radius:16px;padding:24px;margin-bottom:20px}.number{font-size:32px;font-weight:600;color:#487661}.notice{background:#f6eee0;border:1px solid #d8c7a9;border-radius:12px;padding:18px 22px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;padding:13px 10px;border-bottom:1px solid #e7e2d7}th{color:#66685f;font-weight:500}.ok{color:#487661}ol{padding-left:24px}details{margin:16px 0}summary{cursor:pointer}@media(max-width:700px){.cards{grid-template-columns:1fr}section{overflow:auto}}
</style><main><div class="sub">OpenArm Lab · 本地原生仿真</div><h1>瓶身技能 · 泛化复验</h1><p>两轮开发与冻结测试。保留原始失败，把改进归因到实际生效的工具规则与规划。</p>'''
    body+=f'<div class="cards"><div class="card">原生实验记录<div class="number">{trial_count}</div>含失败、对照和回归</div><div class="card">独立新位置<div class="number">16 + 8</div>每个位置两种条件，分轮统计</div><div class="card">最终组左手最大静止下沉<div class="number">{max_drop:.2f} mm</div>原5 mm门槛未放宽</div></div>'
    body+='<div class="notice"><b>尚未取得真机迁移验收。</b>仿真使用几何真值和暂停规划；公司端仅有纯真机接入客户端，物体感知与动作服务未连接。新旧方法在各自留出组打平，不能声称总体成功率已提升或首次真机必成功。</div>'
    body+='<section style="margin-top:24px"><h2>同组比较</h2><table><tr><th>集合</th><th>对照</th><th>改进</th></tr>'+''.join(rows)+'</table><p class="sub">第一轮两例失败已转作第二轮开发，历史回归不能再次当成未见测试。所有方法使用相同的下半瓶身、接触、碰撞、释放、支撑和稳定验收。</p></section>'
    body+='<section><h2>改进了哪里</h2><ol><li>右手每个有效接触点必须在瓶身下半部。58%高抓会触发支撑确认、释放、实际撤离和重抓，不能抬起后再盲目重试交接。</li><li>带载入篮保留瓶体中心和直立轴，允许绕瓶轴尝试小角度替代朝向；先检查完整IK/碰撞路径，再实际执行。两例旧失败用5°替代完成。</li><li>真机客户端要求包含不确定度的实测区域证据；支撑未知不释放，终止状态不继续，超时后不假报零运动。</li></ol><p>同一已登记直立瓶型；位置、全周轴向朝向、篮位±23 mm、质量/摩擦×0.8—1.2、一次命令丢失或合成高度误差。没有训练新的神经权重；开发模型只调用1次，3620 tokens。</p></section>'
    body+='<section><h2>最终版本逐例结果</h2><details><summary>展开48个条件的关键参数与关键帧</summary><table><tr><th>集合</th><th>案例</th><th>条件</th><th>结果</th><th>左手下沉</th><th>历史错误与恢复</th><th>图像</th></tr>'+''.join(cases)+'</table></details></section>'
    body+=f'<p><a href="summary.json">统计JSON</a> · <a href="protocol.json">第一轮协议</a> · <a href="refine/protocol.json">第二轮协议</a> · <a href="model-review.json">开发诊断</a> · <a href="{source}">NVIDIA ASPIRE</a></p></main></html>'
    (OUT/'index.html').write_text(body)
    print(json.dumps({k:v for k,v in summary.items() if k not in [c[0] for c in comparisons]},ensure_ascii=False))


if __name__=='__main__':main()
