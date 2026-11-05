"""Curate measured evidence and publish only a separately verified robot skill."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def publish(folder):
    result=load(folder/'result.json')
    if result.get('complete') is not True:raise ValueError('campaign is incomplete')
    frozen=load(folder/'frozen.json');program=folder/'skill/program.py'
    if sha(program)!=frozen['program_sha256']:raise ValueError('frozen program changed')
    held=load(folder/'heldout/results.json');baseline=load(folder/'baseline/results.json')
    if len(held)!=20 or len(baseline)!=20:raise ValueError('paired coverage is incomplete')
    if [r['case'] for r in held]!=[r['case'] for r in baseline]:raise ValueError('paired fixtures differ')
    if any(r['program_sha256']!=frozen['program_sha256'] for r in held):
        raise ValueError('evaluation mixed different generated programs')
    verification=load(folder/'verification.json')
    skill=folder/'skill/SKILL.md'
    admitted=(verification['verdict']=='verified' and not verification['proposer_code_provided']
              and skill.exists() and result['heldout']['passes']>=16)
    stress_path=folder/'wide-result.json'
    stress=load(stress_path) if stress_path.exists() else None
    scope={'object':'same registered upright black bottle','task':'right pick -> right-to-left transfer -> left basket placement',
           'position_domain_m':load(folder/'protocol.json')['bottle_workspace_m'],
           'oracle_geometry':True,'vision_only':False,'hardware_access':False,'transfer_ready':False,
           'recovery_verified':False,
           'contribution':'model-generated state checks and validated composition; geometric generalization inherited from TAMP',
           'success_gain_over_baseline':result['heldout']['passes']-result['baseline']['passes']}
    manifest={'name':'bottle-relay','admitted':admitted,'program_sha256':sha(program),
              'campaign':str(folder.relative_to(ROOT)), 'upstream_commit':load(folder/'protocol.json')['upstream_commit'],
              'measured':{k:result[k] for k in ('development','unseen','heldout','baseline','verification')},
              'wide_position_test':stress,'scope':scope,
              'baseline_label_correction':'original baseline is existing parameterized geometry tools, NOT fixed trajectory',
              'skill_text_correction':'Only v01 existed; proposal mentions of an earlier version do not establish a version comparison.'}
    dump(folder/'assessment.json',manifest)
    if admitted:
        target=ROOT/'skills/bottle_relay'
        if target.exists() and (sha(target/'program.py')!=sha(program)
            or load(target/'manifest.json')['campaign']!=str(folder.relative_to(ROOT))):
            raise ValueError('another admitted skill already exists; do not overwrite it')
        target.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(program,target/'program.py')
        prefix=('---\nname: bottle-relay\ndescription: Validated OpenArm simulator bottle relay using geometry-oracle TAMP tools.\n---\n\n'
            '# 抓瓶、交接与入篮\n\n'
            f'冻结最终测试：{result["heldout"]["passes"]}/20；相同工具基线：{result["baseline"]["passes"]}/20。'
            +(f'扩展16位置：生成程序{stress["skill"]["passes"]}/16，参数化工具{stress["geometry"]["passes"]}/16，固定轨迹{stress["fixed"]["passes"]}/16。' if stress else '')+
            '本轮没有证明技能凝练提高成功率。当前只覆盖相同直立瓶子的位置变化，使用仿真几何。未验证真机迁移、视觉定位或失败恢复。\n\n'
            '以下是 Haiku 从开发证据凝练的说明。独立上下文成功复写代表方法能够传达；'
            '未发生的故障分支只属于程序保护，不属于已经学会或实测过的恢复技能。'
            '原文中的单次坐标和步数不得用作新场景的动作目标；文末“earlier version”没有独立版本对照证据。\n\n')
        (target/'SKILL.md').write_text(prefix+skill.read_text())
        dump(target/'manifest.json',manifest)
    # Select one full, successful unseen-position run for the compact stage strip.
    witness=next((r for r in held if r.get('task_success')),held[0])
    media=Path(witness['trial'])/'media'
    gallery=folder/'gallery';gallery.mkdir(exist_ok=True)
    frames=[('初始场景','initial-overview.jpg'),('右手抓取','01-after-pick-overview.jpg'),
            ('左手接收','02-after-handover-overview.jpg'),('释放入篮','final-overview.jpg')]
    cards=[]
    for i,(label,name) in enumerate(frames):
        if (media/name).exists():
            shutil.copyfile(media/name,gallery/f'{i}.jpg')
            cards.append(f'<figure><a href="gallery/{i}.jpg"><img src="gallery/{i}.jpg" alt="{label}"></a><figcaption>{label}</figcaption></figure>')
    points=[]
    for row in held:
        x,y=row['case']['bottle_xy_m'];px=70+(x-.40)/.20*560;py=310-(y+.30)/.22*260
        color='#64856b' if row.get('task_success') else '#b16d4c'
        points.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5" fill="{color}"><title>seed {row["case"]["seed"]}: X={x:.4f}, Y={y:.4f}, {row.get("task_success")}</title></circle>')
    if stress:
        for row in load(folder/'wide/skill/results.json'):
            x,y=row['case']['bottle_xy_m'];px=70+(x-.40)/.20*560;py=310-(y+.30)/.22*260
            color='#64856b' if row.get('task_success') else '#b16d4c'
            points.append(f'<rect x="{px-4:.2f}" y="{py-4:.2f}" width="8" height="8" fill="{color}"><title>boundary X={x}, Y={y}: {row.get("task_success")}</title></rect>')
    stats=([('固定轨迹',stress['fixed']),('已有参数化工具',stress['geometry']),('Haiku 生成 skill',stress['skill'])]
           if stress else [('开发位置',result['development']),('独立检查',result['unseen']),
                           ('局部冻结测试',result['heldout']),('参数化工具基线',result['baseline'])])
    tiles=''.join(f'<div><strong>{s["passes"]}/{s["total"]}</strong><span>{label}</span></div>' for label,s in stats)
    delta=scope['success_gain_over_baseline']
    comparison=('两者成功数相同，本轮没有证明技能凝练提高了成功率。位置适应主要来自已有参数化 TAMP。'
                if delta==0 else f'在相同20个位置上，生成程序比基线多成功{delta}次。这是小样本配对结果。')
    boundary=(f'<p>18×20厘米扩展区域：生成程序 {stress["skill"]["passes"]}/{stress["skill"]["total"]}，'
              f'参数化工具直调用 {stress["geometry"]["passes"]}/{stress["geometry"]["total"]}，'
              f'固定轨迹 {stress["fixed"]["passes"]}/{stress["fixed"]["total"]}。方形为扩展测试点，失败全部保留。'
              f'原局部20位置的固定轨迹对照为{stress["fixed_original_heldout"]["passes"]}/20。</p>' if stress else '')
    if stress:
        for row in load(folder/'wide/skill/results.json'):
            if row.get('task_success'):continue
            xy=row['case']['bottle_xy_m'];relative=Path(row['trial']).relative_to(folder).as_posix()
            boundary+=f'<p>保留失败：X={xy[0]:.2f}、Y={xy[1]:.2f}米，{html.escape(str(row.get("failure_stage")))}阶段返回{html.escape(str(row.get("failure_code")))}。<a href="{relative}/result.json">原始失败记录</a></p>'
    attempts=load(folder/'attempts.json')
    search_note=('首次程序已通过全部开发位置，本轮没有实际修复失败的迭代，也没有验证新的失败恢复策略。'
                 if len(attempts)==1 and result['development']['passes']==5 else
                 f'本轮共测试{len(attempts)}版程序，选取开发位置成功数最多的版本；失败记录保留。')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ASPIRE 抓瓶技能 · OpenArm Lab</title>
<style>:root{color-scheme:light;background:#f7f5f0;color:#302e29;font:16px/1.75 system-ui,sans-serif}body{max-width:1040px;margin:auto;padding:48px 24px}h1{font-size:30px;line-height:1.4;letter-spacing:-.7px}h2{font-size:20px;margin:32px 0 12px}p{max-width:880px}.muted,small{color:#777266}a{color:#94634a}section{background:#fffefb;border:1px solid #e7e2d8;border-radius:18px;padding:24px;margin:20px 0}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.stats strong{font-size:32px;display:block;line-height:1.4}.stats span,figcaption{font-size:14px;color:#777266}.gallery{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}img{width:100%;border-radius:10px}svg{width:100%;max-width:700px;display:block}code{background:#eeebe4;padding:3px 6px;border-radius:4px}li{margin:6px 0}@media(max-width:650px){body{padding:24px 16px}.stats,.gallery{grid-template-columns:repeat(2,1fr)}section{padding:18px}}</style>
<small>OPENARM LAB · NVIDIA ASPIRE ADAPTATION</small><h1>右手抓瓶 → 左手接收 → 放入篮子</h1>
<p>位置适应来自参数化规划，本轮没有证明 Haiku 凝练提高了成功率。以下是18×20厘米区域、16个预先声明位置的完整任务对照。</p>
<section><div class="stats">'''+tiles+'''</div></section>
<div class="gallery">'''+''.join(cards)+'''</div>
<h2>泛化验证</h2><p>圆点为20个最终测试位置。绿色为成功，棕色为失败；鼠标悬停可查看坐标。机器人+X向前，+Y向左，单位米。</p>
<section><svg viewBox="0 0 700 370" role="img" aria-label="新瓶子位置与完整任务结果"><path d="M70 35V310H650" fill="none" stroke="#aaa497"/><text x="340" y="355" fill="#777266">瓶子世界 X：0.40 → 0.60 m</text><text x="12" y="24" fill="#777266">Y：-0.30 → -0.08 m</text>'''+''.join(points)+'''</svg>'''+boundary+'''</section>
<p>'''+html.escape(comparison)+'''</p><p>本轮支持相同直立瓶子的局部位置泛化。使用仿真几何定位与规划，不能据此声称视觉泛化、任意位置成功或已完成真机迁移。</p>
<h2>技能是怎样得到的</h2><ol><li>读取初始三路相机图像和工具契约，由你指定的 claude-haiku-5-5 生成程序。</li><li>在开发位置执行，记录每次原语前后的相机帧、原生接触与工具结果。</li><li>固定程序SHA256，以新位置测完整链条；测试结果不回流修改程序。</li><li>只从开发证据写技能说明，再用独立API上下文、不提供获胜程序，复写并验证。</li></ol>
<p>'''+html.escape(search_note)+'''没有更新 ACT 或 π0.5 权重。</p>
<h2>复现与证据</h2><p><code>./run aspire test --x 0.50 --y -0.17</code></p>
<p><a href="assessment.json">验收清单</a> · <a href="result.json">实验结果和 API 用量</a> · <a href="skill/program.py">冻结程序</a> · <a href="skill/SKILL.md">技能说明</a> · <a href="protocol.json">实验协议</a></p></html>'''
    (folder/'index.html').write_text(page)
    print(json.dumps({'admitted':admitted,'paired_success_difference':delta,
                      'report':str(folder/'index.html')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    folder=args.folder.resolve()
    if not folder.is_relative_to(ROOT/'runs'):parser.error('campaign must stay in runs/')
    publish(folder)
