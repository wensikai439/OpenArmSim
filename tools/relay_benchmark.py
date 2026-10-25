"""Freeze/evaluate the complete non-TAMP relay; retain every trial and video."""
import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.visual_benchmark import verify,save


def report(folder,protocol,rows):
    esc=html.escape;passed=sum(r['full_task_success'] for r in rows);cards=[]
    for row in rows:
        name=row['name'];good=row['full_task_success']
        stages=' → '.join(e['name'] for e in row.get('events',[]))
        cards.append(f'''<article><div class="label">{esc(row['label'])}<b class="{'pass' if good else 'fail'}">{'通过' if good else '失败'}</b></div>
        <video controls playsinline preload="metadata" src="{name}/rollout.mp4"></video>
        <p>{row.get('sim_seconds',0):.1f} 秒 · {esc(stages)}</p>
        <details><summary>实验记录</summary><p>{esc(row.get('error') or '原生完整任务评分通过；评分不作为控制程序输入。')}</p>
        <a href="{name}/result.json">物理评分</a> · <a href="{name}/tools.jsonl">工具调用</a> · <a href="{name}/vision/estimated-projection.json">在线投影估计</a></details></article>''')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整视觉 skill · 视频</title><style>
    :root{font-family:Inter,"Noto Sans CJK SC",sans-serif;color:#302f2c;background:#f6f4ef;line-height:1.65}*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:44px auto;padding:0 24px}h1{font-size:32px;margin:10px 0}p{font-size:14px;color:#77716a}.eyebrow{font-size:12px;letter-spacing:2px;color:#a17b60}.metrics{display:flex;gap:18px;margin:25px 0;flex-wrap:wrap}.metric{flex:1;background:#fffefa;border:1px solid #e7e2d9;border-radius:14px;padding:18px 24px}.metric strong{display:block;font-size:26px}.metric span{font-size:13px;color:#83786c}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin-top:26px}article{border:1px solid #e7e2d9;border-radius:15px;background:#fffefa;padding:18px}.label{display:flex;justify-content:space-between;margin-bottom:12px}b{font-size:12px;padding:2px 10px;border-radius:18px}.pass{background:#edf2e9;color:#617252}.fail{background:#f7e8df;color:#a16650}video{display:block;width:100%;border-radius:9px;background:#686f68}a{color:#956d50}details{font-size:13px;color:#85796b}.note{border-left:3px solid #b59b83;padding-left:18px}footer{margin:30px 0}@media(max-width:750px){.grid{grid-template-columns:1fr}}
    </style><main><div class="eyebrow">OPENARM LAB / COMPLETE VISUAL SKILL</div><h1>抓瓶、交接、放入篮子</h1>
    <p>新的完整流程独立执行，旧 TAMP 未参与。程序使用图像目标、机器人反馈和可达性搜索。</p>'''
    seconds=[r.get('sim_seconds',0) for r in rows if r['full_task_success']]
    duration=f'{min(seconds):.0f}–{max(seconds):.0f} 秒' if seconds else '测试中'
    page+=f'<div class="metrics"><div class="metric"><strong>{passed} / {len(rows)}</strong><span>已完成 {len(rows)} / {len(protocol["cases"])} 个预声明测试</span></div><div class="metric"><strong>{duration}</strong><span>完整动作 · 原速视频</span></div><div class="metric"><strong>0</strong><span>旧 TAMP 调用 · 真机操作</span></div></div>'
    page+='<div class="note"><p>通过要求：右手下半瓶身接触、实际抬升、双手交接、右手脱离后左手持瓶、整瓶入篮、双手释放、篮底支撑并稳定至少 1 秒。失败和初始化错误均计入分母。</p></div>'
    page+='<section class="grid">'+''.join(cards)+'</section><footer><details><summary>适用条件与内部反馈</summary><p>已知直立黑瓶和固定白篮。两路固定合成 RGB、虚拟工具标记及机器人 FK 用于在线估计投影；未读取模拟器相机矩阵或物体位姿来控制。瓶身尺寸、接瓶高度和篮深为工程先验。新执行器显式使用模型夹爪全行程，未改质量、摩擦、控制增益或物理评分；不等于已经验证真机夹爪限位。没有通用遮挡处理、任意物体或真机迁移证明。</p></details><p><a href="protocol.json">冻结协议</a> · <a href="summary.json">完整结果</a></p></footer></main></html>'
    (folder/'index.html').write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();folder=a.output.resolve();protocol=json.loads(a.protocol.read_text());verify(protocol)
    if not folder.is_relative_to(ROOT/'runs') or folder.exists():p.error('new output under runs/ required')
    folder.mkdir(parents=True);save(folder/'protocol.json',protocol)
    for name in protocol['sources']:
        t=folder/'source'/name;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes((ROOT/name).read_bytes())
    rows=[]
    for spec in protocol['cases']:
        verify(protocol);name=spec['name'];case=folder/(name+'.json');save(case,spec['case']);start=time.monotonic()
        try:
            proc=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','harness.aspire.visual_relay','--case',str(case),
                '--output',str(folder/name),'--program',str(ROOT/protocol['program'])],cwd=ROOT,
                env={**os.environ,'MUJOCO_GL':'egl'},capture_output=True,text=True,timeout=180)
            out=folder/name;out.mkdir(exist_ok=True);(out/'process.txt').write_text(proc.stdout+proc.stderr)
            result=json.loads((out/'result.json').read_text())
        except Exception as exc:
            result={'full_task_success':False,'error':type(exc).__name__+': '+str(exc)}
            out=folder/name;out.mkdir(exist_ok=True);save(out/'result.json',result)
        rows.append({**spec,**result,'wall_seconds':time.monotonic()-start})
        save(folder/'summary.json',{'passed':sum(r['full_task_success'] for r in rows),'total':len(rows),'cases':rows})
        report(folder,protocol,rows)
        print(json.dumps({'case':name,'pass':result['full_task_success'],'error':result.get('error')},ensure_ascii=False),flush=True)
    verify(protocol)


if __name__=='__main__':main()
