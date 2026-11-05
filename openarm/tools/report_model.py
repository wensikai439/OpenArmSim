"""Report frozen software model identity and native contact diagnostics."""
import json
from pathlib import Path
import tempfile
import os

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/20261008-model'


def load(name):return json.loads((OUT/name).read_text())


def main():
    with tempfile.TemporaryDirectory(prefix='openarm-model-plot-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,
                             'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        result=load('mechanics/summary.json');rows=result['samples']
        t=np.asarray([s['sim_seconds'] for s in rows]);speed=[];util=[]
        for row in rows:
            cs=[c for c in row['contacts'] if 'finger' in c['other_body']]
            speed.append(np.median([c['tangential_speed_m_s'] for c in cs])*1000 if cs else np.nan)
            util.append(np.median([c['tangential_friction_utilization'] for c in cs]) if cs else np.nan)
        fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
        axes[0].plot(t,speed,color='#967153',lw=1.2);axes[0].set_ylabel('接触切向速度 / mm/s（对数轴）');axes[0].set_yscale('log')
        axes[1].plot(t,util,color='#637f76',lw=1.2);axes[1].axhline(1,color='#b87556',ls='--',lw=1,label='切向摩擦圆边界')
        axes[1].set_ylabel('切向摩擦利用率');axes[1].set_xlabel('原生仿真时间 / s');axes[1].legend(frameon=False)
        axes[0].set_title('同一真实 API 失败轨迹 · 接触力与滑动诊断（30 Hz 采样）',loc='left')
        axes[1].set_xlim(0,t[-1])
        last_contact=max(time for time,value in zip(t,speed) if np.isfinite(value))
        axes[0].text((last_contact+t[-1])/2,.45,'无夹指接触\n不绘制速度／利用率',
                     transform=axes[0].get_xaxis_transform(),ha='center',fontsize=8,color='#777166')
        for ax in axes:
            ax.grid(alpha=.15)
            for lo,hi,label in [(47/30,319/30,'抓取'),(871/30,1221/30,'cuRobo 持瓶运动')]:
                ax.axvspan(lo,hi,color='#b7bfb5',alpha=.18)
                if ax is axes[0]:ax.text((lo+hi)/2,.96,label,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
        fig.savefig(OUT/'contacts.png',dpi=180);plt.close(fig)
        dims=load('dimensions-final/summary.json');position=[];rotation=[]
        for row in dims:
            a=row['grasp_stability']['arms']['right'];position.append(a['max_window_relative_drift_m']*1000);rotation.append(a['max_window_relative_rotation_rad']*1000)
        fig,axes=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
        for ax,values,label,limit in [(axes[0],position,'2秒窗口最大位置范围 / mm',1),(axes[1],rotation,'2秒窗口最大转动范围 / mrad',20)]:
            ax.bar(['3维原方案','4维反事实','6维反事实'],values,color=['#a4866c','#78928a','#93a3a1'],width=.58)
            ax.axhline(limit,color='#b87556',ls='--',lw=1);ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
            for i,v in enumerate(values):ax.text(i,v+.02*max(values),f'{v:.2f}',ha='center')
            ax.set_ylim(0,max(max(values),limit)*1.22)
        axes[0].set_title('相同抓取状态后的10秒保持 · 原阈值不变',loc='left')
        fig.savefig(OUT/'dimensions.png',dpi=180);plt.close(fig)
    model=load('model.json');verification=load('verification.json')
    compact={k:v for k,v in result.items() if k not in ('samples','records')}
    (OUT/'results.json').write_text(json.dumps({'model_verification':verification,'mechanics':compact,'contact_dimensions':dims},ensure_ascii=False,indent=2)+'\n')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>模型与接触审计</title>
<style>body{margin:0;background:#f7f6f2;color:#332f28;font:15px/1.7 system-ui,sans-serif}main{max-width:1060px;margin:32px auto;padding:0 24px}h1{font-size:30px;line-height:1.3}h2{font-size:19px}section{background:#fff;border:1px solid #e6e1d7;border-radius:15px;padding:24px;margin:18px 0}img{width:100%;height:auto}a{color:#806046;text-underline-offset:3px}.muted,small{color:#777166}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #eae6dd}.scroll{overflow:auto}details{margin-top:14px}code{font-size:12px;overflow-wrap:anywhere}</style>
<main><a href="/">← OpenArm Lab</a><p class="muted">2026-10-08 · 候选模型的可复核证据</p><h1>模型身份补齐，持续夹持仍未通过</h1><p>这一轮补齐软件模型的证据，并复现已有失败来检查接触。没有修改生产质量、摩擦、夹持力、增益、求解器、碰撞或成功门槛。</p>
<section><h2>实际加载什么，现在能核对</h2><p>当前 v1 模型冻结了 <strong>48份依赖文件：3份MJCF和45份网格文件</strong>。59个文件型编译资源使用包含同一网格的镜像或重复引用。冻结字节经 MuJoCo VFS 加载，文件内容与加载内容绑定；编译后 MJB 摘要覆盖几何、惯量、约束、执行器、相机、求解器和过程纹理。</p><p>新加载方式的完整编译模型摘要与修改前一致：<code>3f82ee0029404c39f4e3ad4a3cec67078a98fb877358643462ab935df6f6ac2b</code>。新增小型运行参数审计，增益、接触维数或求解器被改变后会留下差异并禁止普通实验开跑；控制器动态前馈单独记录。大网格／纹理编译缓存只在初始快照识别，不连续重哈希，主运行接口没有修改这些缓存的工具。</p><p>同尺寸文件修改且恢复mtime仍会被检测；缺失文件、外部依赖和冲突VFS名称不能静默通过。20次热文件缓存条件下manifest读取平均约3.61ms，不代表整页延迟。<strong>这些证据证明软件版本和参数可追溯，不能证明实物已经标定。</strong></p><a href="model.json">完整模型来源</a> · <a href="baseline.json">修改前快照</a> · <a href="verification.json">一致性与开销</a> · <a href="model-checks.log">针对性检查</a></section>
<section><h2>失败不是简单的夹持力不足</h2><img src="contacts.png" alt="实际API失败轨迹的接触滑动速度与切向摩擦利用率"><p>回放最终那轮18次API失败的全部1903个控制步，机器人状态和瓶子终点均与归档一致。833个右手仍被判夹持的等待采样中，夹指接触实际全为3维；单接触法向力中位数9.53N，切向摩擦利用率中位数0.32，切向速度中位数0.74mm/s。摩擦利用率低于1仍在慢滑，不能直接归因于法向力不足。</p><p>110个采样的放宽接触力矩空间只有5维，其余723个为6维。这个瞬时几何检查忽略摩擦及力上限，只给必要条件；6维不能证明力闭合或稳定。这也不足以把脱手归因于唯一原因，后续仍需验证接触表示和实际夹爪材料。</p><p>MuJoCo根据接触几何优先级决定实际condim；瓶身写4维不代表夹指接触采用4维。当前夹指优先级1高于瓶子的0。还需纠正术语：原生碰撞复核覆盖原生几何，但普通网格接触采用其凸包，不能声称完成了逐三角形凹表面碰撞检测。</p><a href="mechanics/protocol.json">回放协议</a> · <a href="mechanics/summary.json">逐采样接触、力和速度</a> · <a href="../20261008-alignment/api-terminal.json">原始API失败</a></section>
<section><h2>接触模型的隔离对照，三项均未通过</h2><img src="dimensions.png" alt="三种接触维数的2秒窗口位置和转动范围"><p>从同一已重现的抓取状态开始，只改变夹指有效接触维数，再保持原执行器目标10秒。4维或6维可减小部分窗口转动，但位置范围仍超过1mm，且三种都发生过接触中断。它们是未实测的物理模型反事实，不是正式模型，也没有自动应用。</p><p>最初尝试通过普通Recorder记录反事实时，新增参数保护正确拒绝了4维模型；原失败日志保留。最终对照另存明确的研究档案，没有绕过或放宽生产保护。</p><a href="dimensions-final/protocol.json">固定对照协议</a> · <a href="dimensions-final/summary.json">全部结果</a> · <a href="dimensions.log">首次保护拒绝记录</a></section>
<section><h2>验收范围和后续缺项</h2><p><strong>175项回归通过</strong>，含冻结资产、同尺寸网格修改、编译参数变化、运动／夹持／交接、记忆、API结束与失败报告检查。新增完整模型身份进入记忆兼容判断；旧记录没有采集时网格快照，不强行迁移成当前模型的已验证经验。偏好保留，原始实验和旧精炼证据仍在归档。</p><p>仍缺独立基座／TCP、相机内外参和时间同步、夹爪实际开口／传动／力、指面材料、瓶子尺寸质量以及实物接触实验。本轮未访问SSH、未操作真机，持续持瓶、开放物体泛化和real-to-sim-to-real目标仍未完成。</p><a href="regression-final.log">全部回归日志</a> · <a href="audit.json">部署审计</a> · <a href="results.json">结构化结果</a></section>
<small>原理参考 <a href="https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters">MuJoCo接触优先级</a>、<a href="https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip">慢滑与摩擦维数</a>、<a href="https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection">凸包碰撞</a>。本轮没有把仿真诊断当作真实接触材料测量。</small></main></html>'''
    (OUT/'index.html').write_text(html)


if __name__=='__main__':main()
