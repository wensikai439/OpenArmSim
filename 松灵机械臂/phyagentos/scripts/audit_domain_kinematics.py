"""Offline URDF/MJCF and recorded TCP audit; mj_forward only, no live runtime."""
import os
import argparse,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def axis_rotation(axis,angle):
    v=np.asarray(axis,dtype=float); v=v/np.linalg.norm(v)
    x,y,z=v; K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)

def origin(j):
    e=j.find("origin"); xyz=np.fromstring(e.get("xyz","0 0 0"),sep=" ")
    r,p,y=np.fromstring(e.get("rpy","0 0 0"),sep=" ")
    T=np.eye(4);T[:3,3]=xyz
    T[:3,:3]=axis_rotation([0,0,1],y)@axis_rotation([0,1,0],p)@axis_rotation([1,0,0],r)
    return T

def fk(joints,q):
    T=np.eye(4); chain=[]
    for j,a in zip(joints,q):
        R=np.eye(4);R[:3,:3]=axis_rotation(np.fromstring(j.find("axis").get("xyz"),sep=" "),a)
        T=T@origin(j)@R;chain.append(T.copy())
    return chain

def angle(R):
    return float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))

def quat_matrix(q):
    x,y,z,w=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    P=Path(str(Path(os.environ.get("PAOS_PROJECTS_ROOT", str(Path(__file__).resolve().parents[1] / "external/projects")))))
    urdf=P/"piper_mujoco/assets/piper.urdf"; mjcf=P/"piper_robocasa/robot_assets/piper_dual.xml"
    joints=[ET.parse(urdf).getroot().find(f"./joint[@name='joint{i}']") for i in range(1,7)]
    m=mujoco.MjModel.from_xml_path(str(mjcf));d=mujoco.MjData(m)
    recorded=np.load(args.samples,allow_pickle=False)
    state=recorded["observation.state"]
    limits=np.array([[float(j.find("limit").get("lower")),float(j.find("limit").get("upper"))] for j in joints])
    rng=np.random.default_rng(42)
    tests=[np.zeros(6),(limits[:,0]+limits[:,1])/2,*rng.uniform(limits[:,0],limits[:,1],size=(12,6))]
    report={"thresholds_fixed_before_execution":{"position_m":1e-6,"rotation_rad":1e-5},
            "offline_only":True,"physics_steps":0,"live_runtime_accessed":False,
            "sources_sha256":{str(p):sha(p) for p in (urdf,mjcf,args.samples,Path(__file__))},
            "recorded_samples":len(state),"recorded_sample_selection":"first/middle/last of every real episode",
            "urdf_mjcf":{},"recorded_tcp":{},
            "physical_hardware_zero_calibrated":False,
            "recorded_tcp_limitation":"Firmware-derived TCP and temporally aligned feedback; no independent measured pose or physical encoder-zero calibration."}
    for side,start in (("left",0),("right",7)):
        ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"{side}_joint{i}") for i in range(1,7)]
        adr=m.jnt_qposadr[ids];refs=m.qpos0[adr].copy()
        base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f"{side}_base")
        links=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f"{side}_link{i}") for i in range(1,7)]
        arm_tests=[*tests,refs,*state[:,start:start+6]]
        for name,offset in (("qpos_equals_urdf",np.zeros(6)),("qpos_equals_urdf_plus_ref",refs),("qpos_equals_urdf_minus_ref",-refs)):
            errors=[]
            for q in arm_tests:
                mujoco.mj_resetData(m,d);d.qpos[adr]=q+offset;mujoco.mj_forward(m,d)
                Rb=d.xmat[base].reshape(3,3);pb=d.xpos[base]
                for body,T in zip(links,fk(joints,q)):
                    R=Rb.T@d.xmat[body].reshape(3,3);p=Rb.T@(d.xpos[body]-pb)
                    errors.append((np.linalg.norm(p-T[:3,3]),angle(T[:3,:3].T@R)))
            e=np.asarray(errors);mx=e.max(0)
            report["urdf_mjcf"][f"{side}/{name}"]={"ref_rad":refs.tolist(),"cases":len(arm_tests),
               "all_six_links_compared":True,"max_position_error_m":float(mx[0]),"max_rotation_error_rad":float(mx[1]),
               "passed":bool(mx[0]<=1e-6 and mx[1]<=1e-5)}
        tcp=recorded[f"observation.tcp_{side}"];translations=[];rotations=[];flange_err=[]
        for q,t in zip(state[:,start:start+6],tcp):
            T=fk(joints,q)[-1];R=quat_matrix(t[3:7])
            translations.append(T[:3,:3].T@(t[:3]-T[:3,3]))
            rotations.append(T[:3,:3].T@R)
            flange_err.append(np.linalg.norm(t[:3]-T[:3,3]))
        tr=np.asarray(translations);rot=np.asarray(rotations)
        center=np.median(tr,axis=0);dist=np.linalg.norm(tr-center,axis=1)
        report["recorded_tcp"][side]={"flange_position_error_quantiles_m":np.quantile(flange_err,[0,.5,.95,1]).tolist(),
          "tcp_in_flange_translation_median_m":center.tolist(),"translation_residual_quantiles_m":np.quantile(dist,[0,.5,.95,1]).tolist(),
          "tcp_vs_flange_rotation_quantiles_rad":np.quantile([angle(R) for R in rot],[0,.5,.95,1]).tolist(),
          "no_transform_fitted_or_applied":True}
    report["software_qpos_equals_urdf"]=all(report["urdf_mjcf"][f"{s}/qpos_equals_urdf"]["passed"] for s in ("left","right"))
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"report":str(args.out),"software_qpos_equals_urdf":report["software_qpos_equals_urdf"],
                     "identity":{s:report["urdf_mjcf"][f"{s}/qpos_equals_urdf"] for s in ("left","right")},
                     "recorded_tcp":report["recorded_tcp"]}),flush=True)
    return 0 if report["software_qpos_equals_urdf"] else 2

if __name__=="__main__":raise SystemExit(main())
