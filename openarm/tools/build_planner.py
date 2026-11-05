"""Export the active MuJoCo robot's kinematics for cuRobo; not physical calibration."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.environment import OpenArmEnv

ROOT=Path(__file__).resolve().parents[1]


def vector(value):
    return ' '.join(format(float(v),'.12g') for v in value)


def rpy(quat):
    matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,quat);r=matrix.reshape(3,3)
    return [np.arctan2(r[2,1],r[2,2]),np.arctan2(-r[2,0],np.hypot(r[2,1],r[2,2])),np.arctan2(r[1,0],r[0,0])]


def build():
    folder=ROOT/'task/planners';folder.mkdir(parents=True,exist_ok=True)
    env=OpenArmEnv(images=False);env.reset(seed=0);m=env.model
    robot=ET.Element('robot',name='openarm_sim');ET.SubElement(robot,'link',name='world')
    bodies=[i for i in range(1,m.nbody) if (m.body(i).name or '').startswith('openarm_')]
    names={i:m.body(i).name for i in bodies};names[0]='world';joint_names=[];defaults=[];spheres={};ignore={}
    try:
        for i in bodies:
            link=ET.SubElement(robot,'link',name=names[i]);inertia=ET.SubElement(link,'inertial')
            ET.SubElement(inertia,'origin',xyz=vector(m.body_ipos[i]),rpy=vector(rpy(m.body_iquat[i])))
            ET.SubElement(inertia,'mass',value=str(max(float(m.body_mass[i]),1e-6)))
            values=m.body_inertia[i];ET.SubElement(inertia,'inertia',ixx=str(values[0]),iyy=str(values[1]),izz=str(values[2]),ixy='0',ixz='0',iyz='0')
            ids=np.flatnonzero(m.jnt_bodyid==i)
            if len(ids)>1 or (len(ids) and np.linalg.norm(m.jnt_pos[ids[0]])>1e-9):raise ValueError('unsupported joint origin')
            j=int(ids[0]) if len(ids) else None
            name=m.joint(j).name if j is not None else names[i]+'_fixed'
            kind='fixed' if j is None else 'prismatic' if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE else 'revolute'
            joint=ET.SubElement(robot,'joint',name=name,type=kind)
            ET.SubElement(joint,'parent',link=names[int(m.body_parentid[i])]);ET.SubElement(joint,'child',link=names[i])
            ET.SubElement(joint,'origin',xyz=vector(m.body_pos[i]),rpy=vector(rpy(m.body_quat[i])))
            if j is not None:
                ET.SubElement(joint,'axis',xyz=vector(m.jnt_axis[j]));ET.SubElement(joint,'limit',lower=str(m.jnt_range[j,0]),upper=str(m.jnt_range[j,1]),effort='40',velocity='2')
                joint_names.append(name);defaults.append(float(env.data.qpos[m.jnt_qposadr[j]]))
            # Sparse vertex proposal spheres (not a surface enclosure). Native mesh rechecks are
            # mandatory before execution; spheres are not an exact contact model.
            points=[]
            for g in np.flatnonzero(m.geom_bodyid==i):
                if not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
                mesh=int(m.geom_dataid[g]);verts=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
                matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,m.geom_quat[g])
                points.extend(verts@matrix.reshape(3,3).T+m.geom_pos[g])
            if points:
                vox=.035;cells=np.unique(np.floor(np.array(points)/vox).astype(int),axis=0)
                spheres[names[i]]=[{'center':((p+.5)*vox).tolist(),'radius':.005} for p in cells]
            neighbors=[]
            for other in bodies:
                a,b=int(m.body_weldid[i]),int(m.body_weldid[other])
                related=(a==b or int(m.body_weldid[m.body_parentid[a]])==b
                         or int(m.body_weldid[m.body_parentid[b]])==a
                         or ((min(i,other)<<16)+max(i,other)) in m.exclude_signature)
                if other!=i and related:neighbors.append(names[other])
            ignore[names[i]]=neighbors
        for side in ('left','right'):
            link=side+'_candidate_tcp';ET.SubElement(robot,'link',name=link)
            j=ET.SubElement(robot,'joint',name=side+'_candidate_tcp_fixed',type='fixed')
            ET.SubElement(j,'parent',link='openarm_'+side+'_hand');ET.SubElement(j,'child',link=link)
            ET.SubElement(j,'origin',xyz='0 0.006 0.075',rpy='0 0 0')
        ET.indent(robot);ET.ElementTree(robot).write(folder/'openarm.urdf',encoding='unicode',xml_declaration=True)
        config={'robot_cfg':{'kinematics':{'format_version':2.,'urdf_path':str(folder/'openarm.urdf'),'base_link':'world',
            'tool_frames':['right_candidate_tcp'],'collision_link_names':list(spheres),'collision_spheres':spheres,
            'collision_sphere_buffer':0.,'self_collision_buffer':{k:0. for k in spheres},'self_collision_ignore':ignore,
            'cspace':{'joint_names':joint_names,'default_joint_position':defaults,'null_space_weight':[1.]*len(defaults),
                      'cspace_distance_weight':[1.]*len(defaults),'max_acceleration':5.,'max_jerk':40.}}},
            'scope':'derived simulator kinematics; candidate limits/sparse proposal spheres, mandatory native collision recheck; not measured hardware'}
        (folder/'openarm.json').write_text(json.dumps(config,indent=2)+'\n')
        print(json.dumps({'robot_bodies':len(bodies),'joints':len(joint_names),'spheres':sum(map(len,spheres.values())),'output':str(folder)}))
    finally:env.close()


if __name__=='__main__':build()
