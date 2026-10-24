"""Measured pose/hold evidence. Pure numerical code; never opens a device."""
from collections import deque
import math
import time
import xml.etree.ElementTree as ET

import numpy as np


def finite(value, shape):
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('invalid finite shape '+str(shape))
    return a


def transform(value):
    t=finite(value,(4,4));r=t[:3,:3]
    if not np.allclose(t[3],[0,0,0,1],atol=1e-7) or not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1,atol=1e-5):
        raise ValueError('invalid rigid transform')
    return t


def fresh(stamp,now,maximum=.2):
    return type(stamp) in (float,int) and math.isfinite(stamp) and -.01<=now-stamp<=maximum


def relative_target(base_object,object_tool):
    """T_base_tool = T_base_object @ T_object_tool; metres, right handed."""
    return transform(base_object)@transform(object_tool)


class Kinematics:
    """URDF joint names, never assumed solver order or quaternion ordering."""
    def __init__(self,path):
        root=ET.parse(path).getroot();self.joints={};children=set()
        for j in root.findall('joint'):
            parent=j.find('parent').get('link');child=j.find('child').get('link');children.add(child)
            origin=j.find('origin');xyz=np.zeros(3);rpy=np.zeros(3)
            if origin is not None:
                xyz=finite([float(v) for v in origin.get('xyz','0 0 0').split()],(3,))
                rpy=finite([float(v) for v in origin.get('rpy','0 0 0').split()],(3,))
            r,p,y=rpy;cx,sx=np.cos(r),np.sin(r);cy,sy=np.cos(p),np.sin(p);cz,sz=np.cos(y),np.sin(y)
            rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]);ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
            rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
            t=np.eye(4);t[:3,:3]=rz@ry@rx;t[:3,3]=xyz
            axis=j.find('axis');axis=np.array([float(v) for v in (axis.get('xyz') if axis is not None else '1 0 0').split()])
            limit=j.find('limit');mimic=j.find('mimic')
            self.joints[child]={'name':j.get('name'),'parent':parent,'type':j.get('type'),'origin':t,'axis':axis,
                'limits':None if limit is None else [float(limit.get('lower','-inf')),float(limit.get('upper','inf'))],
                'mimic':None if mimic is None else (mimic.get('joint'),float(mimic.get('multiplier','1')),float(mimic.get('offset','0')))}
        roots={j['parent'] for j in self.joints.values()}-children
        if len(roots)!=1:raise ValueError('ambiguous URDF root')
        self.root=roots.pop()

    def pose(self,link,values):
        if link==self.root:return np.eye(4)
        j=self.joints[link];t=j['origin'].copy();q=0.
        if j['type']!='fixed':
            if j['mimic']:
                name,mul,offset=j['mimic'];q=float(values[name])*mul+offset
            else:q=float(values[j['name']])
            if not math.isfinite(q):raise ValueError('nonfinite joint')
            a=finite(j['axis'],(3,));a=a/np.linalg.norm(a);motion=np.eye(4)
            if j['type'] in ('revolute','continuous'):
                x,y,z=a;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                motion[:3,:3]=np.eye(3)+np.sin(q)*k+(1-np.cos(q))*(k@k)
            elif j['type']=='prismatic':motion[:3,3]=a*q
            else:raise ValueError('unsupported joint type')
            t=t@motion
        return self.pose(j['parent'],values)@t


class HoldEvidence:
    """Jaw contact effort + measured object in jaw + stable relative pose.

    All thresholds must come from a separately accepted real calibration. A
    closed jaw alone, a commanded grasp or an occluded object is never a hold.
    """
    def __init__(self,config):
        self.config=config;self.rows={side:deque() for side in ('left','right')}

    def read(self,side,*,object_pose,tool_pose,feedback,stamp,now=None):
        now=time.time() if now is None else now;c=self.config[side]
        if not fresh(stamp,now) or not fresh(feedback.get('timestamp_unix_s'),now):
            self.rows[side].clear();return {'verified':False,'held':False,'reason':'stale evidence'}
        obj=transform(object_pose);tool=transform(tool_pose);rel=np.linalg.inv(tool)@obj
        point=rel[:3,3];bounds=finite(c['object_center_bounds_tool_m'],(2,3))
        effort=abs(float(feedback['torque']));jaw=float(feedback['position'])
        if not all(math.isfinite(v) for v in [effort,jaw]):raise ValueError('invalid jaw feedback')
        touching=bool(np.all(point>=bounds[0]) and np.all(point<=bounds[1]) and
            c['contact_motor_range'][0]<=jaw<=c['contact_motor_range'][1] and
            c['contact_torque_range'][0]<=effort<=c['contact_torque_range'][1])
        rows=self.rows[side]
        # Distinct, ordered samples with a measured upper gap bound. Repeated
        # cached frames cannot accumulate contact duration.
        if rows and (stamp<=rows[-1][0] or stamp-rows[-1][0]>.075):rows.clear()
        if not touching:rows.clear()
        if touching:rows.append((stamp,rel.copy()))
        while len(rows)>1 and stamp-rows[1][0]>.15:rows.popleft()
        seconds=stamp-rows[0][0] if rows else 0.
        trans=max([np.linalg.norm(r[:3,3]-rel[:3,3]) for _,r in rows],default=0.)
        angle=max([math.acos(float(np.clip((np.trace(r[:3,:3].T@rel[:3,:3])-1)/2,-1,1))) for _,r in rows],default=0.)
        held=touching and seconds>=.1 and trans<=c['maximum_slip_m'] and angle<=c['maximum_rotation_rad']
        return {'verified':True,'held':bool(held),'touching':touching,'contact_seconds':seconds,
            'translation_span_m':float(trans),'rotation_span_rad':float(angle),
            'source':'measured_geometry_and_motor_torque','timestamp_unix_s':stamp,
            'qualification':'candidate inference; independent grip sensor/real slip trial still required'}


def grasp_region(base_bottle,base_tool,*,height_m,body_bottom_m,pad_points_tool,uncertainty_m):
    if not 0<height_m or not 0<=uncertainty_m<height_m/2:raise ValueError('invalid body uncertainty')
    points=np.asarray(pad_points_tool,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or not np.isfinite(points).all():
        raise ValueError('full pad extent vertices required')
    rel=np.linalg.inv(transform(base_bottle))@transform(base_tool)
    z=np.array([(rel@np.r_[p,1])[2] for p in points])-body_bottom_m
    return {'verified':True,'source':'measured_geometry','part':'body',
        'fraction_min':float((z.min()-uncertainty_m)/height_m),
        'fraction_max':float((z.max()+uncertainty_m)/height_m)}
