"""Calibration numerics: markers, hand-eye, clock, jaw sweep, grip and acceptance.

Pure functions only. Nothing here opens a camera, a CAN socket or a motor; the
on-site wizard in calib.py collects the inputs and shows every derived value.
"""
import math
import struct
import time
import zlib
import xml.etree.ElementTree as ET

import numpy as np

try:
    from .evidence import finite,transform
except ImportError:
    from evidence import finite,transform


BOTTLE_ID=1;BASKET_ID=2
BOARD={'columns':4,'rows':3,'first_id':10}
# ACT normalizer action min/max for both jaws, widened to include the ACT
# startup jaw targets (-0.898/-0.942). Source: act_latest policy stats.
ACT_JAW_RANGE=(-1.0,.05)
JAW_SOURCE='act_latest normalizer action range [-0.9,0.05] widened to ACT startup jaw targets'


def aruco():
    import cv2
    return cv2,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def grid_board(marker_m,gap_m):
    cv2,dictionary=aruco()
    ids=np.arange(BOARD['first_id'],BOARD['first_id']+BOARD['columns']*BOARD['rows'],dtype=np.int32)
    return cv2.aruco.GridBoard((BOARD['columns'],BOARD['rows']),float(marker_m),float(gap_m),dictionary,ids)


def rigid(r=None,t=(0,0,0)):
    out=np.eye(4)
    if r is not None:out[:3,:3]=r
    out[:3,3]=t;return out


def side_marker(height_m,distance_m):
    """Marker on a vertical face. Object frame: origin on the bottom/floor centre,
    +z up, +x along the marker's outward normal. Printed 'UP' arrow points +z.
    Returns T_marker_object, the form used by devices.marker_poses."""
    # Marker axes in object coordinates: x->+y, y(up)->+z, z(out)->+x.
    r=np.array([[0.,0,1],[1,0,0],[0,1,0]])
    return np.linalg.inv(rigid(r,(distance_m,0,height_m)))


def top_marker(height_m):
    """Marker lying flat on the cap, facing up; yaw is irrelevant for a bottle."""
    return np.linalg.inv(rigid(None,(0,0,height_m)))


# ---------------------------------------------------------------- printing

def png_dpi(image,dpi=300):
    import cv2
    ok,data=cv2.imencode('.png',image)
    if not ok:raise RuntimeError('PNG encoding failed')
    data=data.tobytes();ppm=int(round(dpi/.0254));body=b'pHYs'+struct.pack('>IIB',ppm,ppm,1)
    chunk=struct.pack('>I',9)+body+struct.pack('>I',zlib.crc32(body)&0xffffffff)
    return data[:33]+chunk+data[33:]


def _page(dpi):
    return np.full((int(round(297/25.4*dpi)),int(round(210/25.4*dpi))),255,np.uint8)


def _px(mm,dpi):return int(round(mm/25.4*dpi))


def _text(page,text,x,y,dpi,scale=1.):
    import cv2
    cv2.putText(page,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,dpi/300*scale,0,max(1,int(dpi/150)),cv2.LINE_AA)


def _scale_bar(page,x,y,dpi):
    import cv2
    length=_px(100,dpi);cv2.line(page,(x,y),(x+length,y),0,max(1,int(dpi/100)))
    for mm in range(0,101,10):
        tick=_px(4 if mm%50 else 8,dpi);cv2.line(page,(x+_px(mm,dpi),y),(x+_px(mm,dpi),y-tick),0,max(1,int(dpi/150)))
    _text(page,'100 mm check: measure this bar after printing',x,y+_px(8,dpi),dpi,.8)


def _arrow(page,cx,top,dpi):
    import cv2
    cv2.arrowedLine(page,(cx,top),(cx,top-_px(12,dpi)),0,max(2,int(dpi/100)),tipLength=.35)
    _text(page,'UP',cx+_px(3,dpi),top-_px(5,dpi),dpi,.9)


def marker_sheets(board_mm=40,gap_mm=10,bottle_mm=40,basket_mm=60,dpi=300):
    """Two A4 pages. Sizes are nominal; the operator measures the printout."""
    cv2,dictionary=aruco();pages=[]
    page=_page(dpi);b=grid_board(board_mm/1000,gap_mm/1000)
    width=BOARD['columns']*board_mm+(BOARD['columns']-1)*gap_mm;height=BOARD['rows']*board_mm+(BOARD['rows']-1)*gap_mm
    image=b.generateImage((_px(width,dpi),_px(height,dpi)),marginSize=0,borderBits=1)
    x=(page.shape[1]-image.shape[1])//2;y=_px(40,dpi);page[y:y+image.shape[0],x:x+image.shape[1]]=image
    _text(page,'OpenArm hand-eye board  DICT_4X4_50 ids %d-%d'%(BOARD['first_id'],BOARD['first_id']+11),x,_px(20,dpi),dpi)
    _text(page,'marker %g mm, gap %g mm. Print at 100%% (no fit to page). Keep flat; glue on cardboard.'%(board_mm,gap_mm),x,_px(30,dpi),dpi,.8)
    _scale_bar(page,x,y+image.shape[0]+_px(25,dpi),dpi);pages.append(page)
    page=_page(dpi);_text(page,'OpenArm object markers  DICT_4X4_50. Print at 100%. Keep the UP arrow pointing up.',_px(15,dpi),_px(20,dpi),dpi,.9)
    rows=[('BOTTLE id=%d  %g mm'%(BOTTLE_ID,bottle_mm),BOTTLE_ID,bottle_mm),('BASKET id=%d  %g mm'%(BASKET_ID,basket_mm),BASKET_ID,basket_mm)]
    y=_px(55,dpi)
    for label,ident,size in rows:
        for column in range(2):
            side=_px(size,dpi);marker=cv2.aruco.generateImageMarker(dictionary,ident,side,borderBits=1)
            x=_px(25+column*95,dpi);page[y:y+side,x:x+side]=marker
            _arrow(page,x+side//2,y-_px(4,dpi),dpi);_text(page,label,x,y+side+_px(8,dpi),dpi,.8)
        y+=_px(size+45,dpi)
    _scale_bar(page,_px(25,dpi),y,dpi);pages.append(page)
    return pages


# ---------------------------------------------------------------- poses

def board_pose(frame,board):
    """T_camera_board, RMS reprojection px and marker count; None if unusable."""
    cv2,dictionary=aruco()
    detector=cv2.aruco.ArucoDetector(dictionary,cv2.aruco.DetectorParameters())
    corners,ids,_=detector.detectMarkers(frame['bgr'])
    if ids is None:return None
    keep=[i for i,v in enumerate(ids.flatten()) if BOARD['first_id']<=v<BOARD['first_id']+12]
    if len(keep)<3:return None
    corners=[corners[i] for i in keep];ids=ids[keep]
    points,pixels=board.matchImagePoints(corners,ids)
    if points is None or len(points)<12:return None
    ok,rvec,tvec=cv2.solvePnP(points,pixels,frame['K'],frame['distortion'],flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2,0]<=0:return None
    projected,_=cv2.projectPoints(points,rvec,tvec,frame['K'],frame['distortion'])
    rms=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-pixels.reshape(-1,2))**2,axis=1))))
    return {'T_camera_board':rigid(cv2.Rodrigues(rvec)[0],tvec[:,0]),'reprojection_rms_px':rms,'markers':len(keep)}


def angle(r):return float(math.acos(float(np.clip((np.trace(r)-1)/2,-1,1))))


def mean_pose(poses):
    """Chordal mean rotation and median translation of nearby rigid poses."""
    from scipy.spatial.transform import Rotation
    poses=[transform(p) for p in poses]
    r=Rotation.from_matrix([p[:3,:3] for p in poses]).mean().as_matrix()
    return rigid(r,np.median([p[:3,3] for p in poses],axis=0))


def spread(poses):
    m=mean_pose(poses)
    return (float(max(np.linalg.norm(p[:3,3]-m[:3,3]) for p in poses)),
            float(max(angle(m[:3,:3].T@p[:3,:3]) for p in poses)))


def solve_hand_eye(base_link,camera_board):
    """Eye-in-hand AX=XB. Returns T_link_camera and the fixed-board consistency."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    a=[transform(p) for p in base_link];b=[transform(p) for p in camera_board]
    if len(a)!=len(b) or len(a)<5:raise ValueError('at least 5 board views at distinct wrist poses required')
    turns=[angle(a[0][:3,:3].T@p[:3,:3]) for p in a[1:]]
    axes=np.array([Rotation.from_matrix(a[0][:3,:3].T@p[:3,:3]).as_rotvec() for p in a[1:]])
    rank=int(np.sum(np.linalg.svd(axes,compute_uv=False)>.05))
    if max(turns)<.15 or rank<2:raise ValueError('wrist rotations too small or about one axis only')
    best=None
    for method in [cv2.CALIB_HAND_EYE_TSAI,cv2.CALIB_HAND_EYE_PARK,cv2.CALIB_HAND_EYE_HORAUD,cv2.CALIB_HAND_EYE_DANIILIDIS]:
        r,t=cv2.calibrateHandEye([p[:3,:3] for p in a],[p[:3,3].reshape(3,1) for p in a],
            [p[:3,:3] for p in b],[p[:3,3].reshape(3,1) for p in b],method=method)
        x=rigid(r,t[:,0])
        if not np.isfinite(x).all():continue
        score=spread([p@x@q for p,q in zip(a,b)])
        if best is None or score[0]+.1*score[1]<best[1][0]+.1*best[1][1]:best=(x,score)
    if best is None:raise ValueError('hand-eye solvers failed')
    x=best[0];board=mean_pose([p@x@q for p,q in zip(a,b)])
    def unpack(v):return rigid(Rotation.from_rotvec(v[:3]).as_matrix(),v[3:6]),rigid(Rotation.from_rotvec(v[6:9]).as_matrix(),v[9:])
    def residual(v):
        xx,bb=unpack(v);rows=[]
        for p,q in zip(a,b):
            e=np.linalg.inv(bb)@p@xx@q
            rows.extend(e[:3,3]);rows.extend(Rotation.from_matrix(e[:3,:3]).as_rotvec()*.1)
        return np.array(rows)
    v0=np.r_[Rotation.from_matrix(x[:3,:3]).as_rotvec(),x[:3,3],Rotation.from_matrix(board[:3,:3]).as_rotvec(),board[:3,3]]
    x,board=unpack(least_squares(residual,v0).x)
    views=[p@x@q for p,q in zip(a,b)];translation,rotation=spread(views)
    return {'T_link_camera':x,'T_base_board':board,'views':len(a),'translation_spread_m':translation,
        'rotation_spread_rad':rotation,
        'translation_rms_m':float(np.sqrt(np.mean([np.sum((v[:3,3]-board[:3,3])**2) for v in views]))),
        'rotation_rms_rad':float(np.sqrt(np.mean([angle(board[:3,:3].T@v[:3,:3])**2 for v in views]))),
        'largest_wrist_turn_rad':float(max(turns)),'rotation_axes_rank':rank}


# ---------------------------------------------------------------- timing

def clock_envelope(device_ms,arrival_s):
    """Device-clock slope and the earliest-arrival offset (an upper bound on
    exposure time). Exposure lies before arrival; time_offset() refines it."""
    d=np.asarray(device_ms,float);a=np.asarray(arrival_s,float)
    if len(d)<30 or not np.isfinite(d).all() or not np.isfinite(a).all():raise ValueError('at least 30 frames required')
    slope=float(np.polyfit(d-d[0],a-a[0],1)[0]);offsets=a-slope*d;base=float(np.min(offsets))
    return {'seconds_per_ms':slope,'offset_unix_s':base,'arrival_jitter_p95_s':float(np.percentile(offsets-base,95))}


def interpolate(rows,stamp):
    """Linear joint interpolation in a sorted (time, q) list; None outside it."""
    times=[r[0] for r in rows];i=int(np.searchsorted(times,stamp))
    if i<=0 or i>=len(rows):return None
    (t0,q0),(t1,q1)=rows[i-1],rows[i]
    if t1-t0>.06:return None
    return np.asarray(q0)+(np.asarray(q1)-q0)*(stamp-t0)/(t1-t0)


def time_offset(views,feedback,camera_pose,base_board,*,clock,search_s=.12,step_s=.001):
    """Choose the exposure offset that best explains moving board views.

    views: [{'device_ms','R_camera_board'}]; feedback: sorted [(unix_s, q16)];
    camera_pose(q)->T_base_camera. Returns refined offset and an error bound.
    """
    rows=sorted(feedback,key=lambda r:r[0]);base=transform(base_board)
    def cost(delta):
        errors=[]
        for v in views:
            q=interpolate(rows,v['device_ms']*clock['seconds_per_ms']+clock['offset_unix_s']+delta)
            if q is None:continue
            predicted=np.linalg.inv(camera_pose(q))@base
            errors.append(angle(predicted[:3,:3].T@np.asarray(v['R_camera_board'])))
        return errors
    grid=np.arange(-search_s,.01+1e-9,step_s);scores=[]
    for delta in grid:
        e=cost(delta);scores.append(np.mean(e) if len(e)>=.8*len(views) else np.inf)
    scores=np.array(scores);k=int(np.argmin(scores))
    if not np.isfinite(scores[k]) or k in (0,len(grid)-1):raise ValueError('timing search found no interior minimum')
    errors=np.array(cost(grid[k]));sigma=float(np.std(errors))
    rates=[]
    for v in views:
        stamp=v['device_ms']*clock['seconds_per_ms']+clock['offset_unix_s']+grid[k]
        a=interpolate(rows,stamp-.02);b=interpolate(rows,stamp+.02)
        if a is not None and b is not None:rates.append(angle(camera_pose(a)[:3,:3].T@camera_pose(b)[:3,:3])/.04)
    rate=float(np.median(rates)) if rates else 0.
    if rate<.03:raise ValueError('camera rotation too slow for timing estimate')
    bound=3*sigma/(rate*math.sqrt(len(errors)))+step_s+.002
    return {'delta_s':float(grid[k]),'offset_unix_s':clock['offset_unix_s']+float(grid[k]),
        'maximum_error_s':float(bound),'residual_rad':float(np.mean(errors)),'residual_std_rad':sigma,
        'angular_rate_rad_s':rate,'views':len(errors)}


# ---------------------------------------------------------------- geometry

def finger_travel(urdf):
    """Per finger link: opening axis (child frame) and upper travel in metres."""
    out={}
    for j in ET.parse(urdf).getroot().findall('joint'):
        child=j.find('child').get('link');limit=j.find('limit');axis=j.find('axis')
        if child.endswith('_finger') and limit is not None and axis is not None:
            a=np.array([float(v) for v in axis.get('xyz').split()]);a/=np.linalg.norm(a)
            out[child]={'axis':a,'travel_m':float(limit.get('upper'))}
    return out


def jaw_sweeps(collision,urdf):
    """Union of each finger hull at closed and fully open travel (URDF limits)."""
    from scipy.spatial import ConvexHull
    travel=finger_travel(urdf);count=0
    for row in collision['hulls']:
        if not row['link'].endswith('_finger'):continue
        if row['link'] not in travel:raise ValueError('finger travel missing in URDF: '+row['link'])
        v=np.asarray(row['vertices'],float);t=travel[row['link']]
        points=np.vstack([v,v+t['axis']*t['travel_m']]);hull=ConvexHull(points)
        row['jaw_sweep_hull_vertices']=points[hull.vertices].tolist();row['jaw_travel_m']=t['travel_m'];count+=1
    if count!=4:raise ValueError('expected four finger hulls')
    return collision


def camera_hull(collision,link,T_link_camera,size_m):
    """Conservative hull of the wrist camera and its bracket, fixed to `link`."""
    import trimesh
    w,h,d=(float(v) for v in size_m);x=transform(T_link_camera)
    corners=np.array([[sx*(w/2+.01),sy*(h/2+.01),z] for sx in (-1,1) for sy in (-1,1) for z in (-d-.005,.008)])
    # The bracket between the hand origin and the camera is inside this hull.
    hull=trimesh.convex.convex_hull(np.vstack([corners@x[:3,:3].T+x[:3,3],[[0.,0,0]]]))
    rows=[r for r in collision['hulls'] if r.get('kind')!='wrist_camera']
    rows.append({'link':link,'kind':'wrist_camera','vertices':hull.vertices.tolist(),
        'faces':hull.faces.tolist(),'size_m':[w,h,d]})
    collision['hulls']=rows
    pairs={tuple(p) for p in collision['disabled_pairs']};pairs.add((link,link))
    collision['disabled_pairs']=[list(p) for p in sorted(pairs)]
    return collision


def pad_points(collision,kin,tool_link,radius_m):
    """Eight pad-box corners in the tool frame at the measured bottle width."""
    values={j['name']:0. for j in kin.joints.values() if j['type']!='fixed'};cache={}
    tool=kin.pose(tool_link,values,cache);side=tool_link.split('_')[1];points=[]
    for row in collision['hulls']:
        if row['link'] in ('openarm_%s_left_finger'%side,'openarm_%s_right_finger'%side):
            t=np.linalg.inv(tool)@kin.pose(row['link'],values,cache);v=np.asarray(row['vertices'],float)
            points.append(v@t[:3,:3].T+t[:3,3])
    if len(points)!=2:raise ValueError('two finger hulls required for '+side)
    p=np.vstack(points);lo=p.min(axis=0);hi=p.max(axis=0)
    return [[x,y,z] for x in (lo[0],hi[0]) for y in (-radius_m,radius_m) for z in (lo[2],hi[2])]


# ---------------------------------------------------------------- grip

def grip(side,*,open_motor,limits,free,empty,stall,squeeze,center_tool,slip_span_m=None):
    """Thresholds from measured jaw position/torque. Every input is measured:
    free = |torque| samples while closing in air, empty = closed-in-air sample,
    stall = first stable contact on the bottle, squeeze = samples at full command."""
    lo,hi=(float(v) for v in limits);direction=1. if hi>open_motor else -1.
    positions=[stall['position']]+[s['position'] for s in squeeze]
    torques=[abs(stall['torque'])]+[abs(s['torque']) for s in squeeze]
    p0,p1=min(positions)-.05,max(positions)+.05
    if p0<=empty['position']<=p1 or abs(empty['position']-np.median(positions))<.08:
        raise ValueError(side+': closed-in-air and bottle-contact jaw positions overlap')
    if p0<=open_motor<=p1:raise ValueError(side+': contact range includes the open command')
    free_max=max([abs(v) for v in free]+[0.])
    lower=max(1.5*free_max,.3*abs(stall['torque']),.02);upper=max(1.3*max(torques),2*abs(stall['torque']))
    if not lower<min(torques):raise ValueError(side+': contact torque not separable from free closing')
    c=np.asarray(center_tool,float);margin=np.array([.02,.02,.02])
    slip=.006 if slip_span_m is None else float(np.clip(4*slip_span_m,.004,.01))
    return {'open_motor':float(open_motor),'close_motor':float(hi if direction>0 else lo),
        'motor_limits':[lo,hi],'contact_motor_range':[float(p0),float(p1)],
        'contact_torque_range':[float(lower),float(upper)],
        'object_center_bounds_tool_m':[(c-margin).tolist(),(c+margin).tolist()],
        'maximum_slip_m':slip,'maximum_rotation_rad':.06,'accepted_tool_uncertainty_m':.003}


def workspace(points,table_z,margin=.08):
    p=np.asarray(points,float);lo=p.min(axis=0)-margin;hi=p.max(axis=0)+margin
    lo[2]=max(lo[2],table_z+.005)
    if not np.all(lo<hi):raise ValueError('degenerate taught workspace')
    return [lo.tolist(),hi.tolist()]


def acceptance(category,*,subject,operator,measurements,passed,days=30):
    return {'schema':'openarm-physical-check/1','category':category,'passed':bool(passed),
        'subject_sha256':subject,'operator':str(operator),'measurements':measurements,
        'created_unix_s':time.time(),'valid_until_unix_s':time.time()+days*86400,
        'qualification':'operator-attested on-site measurement; repeat after any code, calibration or geometry change'}
