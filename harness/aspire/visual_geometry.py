"""Estimate projective views from RGB tool marks + robot FK, never simulator cameras."""
import numpy as np
from scipy import ndimage
from .visual import RGBRig, MARKERS, tool_feature
from .runtime import dump


def basket_opening(rgb):
    """White-rim enclosed opening, based only on RGB and a shape prior."""
    a=np.asarray(rgb);mask=ndimage.binary_closing(a.min(axis=2)>220,iterations=5)
    holes=ndimage.binary_fill_holes(mask)&~mask;labels,_=ndimage.label(holes);candidates=[]
    for i,sl in enumerate(ndimage.find_objects(labels),1):
        yy,xx=np.where(labels[sl]==i);xx=xx+sl[1].start;yy=yy+sl[0].start
        if len(xx)<1500 or yy.mean()>.55*a.shape[0]:continue
        ratio=np.ptp(xx)/max(np.ptp(yy),1)
        if 1.4<ratio<4:candidates.append((len(xx),np.array([xx.mean(),yy.mean()])))
    if not candidates:raise ValueError('white basket opening not visible')
    return max(candidates,key=lambda x:x[0])[1]


def project(camera,points):
    p=np.c_[np.atleast_2d(points),np.ones(len(np.atleast_2d(points)))]@camera.T
    return p[:,:2]/p[:,2:]


def fit_camera(xyz,uv):
    xyz=np.asarray(xyz,float);uv=np.asarray(uv,float)
    if len(xyz)<16:raise ValueError('more robot marker observations required')
    def normalise(a):
        centre=a.mean(axis=0);scale=np.sqrt(a.shape[1])/np.sqrt(np.mean(np.sum((a-centre)**2,axis=1)))
        t=np.eye(a.shape[1]+1);t[:-1,:-1]*=scale;t[:-1,-1]=-scale*centre
        return (a-centre)*scale,t
    x,t3=normalise(xyz);u,t2=normalise(uv);xh=np.c_[x,np.ones(len(x))];rows=[]
    for p,(a,b) in zip(xh,u):rows.extend([np.r_[p,np.zeros(4),-a*p],np.r_[np.zeros(4),p,-b*p]])
    _,s,v=np.linalg.svd(rows,full_matrices=False)
    camera=np.linalg.inv(t2)@v[-1].reshape(3,4)@t3
    residual=np.linalg.norm(project(camera,xyz)-uv,axis=1)
    if np.percentile(residual,95)>3:raise ValueError('RGB robot projection residual exceeds 3px')
    return camera,{'samples':len(xyz),'rms_px':float(np.sqrt(np.mean(residual**2))),'p95_px':float(np.percentile(residual,95)),'singular_tail':s[-3:].tolist()}


def triangulate(cameras,pixels):
    rows=[]
    for p,(u,v) in zip(cameras,np.asarray(pixels).reshape(2,2)):
        rows.extend([u*p[2]-p[0],v*p[2]-p[1]])
    _,_,v=np.linalg.svd(rows);x=v[-1];x=x[:3]/x[3]
    error=max(np.linalg.norm(project(p,[x])[0]-uv) for p,uv in zip(cameras,np.asarray(pixels).reshape(2,2)))
    if not np.isfinite(x).all() or error>12:raise ValueError('inconsistent two-view image points')
    return x,float(error)


class GeometryRig(RGBRig):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.xyz=[[],[]];self.uv=[[],[]];self.cameras=None;self.quality=None

    def lock_target(self):
        super().lock_target()
        self.basket_pixels=np.concatenate([basket_opening(self.image(v,dots=False)) for v in range(2)])
        dump(self.folder/'basket-target.json',{'pixels':self.basket_pixels.tolist(),'source':'initial RGB white-rim opening; fixed basket and cameras assumed'})

    def read(self):
        result=super().read()
        env=self.env;body=env.ee_ids['right'];r=env.data.xmat[body].reshape(3,3)
        xyz=[env.data.xpos[body]+r@np.array([x,.006+y,.075]) for x,y in MARKERS]
        for view in range(2):
            _,dots=tool_feature(self.image(view));self.xyz[view].extend(xyz);self.uv[view].extend(dots)
        return result

    def calibrate(self):
        values=[fit_camera(x,u) for x,u in zip(self.xyz,self.uv)]
        self.cameras=[v[0] for v in values];self.quality=[v[1] for v in values]
        dump(self.folder/'estimated-projection.json',{'source':'RGB marker centroids and encoder FK; simulator camera matrices not read','quality':self.quality,'estimated_projection':[p.tolist() for p in self.cameras]})

    def locate(self,pixels):
        if self.cameras is None:self.calibrate()
        return triangulate(self.cameras,pixels)

    def basket(self):
        pixels=self.basket_pixels
        position,error=self.locate(pixels)
        return {'estimated_position_m':position.tolist(),'pixels':pixels.tolist(),'reprojection_error_px':error,
                'feature':'white basket opening centre; floor height not directly observed'}
