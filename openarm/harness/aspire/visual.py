"""RGB-relative reach experiment. Legacy skills and physical devices untouched.

Two fixed RGB views and a simulated rigid planar tool fixture are explicit
sensor assumptions. Neither camera matrices, depth, segmentation IDs nor object
coordinates enter the image controller. Robot motion uses an independent
encoder-feedback IK controller; native object/contact state is evaluation only.
"""
from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .runtime import dump


VIEWS = [dict(mode='free',lookat=[.46, -.12, 1.12], distance=1.08, azimuth=75., elevation=-32.),
         dict(mode='free',lookat=[.46, -.12, 1.12], distance=1.12, azimuth=125., elevation=-28.)]
COLORS = [(1., .08, .03, 1.), (.03, 1., .25, 1.), (1., .05, 1., 1.), (.02, .85, 1., 1.)]
MARKERS = [(.16,-.09),(.16,.09),(.24,-.09),(.24,.09)]


def blobs(mask):
    labels, n = ndimage.label(mask)
    answer = []
    for ident, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        yy, xx = np.where(labels[sl] == ident)
        if len(xx) < 3:
            continue
        xx = xx + sl[1].start; yy = yy + sl[0].start
        answer.append((len(xx), np.c_[xx, yy]))
    return sorted(answer, key=lambda p: -p[0])


def tool_feature(rgb):
    """Track the rendered colors as pixels; no FK projection or depth lookup."""
    a = rgb.astype(float)
    masks = [(a[:, :, 0] > 90) & (a[:, :, 0] > 2*a[:, :, 1]) & (a[:, :, 0] > 2*a[:, :, 2]),
             (a[:, :, 1] > 90) & (a[:, :, 1] > 2*a[:, :, 0]) & (a[:, :, 1] > 1.5*a[:, :, 2]),
             (a[:,:,0]>90)&(a[:,:,2]>90)&(np.minimum(a[:,:,0],a[:,:,2])>2*a[:,:,1]),
             (a[:,:,1]>90)&(a[:,:,2]>90)&(np.minimum(a[:,:,1],a[:,:,2])>2*a[:,:,0])]
    dots = []
    strengths=[a[:,:,0]-np.maximum(a[:,:,1],a[:,:,2]),
               a[:,:,1]-np.maximum(a[:,:,0],a[:,:,2]),
               np.minimum(a[:,:,0],a[:,:,2])-a[:,:,1],
               np.minimum(a[:,:,1],a[:,:,2])-a[:,:,0]]
    for mask,strength in zip(masks,strengths):
        found = blobs(mask)
        if not found or found[0][0] > 800:
            raise ValueError('tool dots not visible')
        xy=found[0][1];weights=strength[xy[:,1],xy[:,0]]
        dots.append(np.average(xy,axis=0,weights=weights))
    # A planar tool fiducial: extrapolate its pinch point by homography. This
    # uses the declared tool fixture layout, NOT camera calibration or depth.
    matrix=[];rhs=[]
    for (x,y),(u,v) in zip(MARKERS,dots):
        matrix.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
        rhs.extend([u,v])
    hom=np.r_[np.linalg.solve(np.array(matrix),np.array(rhs)),1.].reshape(3,3)
    p=hom@np.array([0.,0.,1.])
    if abs(p[2])<.1:raise ValueError('tool fiducial degenerate')
    return p[:2]/p[2], dots


def bottle_feature(rgb):
    """Upright black cylinder, restricted image workspace; not arbitrary objects."""
    h, w = rgb.shape[:2]
    area_scale=h*w/(960*540)
    a = rgb.astype(float)
    mask = (a.max(axis=2) < 30)
    mask[:int(.18*h)] = False
    # Camera views and the right-side table region are declared scene priors.
    # This region is constant, not projected from a hidden object position.
    mask[:, :int(.35*w)] = False
    found = []
    for area, pixels in blobs(mask):
        span = np.ptp(pixels, axis=0)+1
        if not 80*area_scale <= area <= 6000*area_scale or not 1.1 < span[1]/span[0] < 8:
            continue
        fill = area/np.prod(span)
        if fill < .55 or pixels[:, 1].max() > .9*h:
            continue
        # Prefer a separated compact cylinder, not the torso/arm silhouette.
        found.append((fill*np.sqrt(area), pixels))
    if not found:
        raise ValueError('upright black bottle not detected')
    pixels = max(found, key=lambda r: r[0])[1]
    centre = pixels.mean(axis=0)
    _, _, vh = np.linalg.svd(pixels-centre, full_matrices=False)
    axis = vh[0]
    if axis[1] < 0:
        axis = -axis
    axial = (pixels-centre)@axis
    lo, hi = np.percentile(axial, [2, 98])
    # Lower-body visual target: 40% from the bottom of the body silhouette.
    target = centre + axis*(lo+.60*(hi-lo))
    return target, {'bbox': [*pixels.min(axis=0).tolist(), *pixels.max(axis=0).tolist()],
                    'area': len(pixels), 'axis': axis.tolist(), 'body_fraction': .40}


class RGBRig:
    def __init__(self, env, folder, *, yaw_offset=0., pixel_noise=0., seed=0):
        self.env, self.folder = env, Path(folder)
        self.folder.mkdir(exist_ok=True)
        self.views = copy.deepcopy(VIEWS)
        for v in self.views:
            v['azimuth'] += yaw_offset
        self.noise, self.rng = pixel_noise, np.random.default_rng(seed)
        self.counter = 0; self.goals = None
        self.renderer=mujoco.Renderer(env.model,height=960,width=1280)

    def image(self, view, *, dots=True):
        # Render-only rigid tool fixture. mjv geometry participates in normal
        # RGB occlusion; it is not an image-coordinate overlay or label mask.
        env = self.env
        config=self.views[view];cam=mujoco.MjvCamera();cam.lookat[:]=config['lookat']
        cam.distance,cam.azimuth,cam.elevation=config['distance'],config['azimuth'],config['elevation']
        renderer=self.renderer;renderer.update_scene(env.data,camera=cam,scene_option=env._option)
        if dots:
            body = env.ee_ids['right']
            rot = env.data.xmat[body].reshape(3, 3)
            for (x,y), color in zip(MARKERS, COLORS):
                point = env.data.xpos[body]+rot@np.array([x, .006+y, .075])
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([.0045, 0., 0.]), pos=point,
                    mat=np.eye(3).ravel(), rgba=np.array(color, dtype=np.float32))
                geom.emission = .8
                renderer.scene.ngeom += 1
        return renderer.render().copy()

    def lock_target(self):
        goals, detections = [], []
        for view in range(2):
            rgb = self.image(view, dots=False)
            goal, evidence = bottle_feature(rgb)
            goals.append(goal); detections.append(evidence)
            Image.fromarray(rgb).save(self.folder/f'target-{view}.png')
        self.goals = np.concatenate(goals);self.detections=detections
        dump(self.folder/'target.json', {'source': 'RGB cylinder silhouettes only',
            'pixels': self.goals.tolist(), 'detections': detections,
            'assumption': 'bottle remains stationary on table until right grasp; fixed cameras during each reach',
                'depth_used': False, 'camera_matrices_used': False})

    def object_motion(self):
        shifts=[]
        for view,det in enumerate(self.detections):
            rgb=self.image(view,dots=False);x0,y0,x1,y1=det['bbox']
            centre=(x0+x1)/2;half=(x1-x0)*.30
            left,right=int(centre-half),int(centre+half)
            top=max(0,int(y0-160));bottom=min(rgb.shape[0],int(y1+12))
            dark=rgb[top:bottom,left:right].max(axis=2)<30
            occupied=np.where(dark.sum(axis=1)>.45*(right-left))[0]
            if len(occupied)<12:raise ValueError('bottle bottom not visible for hold verification')
            observed=top+occupied[-1]
            shifts.append(float(y1-observed))
            Image.fromarray(rgb).save(self.folder/f'hold-check-{view}.jpg',quality=90)
        return shifts

    def read(self):
        features = []; images = []; dots = []
        for view in range(2):
            rgb = self.image(view)
            try:
                centre, pair = tool_feature(rgb)
            except ValueError as exc:
                Image.fromarray(rgb).save(self.folder/f'failed-{self.counter:03d}-{view}.png')
                raise ValueError(f'RGB view {view}: {exc}') from exc
            features.extend(centre); images.append(rgb); dots.append(pair)
        f = np.array(features)
        if self.noise:
            f += self.rng.normal(0, self.noise, 4)
        ident = self.counter; self.counter += 1
        for view, rgb in enumerate(images):
            path = self.folder/f'{ident:03d}-{view}.jpg'
            Image.fromarray(rgb).save(path, quality=88)
            im = Image.fromarray(rgb); d = ImageDraw.Draw(im)
            for p in dots[view]:
                u, v = p; d.ellipse((u-5, v-5, u+5, v+5), outline='#efb44b', width=2)
            u, v = self.goals[2*view:2*view+2]
            d.line((u-8,v,u+8,v),fill='#1f94ff',width=2);d.line((u,v-8,u,v+8),fill='#1f94ff',width=2)
            im.save(self.folder/f'{ident:03d}-{view}-evidence.jpg', quality=88)
        return f, {'id': ident, 'tick': self.env.tick, 'error_px': (self.goals-f).tolist(),
                   'max_view_error_px': float(np.linalg.norm((self.goals-f).reshape(2,2),axis=1).max())}


def main():
    from .visual_control import main as execute
    execute()


if __name__ == '__main__':
    main()
