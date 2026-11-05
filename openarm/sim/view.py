"""Operator-only orbit camera; never part of the policy observation contract."""
import math

FRONT={'mode':'front','azimuth':180.,'elevation':-24.,'distance':1.25,'lookat':[.43,0.,1.20]}


def validate_view(values):
    if not isinstance(values,dict) or set(values)-set(FRONT):
        raise ValueError('unknown view parameter')
    mode=values.get('mode')
    if mode not in ('front','free'):
        raise ValueError('view mode must be front or free')
    if mode=='front':
        return {**FRONT,'lookat':list(FRONT['lookat'])}
    if set(values)!=set(FRONT):
        raise ValueError('free view requires azimuth, elevation, distance and lookat')
    def number(value,lower,upper):
        if type(value) not in (int,float) or not math.isfinite(value) or not lower<=value<=upper:
            raise ValueError('view parameter out of range')
        return float(value)
    point=values['lookat']
    if not isinstance(point,list) or len(point)!=3:
        raise ValueError('view lookat must have three numbers')
    return {'mode':mode,'azimuth':number(values['azimuth'],-360,360),
            'elevation':number(values['elevation'],-85,35),'distance':number(values['distance'],.65,3.5),
            'lookat':[number(v,*limits) for v,limits in zip(point,[(-.5,1.5),(-1.2,1.2),(.3,2.)])]}
