"""Freeze actual MJCF/assets and identify the compiled candidate, not reality.

Bytes supplied through MuJoCo's VFS are the bytes hashed here. Disk checks use
stat identity as a cache invalidator, then compare content; no shared files are
changed. A compiled fingerprint identifies software parameters, not calibration.
"""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class ModelInputs:
    def __init__(self, root, entry, *, overrides=None):
        self.root = Path(root).resolve()
        self.overrides = overrides or {}
        self.contents, self.rows, self.cache = {}, {}, {}
        self.includes, self.assets = {}, {}
        self._visit(self._path(entry))

    def _path(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('model input outside local project: '+str(path))
        return path

    def _freeze(self, path, kind, table):
        path = self._path(path)
        name = str(path.relative_to(self.root))
        if name in self.contents:
            return self.contents[name]
        before = identity(path)
        data = path.read_bytes()
        if before != identity(path):
            raise ValueError('model input changed while reading: '+name)
        # MuJoCo VFS resolves file resources by basename. Refuse ambiguous
        # aliases instead of silently selecting bytes from another directory.
        if path.name in table and table[path.name] != data:
            raise ValueError('ambiguous VFS basename: '+path.name)
        table[path.name] = data
        self.contents[name] = data
        self.rows[name] = {'kind':kind, 'bytes':len(data), 'sha256':sha(data)}
        self.cache[name] = (before, self.rows[name]['sha256'])
        return data

    def _visit(self, path):
        name = str(path.relative_to(self.root))
        if name in self.contents:
            return
        tree = ET.fromstring(self._freeze(path, 'mjcf', self.includes))
        compiler = tree.find('compiler')
        attrs = compiler.attrib if compiler is not None else {}
        assetdir = path.parent / attrs.get('assetdir', '')
        dirs = {'mesh': path.parent/attrs.get('meshdir', str(assetdir)),
                'texture': path.parent/attrs.get('texturedir', str(assetdir)),
                'hfield': assetdir}
        for element in tree.iter('include'):
            self._visit(self._path(path.parent/element.attrib['file']))
        for element in tree.findall('asset/model'):
            target = self.overrides.get((name, element.attrib['file']), path.parent/element.attrib['file'])
            self._visit(self._path(target))
        for kind, folder in dirs.items():
            for element in tree.findall('asset/'+kind):
                for attr, value in element.attrib.items():
                    if attr == 'file' or kind == 'texture' and attr.startswith('file'):
                        self._freeze(folder/value, kind, self.assets)

    def xml(self, path):
        return self.contents[str(self._path(path).relative_to(self.root))].decode('utf-8')

    def hashes(self):
        return {name: row['sha256'] for name,row in self.rows.items()}

    def changed(self):
        changed = []
        for name, expected in self.rows.items():
            path = self.root/name
            try:
                stat = identity(path)
                previous, digest = self.cache[name]
                if stat != previous:
                    data = path.read_bytes()
                    if stat != identity(path):
                        changed.append(name)
                        continue
                    digest = sha(data)
                    self.cache[name] = (stat,digest)
                if digest != expected['sha256']:
                    changed.append(name)
            except OSError:
                changed.append(name)
        return sorted(changed)

    def describe(self, model):
        # Compiled resources include mirrored/scaled uses of the same file.
        paths = model.paths
        compiled = []
        for kind, count, field in [('mesh',model.nmesh,'mesh_pathadr'),
                                   ('texture',model.ntex,'tex_pathadr'),
                                   ('hfield',model.nhfield,'hfield_pathadr')]:
            for i, address in enumerate(getattr(model,field)):
                if address < 0:
                    continue  # procedural textures/hfields have no source file
                resource = paths[int(address):].split(b'\0',1)[0].decode()
                matches = [name for name,row in self.rows.items()
                           if row['kind']==kind and Path(name).name==Path(resource).name]
                if len(matches) != 1:
                    raise ValueError('compiled resource not uniquely frozen: '+resource)
                compiled.append({'kind':kind,'id':i,'name':getattr(model,kind)(i).name,
                                 'source':matches[0],'compiled_path':resource})
        return {'schema':'model-inputs/1','loading':'frozen MJCF and resource bytes through MuJoCo VFS',
                'files':copy.deepcopy(self.rows),'compiled_resources':compiled,
                'content_sha256':sha(json.dumps(self.rows,sort_keys=True).encode()),
                'coverage':'active MJCF dependency tree and all file-backed mesh/texture/hfield resources; procedural resources covered by compiled model',
                'disk_check':'content SHA256 cached by device/inode/size/mtime_ns/ctime_ns; cache is not a physical measurement',
                'physically_calibrated':False,'transfer_ready':False}


def compiled_evidence(model):
    # MJB contains compiled geometry, inertia, options, equality constraints,
    # actuators, cameras, procedural resources and visual parameters. Record its
    # digest once, after local overrides, without storing 258 MB in every run.
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    result={'schema':'compiled-model/1','engine':'mujoco-'+mujoco.__version__,
            'format':'MuJoCo MJB','bytes':len(buffer),'sha256':sha(buffer),
            'capture':'after local compile/controller/render overrides, before reset or step',
            'counts':{'bodies':model.nbody,'joints':model.njnt,'geoms':model.ngeom,
                      'meshes':model.nmesh,'actuators':model.nu,'cameras':model.ncam},
            'runtime_mutation':'controller feedforward bias column and experiment-specific camera/render settings are not this initial snapshot; applied actions and controller semantics logged separately',
            'physically_calibrated':False,'transfer_ready':False}
    return result


class ParameterAudit:
    """Small mutable parameter arrays; the initial MJB covers bulk assets.

    This does not continuously rehash large compiled mesh/texture buffers.
    Public runtime has no bulk-asset mutation tool; changing those needs a new
    compile. Dynamic actuator feedforward is explicitly excluded and logged by
    the controller instead of invalidating identity on each physics step.
    """
    prefixes=('body_','jnt_','dof_','geom_','actuator_','eq_','cam_','site_',
              'pair_','tendon_','sensor_')

    def __init__(self,model):
        self.model=model
        self.baseline=self.fields()

    def fields(self):
        fields={}
        for name in dir(self.model):
            if not name.startswith(self.prefixes) and name not in ('qpos0','exclude_signature'):
                continue
            array=getattr(self.model,name)
            if not isinstance(array,np.ndarray):continue
            if name=='actuator_biasprm':array=array[:,1:]
            fields[name]={'shape':list(array.shape),'dtype':str(array.dtype),
                          'sha256':sha(np.ascontiguousarray(array))}
        for name in dir(self.model.opt):
            if name.startswith('_'):continue
            value=getattr(self.model.opt,name)
            if isinstance(value,np.ndarray):value=value.tolist()
            if isinstance(value,(int,float,list)):
                fields['option.'+name]=value
        return fields

    def summary(self):
        current=self.fields()
        return {'schema':'runtime-parameters/1',
                'sha256':sha(json.dumps(current,sort_keys=True).encode()),
                'changed_fields':sorted(name for name in current if current[name]!=self.baseline.get(name)),
                'fields':len(current),
                'scope':'mutable body/joint/dof/geom/actuator/equality/camera/site/pair/tendon/sensor parameters and all numeric options; initial MJB identifies bulk assets',
                'excluded':'actuator_biasprm[:,0] dynamic feedforward; bulk compiled mesh/texture buffers are not continuously rehashed',
                'physically_calibrated':False}
