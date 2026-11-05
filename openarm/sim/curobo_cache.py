"""Exact kinematic cache updates for the pinned official cuRobo V2 API."""
import copy
import json
import xml.etree.ElementTree as ET


def robot_config(base, side, locked, carrying):
    config=copy.deepcopy(base)
    config['kinematics']['tool_frames']=[side+'_candidate_tcp']
    config['kinematics']['lock_joints']=dict(locked)
    if carrying:
        kin=config['kinematics'];link='carried_bottle'
        kin['extra_links']={link:{'parent_link_name':side+'_candidate_tcp','link_name':link,
            'joint_name':'carried_bottle_fixed','joint_type':'FIXED','fixed_transform':[0,0,0,1,0,0,0]}}
        kin['extra_collision_spheres']={link:32}
        kin['collision_link_names'].append(link)
        contacts=[f'openarm_{side}_hand',f'openarm_{side}_right_finger',f'openarm_{side}_left_finger']
        kin['self_collision_ignore'][link]=contacts
        for name in contacts:kin['self_collision_ignore'].setdefault(name,[]).append(link)
        kin['self_collision_buffer'][link]=0.
    return config


def cache_key(side, locked, scene, carrying):
    # Joint values are copied exactly into every rollout on each change. Only
    # those values may vary without rebuilding; world and topology remain exact.
    return json.dumps([side,sorted(locked),scene,carrying],sort_keys=True)


def kinematic_models(planner):
    rollouts=[]
    for solver in (planner.ik_solver,planner.trajopt_solver):
        rollouts.extend(solver.get_all_rollout_instances())
    if planner.graph_planner is not None:
        rollouts.extend(planner.graph_planner.get_all_rollout_instances())
    result=[];seen=set()
    for rollout in rollouts:
        model=rollout.transition_model.robot_model
        if id(model) not in seen:result.append(model);seen.add(id(model))
    return result


def copy_kinematics(models, fresh):
    # Validate the shape/topology BEFORE any in-place modification. CUDA graph
    # pointers must remain valid; never approximate the locked joint values.
    for model in models:
        params=model.config.kinematics_config
        if params.joint_names!=fresh.joint_names or params.tool_frames!=fresh.tool_frames:
            raise ValueError('cached kinematic topology changed')
        for field in ('fixed_transforms','joint_map','link_spheres','link_sphere_idx_map','link_map'):
            old,new=getattr(params,field),getattr(fresh,field)
            if (old is None)!=(new is None) or old is not None and old.shape!=new.shape:
                raise ValueError('cached kinematic shape changed: '+field)
    bindings=[]
    for model in models:
        params=model.config.kinematics_config
        pointers={field:getattr(params,field).data_ptr() for field in ('fixed_transforms','link_spheres')}
        model.update_kinematics_config(fresh)
        if any(getattr(params,field).data_ptr()!=ptr for field,ptr in pointers.items()):
            raise ValueError('kinematic update replaced captured buffers')
        bindings.append(pointers)
    return {'rollout_models_updated':len(models),'captured_buffers_preserved':True,
            'locked_values':'exact incoming joint values through official float32 kinematics; no rounding or stale lock reuse'}


def joint_links(path):
    tree=ET.parse(path)
    result={}
    for item in tree.findall('joint'):
        mimic=item.find('mimic')
        result[item.attrib['name']]={'parent':item.find('parent').attrib['link'],
            'child':item.find('child').attrib['link'],'type':item.attrib['type'],
            'mimic':mimic.attrib['joint'] if mimic is not None else None}
    return result,[item.attrib['name'] for item in tree.findall('link')]


def refresh_from_full_fk(model, poses, links, locked, torch):
    """Same parent^-1 * child operation as official loader, cached topology.

    Only fixed transforms and stored lock feedback change. Original sphere
    geometry, collision pair tables and solver settings remain in place.
    """
    fresh=model.config.kinematics_config.clone()
    selected=set(locked)
    while True:
        expanded=selected|{name for name,row in links.items() if row['mimic'] in selected}
        if expanded==selected:break
        selected=expanded
    for name in selected:
        if name not in links:raise ValueError('locked joint missing from bound URDF')
        row=links[name]
        if row['type']=='fixed':continue
        child=row['child']
        if child not in fresh.link_name_to_idx_map:continue  # irrelevant branch omitted by official loader
        index=fresh.link_name_to_idx_map[child]
        if int(fresh.joint_map_type[index].item())!=-1:
            raise ValueError('requested lock is still active in reduced topology')
        transform=poses.get_link_pose(row['parent']).inverse().multiply(poses.get_link_pose(child)).get_affine_matrix()
        fresh.fixed_transforms[index].copy_(transform.reshape_as(fresh.fixed_transforms[index]))
    if fresh.lock_jointstate is None or set(fresh.lock_jointstate.joint_names)-set(locked):
        raise ValueError('stored lock inventory differs from incoming robot state')
    fresh.lock_jointstate.position.copy_(torch.as_tensor([locked[n] for n in fresh.lock_jointstate.joint_names],
        device=fresh.lock_jointstate.position.device,dtype=fresh.lock_jointstate.position.dtype).reshape_as(fresh.lock_jointstate.position))
    return fresh
