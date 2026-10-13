"""Read-only contact sampling at physics resolution, separate from control I/O."""
import mujoco
import numpy as np


class Contacts:
    def __init__(self, model, bottle_id=-1, box_floor_id=-1, table_id=-1):
        self.model, self.bottle_id, self.box_floor_id, self.table_id = model, bottle_id, box_floor_id, table_id
        self.bodies = [model.body(int(b)).name or f"body#{b}" for b in model.geom_bodyid]
        self.geoms = [model.geom(i).name or f"geom#{i}" for i in range(model.ngeom)]
        self.robot = ["openarm_" in name for name in self.bodies]
        self.is_bottle = model.geom_bodyid == bottle_id
        self.reset()

    def reset(self):
        self.hold_seconds = {"left": 0., "right": 0.}
        self.stable_seconds = self.penetration_seconds = self.max_penetration = 0.
        self.placement_seconds = 0.
        self.deep_penetration = False
        self.begin(0.)

    def begin(self, timestamp):
        self.start = float(timestamp)
        self.samples = 0
        self.duration = self.external_seconds = self.self_seconds = 0.
        self.external_impulse = self.self_impulse = self.peak_force = self.penetration = 0.
        self.external_peak_force = self.self_peak_force = 0.
        self.endpoint_external_force = self.endpoint_self_force = 0.
        self.pairs = {}

    def read(self, data):
        fingers = {"left": set(), "right": set()}
        supported = table_supported = False
        penetration = peak = external_force = self_force = 0.
        robot_contacts = 0
        pairs = {}
        force = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if min(a, b) < 0 or not (self.robot[a] or self.robot[b] or self.is_bottle[a] or self.is_bottle[b]):
                continue
            penetration = max(penetration, -float(c.dist))
            mujoco.mj_contactForce(self.model, data, i, force)
            normal = max(0., float(force[0]))
            peak = max(peak, normal)
            if normal <= .05:
                continue
            robot_pair = self.robot[a] and self.robot[b]
            external_pair = self.robot[a] != self.robot[b]
            robot_contacts += int(robot_pair)
            external_force += normal if external_pair else 0.
            self_force += normal if robot_pair else 0.
            pair = tuple(sorted((a, b)))
            row = pairs.setdefault(pair, {"normal_force_N": 0., "penetration_m": 0.,
                                         "robot_external": external_pair, "robot_self": robot_pair})
            row["normal_force_N"] += normal
            row["penetration_m"] = max(row["penetration_m"], -float(c.dist))
            if self.is_bottle[a] or self.is_bottle[b]:
                other = b if self.is_bottle[a] else a
                supported |= other == self.box_floor_id
                table_supported |= other == self.table_id
                body = self.bodies[other]
                for side in fingers:
                    if f"openarm_{side}_" in body and "finger" in body:
                        fingers[side].add("inner" if "inner" in body else "outer" if "outer" in body else body)
        return {"held": {s: len(v) >= 2 for s, v in fingers.items()},
                "touching": {s: bool(v) for s, v in fingers.items()}, "supported": bool(supported),
                "table_supported": bool(table_supported),
                "penetration_m": penetration, "peak_normal_force_N": peak, "robot_contact_count": robot_contacts,
                "external_force_N": external_force, "self_force_N": self_force, "pairs": pairs}

    def sample(self, contact, dt, timestamp, *, stable=False, placement_stable=False):
        """dt covers a solved physical substep; dt=0 is the final-state check."""
        if dt < 0 or not np.isfinite(dt):
            raise ValueError("invalid contact sampling interval")
        for side in self.hold_seconds:
            self.hold_seconds[side] = self.hold_seconds[side] + dt if contact["held"][side] else 0.
        self.stable_seconds = self.stable_seconds + dt if stable else 0.
        self.placement_seconds = self.placement_seconds + dt if placement_stable else 0.
        self.penetration_seconds = self.penetration_seconds + dt if contact["penetration_m"] > .01 else 0.
        self.deep_penetration |= self.penetration_seconds >= .1
        self.max_penetration = max(self.max_penetration, contact["penetration_m"])
        self.penetration = max(self.penetration, contact["penetration_m"])
        self.peak_force = max(self.peak_force, contact["peak_normal_force_N"])
        self.samples += int(dt > 0)
        self.duration += dt
        self.external_seconds += dt if contact["external_force_N"] > 0 else 0.
        self.self_seconds += dt if contact["self_force_N"] > 0 else 0.
        self.external_impulse += contact["external_force_N"] * dt
        self.self_impulse += contact["self_force_N"] * dt
        self.external_peak_force = max(self.external_peak_force, contact["external_force_N"])
        self.self_peak_force = max(self.self_peak_force, contact["self_force_N"])
        if dt == 0:
            self.endpoint_external_force = contact["external_force_N"]
            self.endpoint_self_force = contact["self_force_N"]
        for pair, value in contact["pairs"].items():
            row = self.pairs.setdefault(pair, {"geoms": [self.geoms[g] for g in pair],
                "bodies": [self.bodies[g] for g in pair], "robot_external": value["robot_external"],
                "robot_self": value["robot_self"], "first_time_s": float(timestamp), "last_time_s": float(timestamp),
                "duration_s": 0., "normal_impulse_Ns": 0., "peak_normal_force_N": 0., "penetration_m": 0.})
            row["last_time_s"] = float(timestamp)
            row["duration_s"] += dt
            row["normal_impulse_Ns"] += value["normal_force_N"]*dt
            row["peak_normal_force_N"] = max(row["peak_normal_force_N"], value["normal_force_N"])
            row["penetration_m"] = max(row["penetration_m"], value["penetration_m"])

    def summary(self):
        return {"sampling": "every solved physics step plus a zero-duration final-state check",
                "physics_dt_s": float(self.model.opt.timestep), "samples": self.samples,
                "interval_start_s": self.start, "duration_s": self.duration,
                "robot_external_seconds": self.external_seconds, "robot_self_seconds": self.self_seconds,
                "robot_external_impulse_Ns": self.external_impulse, "robot_self_impulse_Ns": self.self_impulse,
                "robot_external_peak_force_N": self.external_peak_force, "robot_self_peak_force_N": self.self_peak_force,
                "endpoint_robot_external_force_N": self.endpoint_external_force,
                "endpoint_robot_self_force_N": self.endpoint_self_force,
                "peak_normal_force_N": self.peak_force, "max_penetration_m": self.penetration,
                "hold_seconds": dict(self.hold_seconds), "stable_seconds": self.stable_seconds,
                "placement_seconds": self.placement_seconds,
                "pairs": list(self.pairs.values())}
