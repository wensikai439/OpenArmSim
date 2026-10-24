def run():
    calls = [0]
    sim_time = [0.0]

    def tick(n=1):
        calls[0] += n
        return calls[0] <= 64

    def obs():
        return observe()

    def bail(reason, o=None):
        return {"status": "stopped", "reason": reason, "observation": o}

    o = obs()
    if o["terminal"] or o["invalid"]:
        return bail("terminal_or_invalid", o)
    if len(list(o["held_by"])) != 0 or o["touching"]["left"] or o["touching"]["right"]:
        return bail("hands_not_empty", o)
    if o["operation"] is not None:
        return bail("active_operation", o)

    excluded = ()
    attempts = 0
    checkpoint = None
    grasp_ok = False
    while attempts < 3 and not grasp_ok:
        if not tick():
            return bail("call_budget", obs())
        pg = propose_grasps(exclude_ids=excluded)
        cands = list(pg["candidates"])
        if len(cands) == 0:
            return bail("no_candidates", obs())
        gid = cands[0]["id"]
        if not tick():
            return bail("call_budget", obs())
        r = pregrasp(gid)
        if not (r["ok"] and r["done"]):
            return bail("pregrasp_failed", obs())
        o = obs()
        checkpoint = o["checkpoint_id"]
        if not tick():
            return bail("call_budget", obs())
        r = approach(gid)
        if not (r["ok"] and r["done"]):
            return bail("approach_failed", obs())
        if not tick():
            return bail("call_budget", obs())
        r = close_gripper("right")
        attempts += 1
        if r["ok"] and r["done"]:
            if not tick():
                return bail("call_budget", obs())
            v = verify_hold("right")
            o = obs()
            if v["ok"] and "right" in list(o["held_by"]):
                grasp_ok = True
                break
            return bail("hold_unverified", o)
        if r.get("error_code") != "grasp_not_confirmed":
            return bail("unknown_close_error:" + str(r.get("error_code")), obs())
        o = obs()
        if len(list(o["held_by"])) != 0 or o["touching"]["left"]:
            return bail("grasp_failed_with_hold_state", o)
        motor = o["gripper_motor_feedback_rad"]["right"]
        opened = o["gripper_open_rad"]["right"]
        if abs(opened - motor) < 0.12 and not o.get("_topup_used"):
            if not tick():
                return bail("call_budget", obs())
            r2 = close_gripper("right")
            if r2["ok"] and r2["done"]:
                if not tick():
                    return bail("call_budget", obs())
                v = verify_hold("right")
                o = obs()
                if v["ok"] and "right" in list(o["held_by"]):
                    grasp_ok = True
                    break
        if not tick():
            return bail("call_budget", obs())
        r = open_gripper("right")
        if not (r["ok"] and r["done"]):
            return bail("open_failed", obs())
        if checkpoint is None:
            return bail("no_checkpoint", obs())
        if not tick():
            return bail("call_budget", obs())
        r = retreat(checkpoint)
        if not (r["ok"] and r["done"]):
            return bail("retreat_failed", obs())
        o = obs()
        excluded = tuple(list(excluded) + [gid])

    if not grasp_ok:
        return bail("grasp_attempts_exhausted", obs())

    if not tick():
        return bail("call_budget", obs())
    r = lift()
    if not (r["ok"] and r["done"]):
        return bail("lift_failed", obs())
    if not tick():
        return bail("call_budget", obs())
    v = verify_hold("right")
    o = obs()
    if not (v["ok"] and "right" in list(o["held_by"])):
        return bail("post_lift_hold_lost", o)

    if not tick():
        return bail("call_budget", obs())
    r = start_transfer(placement_aware=True)
    if not r["ok"]:
        return bail("transfer_start_failed", obs())

    def drive_advance():
        steps = 0
        while steps < 16:
            if not tick():
                return {"ok": False, "done": False, "error_code": "call_budget"}
            a = advance()
            steps += 1
            if not a["ok"]:
                return a
            if a["done"]:
                return a
        return {"ok": False, "done": False, "error_code": "advance_limit"}

    a = drive_advance()
    if not a["ok"] or not a["done"]:
        code = a.get("error_code")
        o = obs()
        if code in ("dual_hold_not_confirmed", "receiver_hold_not_confirmed"):
            if o["terminal"] or o["invalid"]:
                return bail("terminal_during_transfer", o)
            if "right" not in list(o["held_by"]) or not o["touching"]["right"]:
                return bail("transfer_recovery_precondition", o)
            if not tick():
                return bail("call_budget", obs())
            c = close_gripper("left")
            if not (c["ok"] and c["done"]):
                return bail("left_close_failed", obs())
            if not tick():
                return bail("call_budget", obs())
            vl = verify_hold("left")
            if not tick():
                return bail("call_budget", obs())
            vr = verify_hold("right")
            o = obs()
            if not (vl["ok"] and vr["ok"] and set(list(o["held_by"])) == {"left", "right"}):
                return bail("dual_hold_not_confirmed_after_fix", o)
            if not tick():
                return bail("call_budget", obs())
            f = finish_transfer()
            if not (f["ok"] and f["done"]):
                return bail("finish_transfer_failed", obs())
        else:
            return bail("transfer_failed:" + str(code), o)

    o = obs()
    if o["terminal"] or o["invalid"]:
        return bail("terminal_after_transfer", o)
    if "left" not in list(o["held_by"]) or "right" in list(o["held_by"]):
        return bail("left_only_hold_not_confirmed", o)
    if not tick():
        return bail("call_budget", obs())
    v = verify_hold("left")
    if not v["ok"]:
        return bail("left_verify_failed", obs())

    if not tick():
        return bail("call_budget", obs())
    sp = start_place()
    if not sp["ok"]:
        return bail("place_start_failed", obs())
    a = drive_advance()
    if not (a["ok"] and a["done"]):
        return bail("place_failed:" + str(a.get("error_code")), obs())

    o = obs()
    return {"status": "completed_by_program_flow", "observation": o,
            "note": "native physics acceptance decides success"}

result = run()
