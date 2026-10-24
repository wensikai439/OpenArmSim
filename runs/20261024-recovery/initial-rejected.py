obs = observe()
if obs["held_by"] != "none" and obs["held_by"] != "right":
    raise ValueError("unexpected initial hold state")
if obs["last_failure"] is not None:
    raise ValueError("initial last_failure present")

grasps = propose_grasps()
cands = grasps["candidates"]
if len(cands) == 0:
    raise ValueError("no grasp candidates")
g = cands[0]
gid = g["id"]

r = pregrasp(gid)
if not r["ok"] or not r["done"]:
    raise ValueError("pregrasp failed: " + str(r["error_code"]))

r = approach(gid)
if not r["ok"] or not r["done"]:
    raise ValueError("approach failed: " + str(r["error_code"]))

r = close_gripper(arm="right")
if not r["ok"]:
    raise ValueError("close failed: " + str(r["error_code"]))

v = verify_hold(arm="right")
if not v["ok"]:
    raise ValueError("verify_hold failed")

r = lift()
if not r["ok"] or not r["done"]:
    raise ValueError("lift failed: " + str(r["error_code"]))

r = start_transfer(placement_aware=True)
if not r["ok"]:
    raise ValueError("start_transfer failed: " + str(r["error_code"]))

done = False
for i in range(16):
    a = advance()
    if not a["ok"]:
        raise ValueError("transfer advance failed: " + str(a["error_code"]))
    if a["done"]:
        done = True
        break
if not done:
    raise ValueError("transfer did not complete in 16 advances")

r = finish_transfer()
if not r["ok"] or not r["done"]:
    raise ValueError("finish_transfer failed: " + str(r["error_code"]))

r = start_place()
if not r["ok"]:
    raise ValueError("start_place failed: " + str(r["error_code"]))

done = False
for i in range(16):
    a = advance()
    if not a["ok"]:
        raise ValueError("place advance failed: " + str(a["error_code"]))
    if a["done"]:
        done = True
        break
if not done:
    raise ValueError("placement did not complete in 16 advances")

final = observe()
receipt = {
    "grasp_id": gid,
    "held_by": final["held_by"],
    "completed_steps": final["completed_steps"],
    "operation": final["operation"],
    "last_failure": final["last_failure"],
    "remaining_seconds": final["remaining_seconds"],
}
receipt
