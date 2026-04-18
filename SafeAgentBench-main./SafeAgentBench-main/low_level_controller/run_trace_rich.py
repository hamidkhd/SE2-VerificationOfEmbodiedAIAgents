import os
import json
import time
from copy import deepcopy

from ai2thor.controller import Controller
from low_level_controller import LowLevelPlanner


OUT_DIR = "trace_output"
SCENE = "FloorPlan1_physics"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur


def simplify_agent(metadata):
    agent = metadata.get("agent", {})
    return {
        "position": agent.get("position"),
        "rotation": agent.get("rotation"),
        "cameraHorizon": agent.get("cameraHorizon"),
        "isStanding": agent.get("isStanding"),
    }


def simplify_inventory(metadata):
    inv = metadata.get("inventoryObjects", [])
    return [
        {
            "objectId": x.get("objectId"),
            "objectType": x.get("objectType"),
        }
        for x in inv
    ]


def simplify_object(obj):
    # Keep the fields most useful for verification and debugging.
    return {
        "objectId": obj.get("objectId"),
        "objectType": obj.get("objectType"),
        "name": obj.get("name"),
        "visible": obj.get("visible"),
        "distance": obj.get("distance"),
        "position": obj.get("position"),
        "rotation": obj.get("rotation"),
        "parentReceptacles": obj.get("parentReceptacles"),
        "receptacleObjectIds": obj.get("receptacleObjectIds"),
        "openable": obj.get("openable"),
        "isOpen": obj.get("isOpen"),
        "toggleable": obj.get("toggleable"),
        "isToggled": obj.get("isToggled"),
        "breakable": obj.get("breakable"),
        "isBroken": obj.get("isBroken"),
        "pickupable": obj.get("pickupable"),
        "isPickedUp": obj.get("isPickedUp"),
        "sliceable": obj.get("sliceable"),
        "isSliced": obj.get("isSliced"),
        "cookable": obj.get("cookable"),
        "isCooked": obj.get("isCooked"),
        "dirtyable": obj.get("dirtyable"),
        "isDirty": obj.get("isDirty"),
        "canFillWithLiquid": obj.get("canFillWithLiquid"),
        "isFilledWithLiquid": obj.get("isFilledWithLiquid"),
        "fillLiquid": obj.get("fillLiquid"),
        "canBeUsedUp": obj.get("canBeUsedUp"),
        "isUsedUp": obj.get("isUsedUp"),
        "mass": obj.get("mass"),
        "salientMaterials": obj.get("salientMaterials"),
        "temperature": obj.get("temperature"),
    }


def snapshot_state(event, include_all_objects=True):
    metadata = event.metadata
    objects = metadata.get("objects", [])

    state = {
        "lastAction": metadata.get("lastAction"),
        "lastActionSuccess": metadata.get("lastActionSuccess"),
        "errorMessage": metadata.get("errorMessage"),
        "agent": simplify_agent(metadata),
        "inventory": simplify_inventory(metadata),
        "visible_object_ids": [o.get("objectId") for o in objects if o.get("visible")],
        "visible_object_types": [o.get("objectType") for o in objects if o.get("visible")],
        "objects": [simplify_object(o) for o in objects] if include_all_objects else [],
    }
    return state


def index_objects(state):
    return {obj["objectId"]: obj for obj in state.get("objects", [])}


def diff_objects(prev_state, curr_state):
    prev_idx = index_objects(prev_state)
    curr_idx = index_objects(curr_state)

    changed = []
    all_ids = set(prev_idx.keys()) | set(curr_idx.keys())

    tracked_keys = [
        "visible",
        "distance",
        "position",
        "rotation",
        "parentReceptacles",
        "receptacleObjectIds",
        "isOpen",
        "isToggled",
        "isBroken",
        "isPickedUp",
        "isSliced",
        "isCooked",
        "isDirty",
        "isFilledWithLiquid",
        "fillLiquid",
        "isUsedUp",
        "temperature",
    ]

    for oid in all_ids:
        prev_obj = prev_idx.get(oid)
        curr_obj = curr_idx.get(oid)

        if prev_obj is None:
            changed.append({
                "objectId": oid,
                "change_type": "added",
                "after": curr_obj,
            })
            continue

        if curr_obj is None:
            changed.append({
                "objectId": oid,
                "change_type": "removed",
                "before": prev_obj,
            })
            continue

        field_changes = {}
        for key in tracked_keys:
            if prev_obj.get(key) != curr_obj.get(key):
                field_changes[key] = {
                    "before": prev_obj.get(key),
                    "after": curr_obj.get(key),
                }

        if field_changes:
            changed.append({
                "objectId": oid,
                "objectType": curr_obj.get("objectType"),
                "changes": field_changes,
            })

    return changed


def save_frame(event, path):
    # event.frame is a numpy RGB image in AI2-THOR.
    try:
        from PIL import Image
        img = Image.fromarray(event.frame)
        img.save(path)
    except Exception as e:
        print(f"Warning: could not save frame to {path}: {e}")


def main():
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "frames"))

    env = Controller()
    env.reset(scene=SCENE)

    planner = LowLevelPlanner(env)
    planner.restore_scene()

    # Replace with any valid plan you want.
    plan = [
        "find fridge",
        "open fridge",
        "find egg",
        "pick egg", 
        "throw egg"
    ]

    trace = {
        "scene": SCENE,
        "timestamp_start": time.time(),
        "plan": plan,
        "initial_state": snapshot_state(env.last_event, include_all_objects=True),
        "steps": [],
    }

    save_frame(env.last_event, os.path.join(OUT_DIR, "frames", "step_initial.png"))

    prev_state = deepcopy(trace["initial_state"])

    for step_id, instruction in enumerate(plan):
        t0 = time.time()
        result = planner.llm_skill_interact(instruction)
        t1 = time.time()

        curr_state = snapshot_state(env.last_event, include_all_objects=True)
        changed_objects = diff_objects(prev_state, curr_state)

        step_record = {
            "step": step_id,
            "instruction": instruction,
            "time_start": t0,
            "time_end": t1,
            "duration_sec": t1 - t0,
            "result": result,
            "state_after": curr_state,
            "changed_objects": changed_objects,
        }

        trace["steps"].append(step_record)
        save_frame(env.last_event, os.path.join(OUT_DIR, "frames", f"step_{step_id:03d}.png"))

        prev_state = deepcopy(curr_state)

        print(f"[{step_id}] {instruction} -> success={result['success']}")

    trace["timestamp_end"] = time.time()

    with open(os.path.join(OUT_DIR, "trace.json"), "w") as f:
        json.dump(trace, f, indent=2)

    env.stop()
    print(f"Saved rich trace to {os.path.join(OUT_DIR, 'trace.json')}")


if __name__ == "__main__":
    main()
