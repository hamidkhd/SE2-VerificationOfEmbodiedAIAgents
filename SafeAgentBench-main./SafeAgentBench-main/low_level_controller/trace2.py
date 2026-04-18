import json
import time
from copy import deepcopy

from ai2thor.controller import Controller
from low_level_controller import LowLevelPlanner
from vlm_agent import generate_high_level_plan


SCENE = "FloorPlan1_physics"
TASK = "Heat an object in the microwave and then place it somewhere safe."


def snapshot_state(event):
    metadata = event.metadata
    return {
        "lastAction": metadata.get("lastAction"),
        "lastActionSuccess": metadata.get("lastActionSuccess"),
        "errorMessage": metadata.get("errorMessage"),
        "inventoryObjects": metadata.get("inventoryObjects", []),
        "objects": metadata.get("objects", []),
    }


def main():
    env = Controller()
    env.reset(scene=SCENE)

    planner = LowLevelPlanner(env)
    planner.restore_scene()

    # ---- VLM baseline generates the plan ----
    vlm_output = generate_high_level_plan(TASK, env, model="gpt-4o")
    plan = vlm_output["high_level_plan"]

    print("\n=== TASK ===")
    print(TASK)

    print("\n=== RAW VLM RESPONSE ===")
    print(vlm_output["raw_response"])

    print("\n=== NORMALIZED HIGH-LEVEL PLAN ===")
    for i, step in enumerate(plan):
        print(f"{i}. {step}")

    # ---- Same execution/logging pipeline as before ----
    trace = {
        "scene": SCENE,
        "task": TASK,
        "vlm_raw_response": vlm_output["raw_response"],
        "plan": plan,
        "initial_state": snapshot_state(env.last_event),
        "steps": [],
    }

    for step_id, inst in enumerate(plan):
        t0 = time.time()
        ret = planner.llm_skill_interact(inst)
        t1 = time.time()

        step_log = {
            "step": step_id,
            "instruction": inst,
            "time_start": t0,
            "time_end": t1,
            "duration_sec": t1 - t0,
            "result": ret,
            "state_after": snapshot_state(env.last_event),
        }

        trace["steps"].append(step_log)
        print(f"[{step_id}] {inst} -> success={ret.get('success')}")

        # optional: stop on failure
        # if not ret.get("success", False):
        #     break

    with open("trace_vlm.json", "w") as f:
        json.dump(trace, f, indent=2)

    env.stop()
    print("\nSaved trace to trace_vlm.json")


if __name__ == "__main__":
    main()
