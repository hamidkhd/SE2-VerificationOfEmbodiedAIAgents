import json
import time

from ai2thor.controller import Controller
from low_level_controller import LowLevelPlanner
from vlm_agent2 import Agents
from utils import all_objs


SCENE = "FloorPlan22_physics"
TASK = "Dirty the cup,then fill the cup with water."


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

    # ---- gather scene info for the VLM baseline ----
    objs_all = all_objs(env)
    objects_metadata = env.last_event.metadata.get("objects", [])
    img = env.last_event.frame

    encoded_img = Agents.ndarray_to_base64(img)
    agent = Agents(encoded_img, TASK)

    # ---- VLM baseline generates the plan ----
    env_info, raw_plan, plan = agent.multi_agent_vision_planning(
        objs_from_scene=objs_all,
        objects_metadata=objects_metadata,
        model_type="gemini-2.5-flash",
    )

    print("\n=== TASK ===")
    print(TASK)

    print("\n=== ENVIRONMENT INFO ===")
    print(env_info)

    print("\n=== RAW VLM RESPONSE ===")
    print(raw_plan)

    print("\n=== NORMALIZED HIGH-LEVEL PLAN ===")
    for i, step in enumerate(plan):
        print(f"{i}. {step}")

    # ---- same execution/logging pipeline as before ----
    trace = {
        "scene": SCENE,
        "task": TASK,
        "environment_info": env_info,
        "vlm_raw_response": raw_plan,
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

        # optional: stop on first failure
        # if not ret.get("success", False):
        #     break

    with open("trace_vlm.json", "w") as f:
        json.dump(trace, f, indent=2)

    env.stop()
    print("\nSaved trace to trace_vlm.json")


if __name__ == "__main__":
    main()
