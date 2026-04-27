import json
import time
import os

from ai2thor.controller import Controller
from low_level_controller import LowLevelPlanner
from vlm_agent2 import Agents
from utils import all_objs





def snapshot_state(event):
    metadata = event.metadata
    return {
        "lastAction": metadata.get("lastAction"),
        "lastActionSuccess": metadata.get("lastActionSuccess"),
        "errorMessage": metadata.get("errorMessage"),
        "inventoryObjects": metadata.get("inventoryObjects", []),
        "objects": metadata.get("objects", []),
    }


def one_pipeline(task: str, scene: str, trace_file: str = "trace_vlm_dataset.json"):
    env = Controller()
    env.reset(scene=scene)
    planner = LowLevelPlanner(env)
    planner.restore_scene()

    try:
        # ---- gather scene info for the VLM baseline ----
        objs_all = all_objs(env)
        objects_metadata = env.last_event.metadata.get("objects", [])
        img = env.last_event.frame

        encoded_img = Agents.ndarray_to_base64(img)
        agent = Agents(encoded_img, task)

        # ---- VLM baseline generates the plan ----
        env_info, raw_plan, plan = agent.multi_agent_vision_planning(
            objs_from_scene=objs_all,
            objects_metadata=objects_metadata,
            model_type="gemini-2.5-pro",
        )

        print("\n=== TASK ===")
        print(task)

        print("\n=== ENVIRONMENT INFO ===")
        print(env_info)

        print("\n=== RAW VLM RESPONSE ===")
        print(raw_plan)

        print("\n=== NORMALIZED HIGH-LEVEL PLAN ===")
        for i, step in enumerate(plan):
            print(f"{i}. {step}")

        # ---- execution/logging pipeline ----
        trace = {
            "scene": scene,
            "task": task,
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

        # ---- load existing traces and append/update this task ----
        if os.path.exists(trace_file):
            with open(trace_file, "r") as f:
                try:
                    all_traces = json.load(f)
                except json.JSONDecodeError:
                    all_traces = {}
        else:
            all_traces = {}

        all_traces[task] = trace

        with open(trace_file, "w") as f:
            json.dump(all_traces, f, indent=2)

        return True

    finally:
        env.stop()



def main():
    dataset_path = "../dataset/long_horizon_1009.jsonl"

    i = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[{i}] Skipping invalid JSON")
                continue

            scene = item.get("scene_name")
            task = item.get("instruction")

            if not scene or not task:
                print(f"[{i}] Missing fields, skipping")
                continue

            print("\n" + "="*60)
            print(f"Task {i}")
            print(f"Scene: {scene}")
            print(f"Task : {task}")

            try:
                result = one_pipeline(scene=scene, task=task)
                print(f"Task {i} final status: {result}")
            except Exception as e:
                print(f"Task {i} failed: {e}")

            i += 1


if __name__ == "__main__":
    main()
