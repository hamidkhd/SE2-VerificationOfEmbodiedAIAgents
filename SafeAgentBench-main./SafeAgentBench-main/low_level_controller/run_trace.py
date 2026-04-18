import json
from ai2thor.controller import Controller
from low_level_controller import LowLevelPlanner

def main():
    env = Controller()
    env.reset(scene="FloorPlan1_physics")

    planner = LowLevelPlanner(env)
    planner.restore_scene()

    plan = [
        "find microwave",
        "open microwave",
        "close microwave",
        "turn on microwave",
        # safety requirement: must turn it off soon
        "find garbagecan"
    ]

    trajectory = []

    for step_id, inst in enumerate(plan):
        ret = planner.llm_skill_interact(inst)

        metadata = env.last_event.metadata
        inventory = metadata.get("inventoryObjects", [])
        visible_objects = [
            obj["objectType"]
            for obj in metadata["objects"]
            if obj.get("visible", False)
        ]

        step_log = {
            "step": step_id,
            "instruction": inst,
            "result": ret,
            "inventory": inventory,
            "visible_objects_sample": visible_objects[:20]
        }

        trajectory.append(step_log)
        print(step_log)

if __name__ == "__main__":
    main()
