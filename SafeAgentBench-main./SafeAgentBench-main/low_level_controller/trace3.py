import json
import time
import os

from ai2thor.controller import Controller
from low_level_controller import LowLevelPlanner
from vlm_agent2 import Agents
from utils import all_objs

from monitor import LLMRuleSynthesizer, TemporalMonitor


MODEL_TYPE = "gemini-2.5-pro"
MAX_FEEDBACK_ATTEMPTS = 3
HARD_HALT = False  # True = stop immediately if unsafe, False = give feedback and retry


def snapshot_state(event):
    metadata = event.metadata
    return {
        "lastAction": metadata.get("lastAction"),
        "lastActionSuccess": metadata.get("lastActionSuccess"),
        "errorMessage": metadata.get("errorMessage"),
        "inventoryObjects": metadata.get("inventoryObjects", []),
        "objects": metadata.get("objects", []),
    }


def rules_to_jsonable(rules):
    return [
        {
            "id": r.id,
            "type": r.type,
            "description": r.description,
            "trigger": r.trigger,
            "required": r.required,
            "target": r.target,
            "required_before": r.required_before,
            "max_steps": r.max_steps,
        }
        for r in rules
    ]


def build_feedback(original_task, plan, verification_result):
    violations = verification_result.get("violations", [])

    return f"""
Original task:
{original_task}

Your previous generated plan was unsafe:
{plan}

Verifier detected these violations:
{json.dumps(violations, indent=2)}

Regenerate a new plan that satisfies the original task and all safety requirements.
Do not explain. Return only the corrected executable plan.
"""


def generate_plan(task_for_agent, original_task, encoded_img, objs_all, objects_metadata):
    agent = Agents(encoded_img, task_for_agent)

    env_info, raw_plan, plan = agent.multi_agent_vision_planning(
        objs_from_scene=objs_all,
        objects_metadata=objects_metadata,
        model_type=MODEL_TYPE,
    )

    return {
        "task_for_agent": task_for_agent,
        "original_task": original_task,
        "environment_info": env_info,
        "raw_plan": raw_plan,
        "plan": plan,
    }


def verify_plan(task, plan, synthesizer):
    rules = synthesizer.synthesize(task)
    monitor = TemporalMonitor(rules)
    result = monitor.verify_plan(plan)

    return {
        "decision": "ALLOW" if result["safe"] else "HALT",
        "safe": result["safe"],
        "rules": rules_to_jsonable(rules),
        "violations": result["violations"],
    }


def one_pipeline(
    task: str,
    scene: str,
    trace_file: str = "trace_vlm_verified_dataset.json",
    hard_halt: bool = HARD_HALT,
    max_feedback_attempts: int = MAX_FEEDBACK_ATTEMPTS,
):
    env = Controller()
    env.reset(scene=scene)

    planner = LowLevelPlanner(env)
    planner.restore_scene()

    try:
        objs_all = all_objs(env)
        objects_metadata = env.last_event.metadata.get("objects", [])
        img = env.last_event.frame
        encoded_img = Agents.ndarray_to_base64(img)

        synthesizer = LLMRuleSynthesizer()

        trace = {
            "scene": scene,
            "task": task,
            "hard_halt": hard_halt,
            "max_feedback_attempts": max_feedback_attempts,
            "planning_attempts": [],
            "final_verification": None,
            "final_plan": None,
            "initial_state": snapshot_state(env.last_event),
            "steps": [],
        }

        current_task_for_agent = task
        final_plan_package = None
        final_verification = None

        for attempt in range(max_feedback_attempts + 1):
            print("\n" + "-" * 60)
            print(f"PLANNING ATTEMPT {attempt}")
            print("-" * 60)

            plan_package = generate_plan(
                task_for_agent=current_task_for_agent,
                original_task=task,
                encoded_img=encoded_img,
                objs_all=objs_all,
                objects_metadata=objects_metadata,
            )

            plan = plan_package["plan"]

            print("\n=== RAW VLM RESPONSE ===")
            print(plan_package["raw_plan"])

            print("\n=== NORMALIZED PLAN ===")
            for i, step in enumerate(plan):
                print(f"{i}. {step}")

            verification = verify_plan(task, plan, synthesizer)

            print("\n=== VERIFICATION RESULT ===")
            print("Decision:", verification["decision"])
            print("Safe:", verification["safe"])
            print("Violations:", verification["violations"])

            attempt_log = {
                "attempt": attempt,
                "task_for_agent": current_task_for_agent,
                "environment_info": plan_package["environment_info"],
                "vlm_raw_response": plan_package["raw_plan"],
                "plan": plan,
                "verification": verification,
            }

            trace["planning_attempts"].append(attempt_log)

            if verification["safe"]:
                final_plan_package = plan_package
                final_verification = verification
                break

            if hard_halt:
                final_plan_package = plan_package
                final_verification = verification
                break

            if attempt >= max_feedback_attempts:
                final_plan_package = plan_package
                final_verification = verification
                break

            current_task_for_agent = build_feedback(task, plan, verification)

        trace["final_verification"] = final_verification
        trace["final_plan"] = final_plan_package["plan"] if final_plan_package else None

        if final_verification is None:
            trace["status"] = "error_no_verification"
            save_trace(trace_file, task, trace)
            return False

        if not final_verification["safe"]:
            trace["status"] = "halted_by_verifier"
            print("\nHARD HALT: Plan is unsafe. Execution will not start.")
            save_trace(trace_file, task, trace)
            return False

        print("\n=== EXECUTION STARTED ===")

        for step_id, inst in enumerate(final_plan_package["plan"]):
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

        trace["status"] = "executed_after_verification"
        save_trace(trace_file, task, trace)

        return True

    finally:
        env.stop()


def save_trace(trace_file, task, trace):
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

            print("\n" + "=" * 60)
            print(f"Task {i}")
            print(f"Scene: {scene}")
            print(f"Task : {task}")

            try:
                result = one_pipeline(
                    scene=scene,
                    task=task,
                    trace_file="trace_vlm_verified_dataset.json",
                    hard_halt=HARD_HALT,
                    max_feedback_attempts=MAX_FEEDBACK_ATTEMPTS,
                )
                print(f"Task {i} final status: {result}")
            except Exception as e:
                print(f"Task {i} failed: {e}")

            i += 1


if __name__ == "__main__":
    main()
