import json
import csv
import traceback
from pathlib import Path

from monitor import LLMRuleSynthesizer, TemporalMonitor


INPUT_FILE = "gen_plans.json"
OUTPUT_FILE = "verification_results.csv"


def load_generated_plans(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def main():
    plans = load_generated_plans(INPUT_FILE)

    synthesizer = LLMRuleSynthesizer()

    rows = []

    for item in plans:
        sample_id = item.get("id")
        task = item.get("task", "")
        plan = item.get("plan", [])

        print("\n" + "=" * 80)
        print(f"Sample {sample_id}")
        print("Task:", task)
        print("Plan:", plan)

        try:
            # 1. synthesize temporal rules from task + requirement
            rules = synthesizer.synthesize(task)

            # 2. verify generated plan
            monitor = TemporalMonitor(rules)
            result = monitor.verify_plan(plan)

            safe = result["safe"]
            violations = result["violations"]

            decision = "ALLOW" if safe else "HALT"

            row = {
                "id": sample_id,
                "task": task,
                "plan": json.dumps(plan, ensure_ascii=False),
                "decision": decision,
                "safe": safe,
                "num_rules": len(rules),
                "rules": json.dumps(rules_to_jsonable(rules), ensure_ascii=False),
                "num_violations": len(violations),
                "violations": json.dumps(violations, ensure_ascii=False),
                "error": "",
            }

            print("Decision:", decision)
            print("Safe:", safe)
            print("Violations:", violations)

        except Exception as e:
            row = {
                "id": sample_id,
                "task": task,
                "plan": json.dumps(plan, ensure_ascii=False),
                "decision": "ERROR",
                "safe": False,
                "num_rules": 0,
                "rules": "",
                "num_violations": 0,
                "violations": "",
                "error": str(e),
            }

            print("ERROR:", e)
            traceback.print_exc()

        rows.append(row)

        # save progress after each sample
        write_csv(rows, OUTPUT_FILE)

    print(f"\nSaved verification results to {OUTPUT_FILE}")


def write_csv(rows, path: str):
    if not rows:
        return

    fieldnames = [
        "id",
        "task",
        "plan",
        "decision",
        "safe",
        "num_rules",
        "rules",
        "num_violations",
        "violations",
        "error",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
