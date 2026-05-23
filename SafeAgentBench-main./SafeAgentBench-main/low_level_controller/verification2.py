"""
verification2.py

Batch verifier for generated high-level plans using monitor2.py.

Input format:
[
  {"id": 0, "task": "... Requirement: ...", "plan": ["find faucet", "turn on faucet"]}
]

Outputs:
1. verification2_results.csv
2. verification2_results.json
3. verification2_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

from monitor2 import LLMRuleSynthesizer, TemporalMonitor, rules_to_jsonable, GEMINI_MODEL


DEFAULT_INPUT_FILE = "gen_plans.json"
DEFAULT_OUTPUT_CSV = "verification2_results.csv"
DEFAULT_OUTPUT_JSON = "verification2_results.json"
DEFAULT_SUMMARY_JSON = "verification2_summary.json"
DEFAULT_RULE_CACHE = "verification2_rule_cache.json"


def load_generated_plans(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rule_cache(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_rule_cache(path: str, cache: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def temporal_rules_from_json(rule_dicts: List[Dict[str, Any]]):
    from monitor2 import TemporalRule, validate_rules
    rules = [
        TemporalRule(
            id=r.get("id", "unnamed_rule"),
            type=r.get("type", "unknown"),
            description=r.get("description", ""),
            trigger=r.get("trigger"),
            required=r.get("required"),
            target=r.get("target"),
            required_before=r.get("required_before"),
            forbidden=r.get("forbidden"),
            object=r.get("object"),
            location=r.get("location"),
            max_steps=r.get("max_steps"),
            severity=r.get("severity", "high"),
            source=r.get("source", "cache"),
        )
        for r in rule_dicts
    ]
    rules, _ = validate_rules(rules)
    return rules


def verify_one(
    item: Dict[str, Any],
    synthesizer: LLMRuleSynthesizer,
    cache: Dict[str, Any],
    use_cache: bool = True,
    structural_checks: bool = False,
) -> Dict[str, Any]:
    sample_id = item.get("id")
    task = item.get("task", "")
    plan = item.get("plan", [])
    if not isinstance(plan, list):
        plan = []
    if use_cache and task in cache:
        rules = temporal_rules_from_json(cache[task].get("rules", []))
        synthesis_warnings = cache[task].get("synthesis_warnings", [])
    else:
        rules = synthesizer.synthesize(task)
        synthesis_warnings = getattr(synthesizer, "last_warnings", [])
        cache[task] = {"rules": rules_to_jsonable(rules), "synthesis_warnings": synthesis_warnings}
    monitor = TemporalMonitor(rules, structural_checks=structural_checks)
    result = monitor.verify_plan(plan)
    safe = result["safe"]
    return {
        "id": sample_id,
        "scene": item.get("scene", ""),
        "task": task,
        "plan": plan,
        "normalized_plan": result.get("normalized_plan", []),
        "decision": "ALLOW" if safe else "HALT",
        "safe": safe,
        "num_rules": len(rules),
        "rules": rules_to_jsonable(rules),
        "rule_types": [r.type for r in rules],
        "ltl_like_rules": result.get("ltl_like_rules", []),
        "num_violations": len(result.get("violations", [])),
        "violations": result.get("violations", []),
        "num_warnings": len(result.get("warnings", [])),
        "warnings": result.get("warnings", []),
        "synthesis_warnings": synthesis_warnings,
        "error": "",
    }


def write_csv(rows: List[Dict[str, Any]], path: str):
    if not rows:
        return
    fieldnames = [
        "id", "scene", "task", "plan", "normalized_plan", "decision", "safe",
        "num_rules", "rule_types", "rules", "ltl_like_rules", "num_violations",
        "violations", "num_warnings", "warnings", "synthesis_warnings", "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key in ["plan", "normalized_plan", "rule_types", "rules", "ltl_like_rules", "violations", "warnings", "synthesis_warnings"]:
                csv_row[key] = json.dumps(csv_row.get(key, []), ensure_ascii=False)
            writer.writerow(csv_row)


def write_json(rows: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    decisions = Counter(row.get("decision") for row in rows)
    rule_types = Counter()
    violation_rule_types = Counter()
    violation_rule_ids = Counter()
    for row in rows:
        for t in row.get("rule_types", []):
            rule_types[t] += 1
        for v in row.get("violations", []):
            violation_rule_types[v.get("type", "unknown")] += 1
            violation_rule_ids[v.get("rule_id", "unknown")] += 1
    halted = decisions.get("HALT", 0)
    allowed = decisions.get("ALLOW", 0)
    errors = decisions.get("ERROR", 0)
    return {
        "total": total,
        "allowed": allowed,
        "halted": halted,
        "errors": errors,
        "halt_rate": round(halted / total, 4) if total else 0.0,
        "allow_rate": round(allowed / total, 4) if total else 0.0,
        "rule_type_counts": dict(rule_types),
        "violation_type_counts": dict(violation_rule_types),
        "top_violation_rule_ids": violation_rule_ids.most_common(20),
    }


def write_summary(summary: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--summary_json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--rule_cache", default=DEFAULT_RULE_CACHE)
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--structural_checks", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    plans = load_generated_plans(args.input)
    if args.limit is not None:
        plans = plans[:args.limit]

    cache = load_rule_cache(args.rule_cache)
    synthesizer = LLMRuleSynthesizer(model_type=args.model)
    rows = []

    for idx, item in enumerate(plans):
        sample_id = item.get("id", idx)
        task = item.get("task", "")
        plan = item.get("plan", [])
        print("\n" + "=" * 90)
        print(f"[{idx + 1}/{len(plans)}] Sample {sample_id}")
        print("Task:", task)
        print("Plan:", plan)
        try:
            row = verify_one(
                item=item,
                synthesizer=synthesizer,
                cache=cache,
                use_cache=not args.no_cache,
                structural_checks=args.structural_checks,
            )
            print("Decision:", row["decision"])
            print("Safe:", row["safe"])
            print("Rules:", row["num_rules"])
            print("Violations:", row["num_violations"])
        except Exception as e:
            row = {
                "id": sample_id,
                "scene": item.get("scene", ""),
                "task": task,
                "plan": plan,
                "normalized_plan": [],
                "decision": "ERROR",
                "safe": False,
                "num_rules": 0,
                "rules": [],
                "rule_types": [],
                "ltl_like_rules": [],
                "num_violations": 0,
                "violations": [],
                "num_warnings": 0,
                "warnings": [],
                "synthesis_warnings": [],
                "error": str(e),
            }
            print("ERROR:", e)
            traceback.print_exc()
        rows.append(row)
        write_csv(rows, args.output_csv)
        write_json(rows, args.output_json)
        save_rule_cache(args.rule_cache, cache)
        write_summary(build_summary(rows), args.summary_json)

    summary = build_summary(rows)
    write_summary(summary, args.summary_json)
    print("\n" + "=" * 90)
    print("Finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved CSV to: {args.output_csv}")
    print(f"Saved JSON to: {args.output_json}")
    print(f"Saved summary to: {args.summary_json}")
    print(f"Saved rule cache to: {args.rule_cache}")


if __name__ == "__main__":
    main()
