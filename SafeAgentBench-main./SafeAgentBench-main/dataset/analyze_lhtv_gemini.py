#!/usr/bin/env python3
"""
Analyze a long-horizon task verification JSONL dataset using Gemini labels.

Input JSONL format, one object per line:
{"scene_name": "FloorPlan22_physics", "instruction": "put ... Requirement: ..."}

Outputs:
  output_dir/
    annotations.jsonl
    annotations.csv
    hazard_type_distribution.png
    strictness_distribution.png
    room_type_distribution.png
    hazard_by_strictness_heatmap.png
    verifier_capability_distribution.png
    safety_action_type_distribution.png
    severity_distribution.png
    complexity_distribution.png
    ambiguity_distribution.png
    room_by_hazard_heatmap.png
    allowed_delay_distribution.png
    object_role_top_items.png

Example:
  python analyze_lhtv_dataset_gemini.py \
    --input tasks.jsonl \
    --output_dir analysis_outputs \
    --credentials cred.json \
    --project_id robust-slice-493713-e1 \
    --location us-central1 \
    --model gemini-2.5-pro \
    --workers 4
"""

import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.oauth2 import service_account


DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_KEY_PATH = "cred.json"
DEFAULT_PROJECT_ID = "robust-slice-493713-e1"
DEFAULT_LOCATION = "us-central1"


# -----------------------------
# Vertex / Gemini setup
# -----------------------------

def init_vertexai(credentials_path: str, project_id: str, location: str) -> None:
    creds = service_account.Credentials.from_service_account_file(credentials_path)
    vertexai.init(
        project=project_id,
        location=location,
        credentials=creds,
    )


def call_vertexai(
    model: str,
    prompt: str,
    system_prompt: str = "You are a careful dataset annotation assistant.",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    retries: int = 3,
    sleep_seconds: float = 3.0,
) -> str:
    """
    Calls Gemini through Vertex AI, following the style of your evaluation code.
    Includes retries because batch labeling can hit transient API errors.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            gemini_model = GenerativeModel(
                model_name=model,
                system_instruction=system_prompt,
            )
            response = gemini_model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            output = response.text.strip() if response.text else ""
            print("\033[91mcalling finished...\033[0m")
            return output

        except Exception as exc:
            last_error = exc
            print(f"[WARN] Gemini call failed on attempt {attempt}/{retries}: {exc}")
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)

    raise RuntimeError(f"Gemini call failed after {retries} attempts: {last_error}")


# -----------------------------
# Dataset loading
# -----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                row["_row_id"] = line_idx
                rows.append(row)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx}: {exc}") from exc
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_for_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for key, value in row.items():
        if isinstance(value, (list, dict)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return

    all_keys = []
    seen = set()
    for row in rows:
        for key in flatten_for_csv(row).keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))


# -----------------------------
# Prompting and parsing
# -----------------------------

ANNOTATION_SCHEMA = {
    "room_type": "kitchen | bathroom | bedroom | living_room | office | unknown",
    "task_domain": ["cooking", "cleaning", "storage", "electronics", "object_manipulation"],
    "hazard_type": ["fire_overheating", "electrical", "water_overflow", "object_damage"],
    "hazard_source": ["stove burner"],
    "vulnerable_objects": ["book"],
    "protected_entity": ["human", "food", "object", "environment", "appliance"],
    "trigger_event": "short natural language description",
    "required_safety_actions": ["turn off stove burner"],
    "safety_action_type": ["turn_off", "remove_object", "close_object", "clean", "empty", "reposition", "avoid_until_safe"],
    "temporal_constraint_type": "before | immediately_after | within_n_steps | eventually_before_end | invariant_while | multi_action_sequence | unclear",
    "strictness": "precondition | immediate_after | bounded_delay | eventual | persistent_invariant | unclear",
    "max_allowed_steps": None,
    "violation_mode": "omission | wrong_order | delay_violation | unsafe_colocation | unsafe_state | unsafe_surface_action | unclear",
    "required_capabilities": [
        "object_state_tracking",
        "object_location_tracking",
        "container_content_tracking",
        "temporal_deadline_tracking",
        "causal_hazard_reasoning",
        "commonsense_material_reasoning",
        "multi_object_relation_reasoning",
        "action_precondition_checking",
        "postcondition_checking",
    ],
    "severity": "low | medium | high | critical",
    "recoverability": "easy | moderate | hard | irreversible",
    "constraint_complexity": 1,
    "ambiguity_level": "low | medium | high",
    "ambiguity_issues": ["typo", "unclear_object", "unclear_time_bound"],
    "template": "Turn off X within N steps after Y",
    "formal_rule": {
        "operator": "BEFORE | NEXT | WITHIN_N | UNTIL | EVENTUALLY | AND_SEQUENCE | UNCLEAR",
        "trigger": "predicate-like trigger",
        "requirement": "predicate-like requirement",
    },
    "one_sentence_rationale": "brief explanation",
}


def build_annotation_prompt(row: Dict[str, Any]) -> str:
    scene_name = row.get("scene_name", "")
    instruction = row.get("instruction", "")

    schema_text = json.dumps(ANNOTATION_SCHEMA, indent=2, ensure_ascii=False)

    return f"""
You are annotating a long-horizon embodied task verification dataset.

Each example contains:
- scene_name: the simulated floorplan.
- instruction: a task plus a safety Requirement.

Your job is to classify the safety requirement and the reasoning needed to verify it.

Return ONLY valid JSON.
Do not include markdown.
Do not include comments.
Do not include trailing commas.

Use this exact JSON schema shape:
{schema_text}

Important labeling guidelines:

1. hazard_type:
   Use a list. Choose one or more from:
   - fire_overheating
   - explosion
   - electrical
   - water_overflow
   - water_damage
   - object_damage
   - food_contamination
   - poisoning
   - waste_or_mess
   - physical_injury
   - containment_or_falling
   - hygiene_dust_dirt
   - appliance_damage
   - unclear

2. temporal_constraint_type:
   - before: required action must occur before a risky action.
   - immediately_after: required action must occur right after a trigger.
   - within_n_steps: requirement says within/no more than N steps.
   - eventually_before_end: required before task ends, but no tight deadline.
   - invariant_while: condition must remain true while another condition holds.
   - multi_action_sequence: multiple safety actions must happen in order.
   - unclear: timing is ambiguous.

3. strictness:
   - precondition: safety action before risky action.
   - immediate_after: must happen immediately after trigger.
   - bounded_delay: within N steps.
   - eventual: before the end or before later subtask, but not immediate.
   - persistent_invariant: must remain true during an interval.
   - unclear.

4. max_allowed_steps:
   - integer if the requirement says one/two/three/N steps.
   - 1 for immediately after or within one step.
   - null when no numeric step bound exists.

5. constraint_complexity:
   Use integer 1 to 5.
   1 = simple single object/order rule.
   3 = needs tracking several objects or a deadline.
   5 = multiple safety actions, hidden state, ambiguous wording, or cross-subtask dependency.

6. required_capabilities:
   Use a list. Choose any needed:
   - object_state_tracking
   - object_location_tracking
   - container_content_tracking
   - temporal_ordering
   - temporal_deadline_tracking
   - causal_hazard_reasoning
   - commonsense_material_reasoning
   - multi_object_relation_reasoning
   - action_precondition_checking
   - postcondition_checking
   - liquid_state_tracking
   - heat_source_tracking
   - appliance_state_tracking
   - affordance_reasoning

7. room_type:
   Infer from scene name and objects:
   - kitchen, bathroom, bedroom, living_room, office, unknown.
   Common AI2-THOR convention:
   FloorPlan1-30 = kitchen, 201-230 = living_room, 301-330 = bedroom, 401-430 = bathroom.

8. template:
   Write a short normalized pattern, e.g.
   - Turn off X within N steps after Y
   - Pick up X before Y
   - Close X immediately after Y
   - Empty/Clean X before Y
   - Move X to safe surface before Y

Example to annotate:

scene_name:
{scene_name}

instruction:
{instruction}
""".strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly parse Gemini output as JSON.
    If it accidentally includes extra text, extract the first {...} block.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text}")

    return json.loads(match.group(0))


def normalize_annotation(ann: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure fields exist and have predictable types for plotting.
    """
    defaults = {
        "room_type": "unknown",
        "task_domain": [],
        "hazard_type": ["unclear"],
        "hazard_source": [],
        "vulnerable_objects": [],
        "protected_entity": [],
        "trigger_event": "",
        "required_safety_actions": [],
        "safety_action_type": [],
        "temporal_constraint_type": "unclear",
        "strictness": "unclear",
        "max_allowed_steps": None,
        "violation_mode": "unclear",
        "required_capabilities": [],
        "severity": "medium",
        "recoverability": "moderate",
        "constraint_complexity": 3,
        "ambiguity_level": "medium",
        "ambiguity_issues": [],
        "template": "unclear",
        "formal_rule": {},
        "one_sentence_rationale": "",
    }

    out = dict(defaults)
    out.update(ann)

    list_fields = [
        "task_domain",
        "hazard_type",
        "hazard_source",
        "vulnerable_objects",
        "protected_entity",
        "required_safety_actions",
        "safety_action_type",
        "required_capabilities",
        "ambiguity_issues",
    ]

    for field in list_fields:
        if out[field] is None:
            out[field] = []
        elif isinstance(out[field], str):
            out[field] = [out[field]]
        elif not isinstance(out[field], list):
            out[field] = [str(out[field])]

    try:
        out["constraint_complexity"] = int(out["constraint_complexity"])
    except Exception:
        out["constraint_complexity"] = 3

    if out["max_allowed_steps"] in ["null", "None", "", "unknown"]:
        out["max_allowed_steps"] = None

    try:
        if out["max_allowed_steps"] is not None:
            out["max_allowed_steps"] = int(out["max_allowed_steps"])
    except Exception:
        out["max_allowed_steps"] = None

    return out


def annotate_one(
    row: Dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    prompt = build_annotation_prompt(row)

    output = call_vertexai(
        model=model,
        prompt=prompt,
        system_prompt="You are a careful and consistent dataset annotation assistant. Return only valid JSON.",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        ann = extract_json_object(output)
        ann = normalize_annotation(ann)
        error = None
    except Exception as exc:
        print(f"[ERROR] Failed to parse row {row.get('_row_id')}: {exc}")
        ann = normalize_annotation({})
        error = str(exc)

    result = dict(row)
    result.update(ann)
    result["raw_model_output"] = output
    result["annotation_error"] = error
    return result


def load_existing_annotations(path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Allows resuming without relabeling completed rows.
    """
    if not path.exists():
        return {}

    done = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "_row_id" in row:
                done[int(row["_row_id"])] = row
    return done


def annotate_dataset(
    rows: List[Dict[str, Any]],
    model: str,
    output_jsonl: Path,
    workers: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    resume: bool = True,
) -> List[Dict[str, Any]]:
    existing = load_existing_annotations(output_jsonl) if resume else {}
    results_by_id = dict(existing)

    to_process = [row for row in rows if int(row["_row_id"]) not in results_by_id]

    print(f"[INFO] Total rows: {len(rows)}")
    print(f"[INFO] Existing annotations: {len(existing)}")
    print(f"[INFO] Rows to annotate: {len(to_process)}")

    if not to_process:
        return [results_by_id[int(row["_row_id"])] for row in rows]

    # Append completed rows immediately so interrupted runs can resume.
    with open(output_jsonl, "a", encoding="utf-8") as f:
        if workers <= 1:
            for row in to_process:
                result = annotate_one(row, model, temperature, max_tokens)
                results_by_id[int(row["_row_id"])] = result
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
        else:
            with futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_row = {
                    executor.submit(annotate_one, row, model, temperature, max_tokens): row
                    for row in to_process
                }

                for future in futures.as_completed(future_to_row):
                    row = future_to_row[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(f"[ERROR] Unhandled failure on row {row.get('_row_id')}: {exc}")
                        result = dict(row)
                        result.update(normalize_annotation({}))
                        result["raw_model_output"] = ""
                        result["annotation_error"] = str(exc)

                    results_by_id[int(row["_row_id"])] = result
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

    return [results_by_id[int(row["_row_id"])] for row in rows]


# -----------------------------
# Plotting helpers
# -----------------------------

def ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def counter_from_list_field(rows: List[Dict[str, Any]], field: str) -> Counter:
    c = Counter()
    for row in rows:
        values = row.get(field, [])
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        for value in values:
            if value:
                c[str(value)] += 1
    return c


def counter_from_scalar_field(rows: List[Dict[str, Any]], field: str) -> Counter:
    c = Counter()
    for row in rows:
        value = row.get(field)
        if value is not None and value != "":
            c[str(value)] += 1
    return c


def save_bar(counter: Counter, title: str, xlabel: str, ylabel: str, path: Path, top_k: Optional[int] = None) -> None:
    if not counter:
        print(f"[WARN] No data for plot: {title}")
        return

    items = counter.most_common(top_k)
    labels = [x[0] for x in items]
    values = [x[1] for x in items]

    plt.figure(figsize=(max(8, len(labels) * 0.55), 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[PLOT] {path}")


def save_hist(values: List[int], title: str, xlabel: str, ylabel: str, path: Path, bins: Optional[List[int]] = None) -> None:
    if not values:
        print(f"[WARN] No data for plot: {title}")
        return

    plt.figure(figsize=(8, 5))
    if bins:
        plt.hist(values, bins=bins, align="left", rwidth=0.8)
    else:
        plt.hist(values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[PLOT] {path}")


def save_heatmap(
    rows: List[Dict[str, Any]],
    x_field: str,
    y_field: str,
    title: str,
    path: Path,
    x_is_list: bool = False,
    y_is_list: bool = False,
) -> None:
    x_values = []
    y_values = []
    matrix_counts = defaultdict(int)

    for row in rows:
        xs = row.get(x_field, [])
        ys = row.get(y_field, [])

        if not x_is_list:
            xs = [xs]
        elif isinstance(xs, str):
            xs = [xs]

        if not y_is_list:
            ys = [ys]
        elif isinstance(ys, str):
            ys = [ys]

        xs = [str(x) for x in xs if x not in [None, ""]]
        ys = [str(y) for y in ys if y not in [None, ""]]

        for x in xs:
            for y in ys:
                matrix_counts[(y, x)] += 1
                x_values.append(x)
                y_values.append(y)

    x_labels = sorted(set(x_values))
    y_labels = sorted(set(y_values))

    if not x_labels or not y_labels:
        print(f"[WARN] No data for heatmap: {title}")
        return

    matrix = []
    for y in y_labels:
        matrix.append([matrix_counts[(y, x)] for x in x_labels])

    plt.figure(figsize=(max(8, len(x_labels) * 0.7), max(5, len(y_labels) * 0.45)))
    plt.imshow(matrix, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(range(len(y_labels)), y_labels)
    plt.colorbar(label="Count")

    for i, y in enumerate(y_labels):
        for j, x in enumerate(x_labels):
            value = matrix[i][j]
            if value > 0:
                plt.text(j, i, str(value), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[PLOT] {path}")


def make_all_plots(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    save_bar(
        counter_from_list_field(rows, "hazard_type"),
        "Hazard Type Distribution",
        "Hazard type",
        "Count",
        output_dir / "hazard_type_distribution.png",
    )

    save_bar(
        counter_from_scalar_field(rows, "strictness"),
        "Temporal Strictness Distribution",
        "Strictness",
        "Count",
        output_dir / "strictness_distribution.png",
    )

    save_bar(
        counter_from_scalar_field(rows, "temporal_constraint_type"),
        "Temporal Constraint Type Distribution",
        "Temporal constraint type",
        "Count",
        output_dir / "temporal_constraint_type_distribution.png",
    )

    save_bar(
        counter_from_scalar_field(rows, "room_type"),
        "Room Type Distribution",
        "Room type",
        "Count",
        output_dir / "room_type_distribution.png",
    )

    save_heatmap(
        rows,
        x_field="strictness",
        y_field="hazard_type",
        x_is_list=False,
        y_is_list=True,
        title="Hazard Type by Temporal Strictness",
        path=output_dir / "hazard_by_strictness_heatmap.png",
    )

    save_bar(
        counter_from_list_field(rows, "required_capabilities"),
        "Required Verifier Capability Distribution",
        "Verifier capability",
        "Count",
        output_dir / "verifier_capability_distribution.png",
    )

    save_bar(
        counter_from_list_field(rows, "safety_action_type"),
        "Safety Action Type Distribution",
        "Safety action type",
        "Count",
        output_dir / "safety_action_type_distribution.png",
    )

    save_bar(
        counter_from_scalar_field(rows, "severity"),
        "Hazard Severity Distribution",
        "Severity",
        "Count",
        output_dir / "severity_distribution.png",
    )

    save_hist(
        [int(row["constraint_complexity"]) for row in rows if row.get("constraint_complexity") is not None],
        "Constraint Complexity Distribution",
        "Complexity score, 1-5",
        "Count",
        output_dir / "complexity_distribution.png",
        bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    )

    save_bar(
        counter_from_scalar_field(rows, "ambiguity_level"),
        "Ambiguity Level Distribution",
        "Ambiguity level",
        "Count",
        output_dir / "ambiguity_distribution.png",
    )

    save_heatmap(
        rows,
        x_field="room_type",
        y_field="hazard_type",
        x_is_list=False,
        y_is_list=True,
        title="Room Type by Hazard Type",
        path=output_dir / "room_by_hazard_heatmap.png",
    )

    delays = [
        int(row["max_allowed_steps"])
        for row in rows
        if row.get("max_allowed_steps") is not None
    ]
    if delays:
        save_hist(
            delays,
            "Max Allowed Safety Delay Distribution",
            "Max allowed steps",
            "Count",
            output_dir / "allowed_delay_distribution.png",
            bins=list(range(0, max(delays) + 3)),
        )

    object_counter = Counter()
    for field in ["hazard_source", "vulnerable_objects", "protected_entity"]:
        c = counter_from_list_field(rows, field)
        for key, value in c.items():
            object_counter[f"{field}: {key}"] += value

    save_bar(
        object_counter,
        "Top Object and Entity Roles",
        "Role: object/entity",
        "Count",
        output_dir / "object_role_top_items.png",
        top_k=25,
    )

    save_bar(
        counter_from_scalar_field(rows, "template"),
        "Requirement Template Distribution",
        "Template",
        "Count",
        output_dir / "template_distribution.png",
        top_k=25,
    )

    save_bar(
        counter_from_list_field(rows, "ambiguity_issues"),
        "Ambiguity / Data Quality Issue Distribution",
        "Issue type",
        "Count",
        output_dir / "ambiguity_issue_distribution.png",
    )


# -----------------------------
# Optional cheap rule-based fallback
# -----------------------------

def add_rule_based_room_type_if_missing(rows: List[Dict[str, Any]]) -> None:
    """
    Useful if Gemini returns unknown room type. Uses common AI2-THOR floorplan ranges.
    """
    for row in rows:
        if row.get("room_type") and row["room_type"] != "unknown":
            continue

        scene = row.get("scene_name", "")
        match = re.search(r"FloorPlan(\d+)", scene)
        if not match:
            row["room_type"] = "unknown"
            continue

        num = int(match.group(1))
        if 1 <= num <= 30:
            row["room_type"] = "kitchen"
        elif 201 <= num <= 230:
            row["room_type"] = "living_room"
        elif 301 <= num <= 330:
            row["room_type"] = "bedroom"
        elif 401 <= num <= 430:
            row["room_type"] = "bathroom"
        else:
            row["room_type"] = "unknown"


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSONL dataset.")
    parser.add_argument("--output_dir", default="analysis_outputs", help="Directory for annotations and plots.")

    parser.add_argument("--credentials", default=DEFAULT_KEY_PATH, help="Path to service account JSON, e.g. cred.json.")
    parser.add_argument("--project_id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)

    parser.add_argument("--workers", type=int, default=4, help="Parallel Gemini calls. Reduce if rate-limited.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--no_resume", action="store_true", help="Disable resume from existing annotations.jsonl.")
    parser.add_argument("--skip_llm", action="store_true", help="Use existing annotations.jsonl and only regenerate plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    annotation_jsonl = output_dir / "annotations.jsonl"
    annotation_csv = output_dir / "annotations.csv"

    rows = read_jsonl(args.input)

    if args.skip_llm:
        if not annotation_jsonl.exists():
            raise FileNotFoundError(f"--skip_llm requires existing file: {annotation_jsonl}")
        annotated = list(load_existing_annotations(annotation_jsonl).values())
        annotated = sorted(annotated, key=lambda x: int(x.get("_row_id", 0)))
    else:
        init_vertexai(args.credentials, args.project_id, args.location)

        annotated = annotate_dataset(
            rows=rows,
            model=args.model,
            output_jsonl=annotation_jsonl,
            workers=args.workers,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            resume=not args.no_resume,
        )

    add_rule_based_room_type_if_missing(annotated)

    # Re-write sorted final annotations for cleanliness.
    write_jsonl(annotated, str(annotation_jsonl))
    write_csv(annotated, str(annotation_csv))

    make_all_plots(annotated, output_dir)

    print("\n[DONE]")
    print(f"Annotations JSONL: {annotation_jsonl}")
    print(f"Annotations CSV:   {annotation_csv}")
    print(f"Plots directory:   {output_dir}")


if __name__ == "__main__":
    main()

