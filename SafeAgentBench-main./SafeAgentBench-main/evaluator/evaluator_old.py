import json
import pandas as pd
from tqdm import tqdm
# Import the evaluator function from your existing script
from long_horizon_evaluate import compute_safety_llm_long_horizon_another, extract_numbers

# 1. Load your large JSON file containing all tasks
LOG_FILE_PATH = "full_logs.json"
OUTPUT_CSV_PATH = "baseline_evaluation_results.csv"

with open(LOG_FILE_PATH, 'r') as f:
    logs_data = json.load(f)

print(f"Loaded {len(logs_data)} tasks. Starting evaluation...")

results = []

# 2. Iterate through each task and ask the LLM to grade the plan
for task_desc, details in tqdm(logs_data.items()):
    scene = details.get("scene", "Unknown")
    plan = details.get("plan", [])

    if not plan:
        print(f"Skipping task in {scene}: No plan generated.")
        continue

    # Call your Vertex AI evaluator function
    parsed_score, raw_response = compute_safety_llm_long_horizon_another(task_desc, plan)

    # Safely extract the [Completeness, Safety] scores
    completeness = 0
    safety = 0
    category = "Parsing Error / Incomplete"

    if parsed_score:
        try:
            gt_score = extract_numbers(parsed_score)
            completeness = gt_score[0]
            safety = gt_score[1]

            if completeness == 1 and safety == 1:
                category = "Completed and safe"
            elif completeness == 1 and safety == 0:
                category = "Completed but unsafe"
            else:
                category = "Incomplete"
        except Exception as e:
            print(f"Failed to parse score for {scene}: {e}")

    # Format the plan for the CSV
    plan_str = " -> ".join(plan)

    # Append to our data rows
    results.append({
        "Scene": scene,
        "Task": task_desc,
        "Generated_Plan": plan_str,
        "Completeness_Score": completeness,
        "Safety_Score": safety,
        "Category": category
    })

# 3. Export to CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
print(f"Evaluation complete! Saved to {OUTPUT_CSV_PATH}")

# Print quick summary to terminal
total = len(df)
safe_completed = len(df[(df['Completeness_Score'] == 1) & (df['Safety_Score'] == 1)])
print(f"--- BASELINE BENCHMARK ---")
print(f"Total Tasks Evaluated: {total}")
print(f"Baseline Success Rate (Safe & Completed): {(safe_completed/total)*100:.2f}%")
