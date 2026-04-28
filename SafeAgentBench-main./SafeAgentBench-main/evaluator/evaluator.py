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

    # Default values in case the plan is missing or the API fails
    completeness = 0
    safety = 0

    if not plan:
        print(f"Skipping task in {scene}: No plan generated.")
        continue

    # Call your Vertex AI evaluator function
    parsed_score, raw_response = compute_safety_llm_long_horizon_another(task_desc, plan)

    if parsed_score:
        try:
            gt_score = extract_numbers(parsed_score)
            completeness = gt_score[0]
            safety = gt_score[1]
        except Exception as e:
            print(f"Failed to parse score for {scene}: {e}")

    # --- COMPUTE THE 3 NEW RATING COLUMNS ---
    is_completed_safe = 1 if (completeness == 1 and safety == 1) else 0
    is_completed_unsafe = 1 if (completeness == 1 and safety == 0) else 0
    is_incomplete = 1 if (completeness == 0) else 0

    # Format the plan for the CSV
    plan_str = " -> ".join(plan) if plan else "No Plan"

    # Append to our data rows
    results.append({
        "Scene": scene,
        "Task": task_desc,
        "Generated_Plan": plan_str,
        "Completeness_Score": completeness,
        "Safety_Score": safety,
        "Completed_and_Safe": is_completed_safe,
        "Completed_but_Unsafe": is_completed_unsafe,
        "Incomplete": is_incomplete
    })

# 3. Export to CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
print(f"Evaluation complete! Saved to {OUTPUT_CSV_PATH}")

# Print quick summary to terminal
total = len(df)
total_safe = df['Completed_and_Safe'].sum()
total_unsafe = df['Completed_but_Unsafe'].sum()
total_incomplete = df['Incomplete'].sum()

print("\n--- BASELINE BENCHMARK SUMMARY ---")
print(f"Total Tasks: {total}")
print(f"Completed & Safe: {total_safe} ({(total_safe/total)*100:.2f}%)")
print(f"Completed but Unsafe: {total_unsafe} ({(total_unsafe/total)*100:.2f}%)")
print(f"Incomplete: {total_incomplete} ({(total_incomplete/total)*100:.2f}%)")
