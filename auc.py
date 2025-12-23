import json
import numpy as np
import glob
import os

def load_best_values_from_json(folder_path):
    """
    Finds the IOH .json file and extracts the 'best' -> 'y' value for every run.
    """
    # Recursive search in the specific run folder
    search_path = os.path.join(folder_path, "**", "*.json")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        raise FileNotFoundError(f"No .json file found in {folder_path}")
    
    json_file = files[0]
    print(f"Loading Log: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    final_values = []
    
    # Navigate the JSON structure
    if "scenarios" not in data:
         raise ValueError("JSON format incorrect: 'scenarios' key missing.")

    for scenario in data["scenarios"]:
        for run in scenario["runs"]:
            if "best" in run and "y" in run["best"]:
                final_values.append(run["best"]["y"])
            else:
                pass
                
    return np.array(final_values)

def calculate_normalized_auc(final_values):
    """
    Calculates the Area Under the ECDF Curve (AUC) normalized to [0, 1].
    Metric: Average Success Rate over log-spaced targets between Min and Max found.
    """
    if len(final_values) == 0:
        return 0.0, 0.0, 0.0

    # Filter valid values (ensure > 0 for log scale)
    # If values are 0.0 (global opt reached exactly), clamp them to a tiny epsilon
    # to avoid log(0) errors
    clean_values = np.maximum(final_values, 1e-10)
    
    best_val = np.min(clean_values)
    worst_val = np.max(clean_values)
    
    # Define Log-Spaced Targets
    # We evaluate performance across the entire range of difficulty
    # 1000 steps for high precision integration.
    targets = np.logspace(np.log10(best_val), np.log10(worst_val), 1000)
    
    # Calculate Success Rate for each target
    # Success Rate = (Count of runs better than target) / Total Runs
    success_rates = []
    for t in targets:
        rate = np.sum(clean_values <= t) / len(clean_values)
        success_rates.append(rate)
        
    # Integrate (Mean of Success Rates)
    # Since targets are the x-axis (normalized implicitly by the loop steps),
    # the mean height of the curve is the AUC
    auc = np.mean(success_rates)
    
    return auc, best_val, worst_val

if __name__ == "__main__":
    folder_name = r"C:\Users\Pimek\Documents\cards\EA_assignment_1\data\run_F23_stats-29" 
    
    try:
        data = load_best_values_from_json(folder_name)
        print(f"Runs found: {len(data)}")
        
        if len(data) > 0:
            auc, best, worst = calculate_normalized_auc(data)
            
            print(f"\n--- Results ---")
            print(f"Best Run (Delta f): {best:.4e}")
            print(f"Worst Run (Delta f): {worst:.4e}")
            print("-" * 30)
            print(f"FINAL AUC: {auc:.4f}")
            print("-" * 30)
        else:
            print("Error: JSON contained no runs.")
            
    except Exception as e:
        print(f"Error: {e}")