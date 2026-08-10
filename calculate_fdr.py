import json
from pathlib import Path

def safe_div(n, d):
    return n / d if d > 0 else 0.0

def calculate_rates(tp, fp, total_true):
    total_passed = tp + fp
    ppv = safe_div(tp, total_passed)
    fdr = safe_div(fp, total_passed)
    occurrence = safe_div(total_passed, total_true)
    tpr = safe_div(tp, total_true)
    return ppv, fdr, total_passed, occurrence, tpr

def format_output(counts_dict, title):
    lines = [f"=== {title} ==="]
    for res, data in counts_dict.items():
        total_true = data.get("Total_True", 0)
        lines.append(f"\nResolution: {res} (Total True Topological Area: {total_true})")
        for stage in ["Stage2", "Stage3"]:
            tp = data[stage]["TP"]
            fp = data[stage]["FP"]
            ppv, fdr, total_passed, occurrence, tpr = calculate_rates(tp, fp, total_true)
            lines.append(f"  {stage}:")
            lines.append(f"    TP: {tp}, FP: {fp} (Total Passed: {total_passed})")
            lines.append(f"    FDR (False Discovery Rate): {fdr*100:.2f}%")
            lines.append(f"    PPV (Positive Predictive Value): {ppv*100:.2f}%")
            lines.append(f"    Occurrence (Passed / True Area): {occurrence*100:.2f}%")
            lines.append(f"    TPR / Sensitivity (TP / True Area): {tpr*100:.2f}%")
    lines.append("")
    return "\n".join(lines)

def main():
    base_dir = Path("Data")
    aggregate_counts = {}
    
    # Process each realization
    for d in base_dir.glob("*/"):
        if not d.is_dir():
            continue
            
        json_path = d / "tp_fp_counts.json"
        if not json_path.exists():
            continue
            
        with open(json_path, "r") as f:
            counts = json.load(f)
            
        # Write individual probabilities
        prob_text = format_output(counts, f"Probabilities for {d.name}")
        with open(d / "probabilities.txt", "w") as f:
            f.write(prob_text)
            
        # Accumulate for aggregate
        for res, data in counts.items():
            if res not in aggregate_counts:
                aggregate_counts[res] = {
                    "Total_True": 0,
                    "Stage2": {"TP": 0, "FP": 0},
                    "Stage3": {"TP": 0, "FP": 0}
                }
            aggregate_counts[res]["Total_True"] += data.get("Total_True", 0)
            for stage in ["Stage2", "Stage3"]:
                aggregate_counts[res][stage]["TP"] += data[stage]["TP"]
                aggregate_counts[res][stage]["FP"] += data[stage]["FP"]
                
    if aggregate_counts:
        # Write aggregate probabilities
        agg_text = format_output(aggregate_counts, "Aggregate Probabilities (Over all realizations)")
        with open(base_dir / "aggregate_probabilities.txt", "w") as f:
            f.write(agg_text)
        print("Aggregate calculation complete!")
        print(agg_text)

if __name__ == "__main__":
    main()
