import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy import stats
from typing import Dict, Any, List

def compare_configs(metric_a: List[float], metric_b: List[float]) -> Dict[str, float]:
    """
    Computes significance tests (paired t-test and Wilcoxon signed-rank)
    between two lists of scores paired by prompt ID.
    """
    if len(metric_a) != len(metric_b):
        raise ValueError(f"Paired comparison requires same length vectors. Got {len(metric_a)} vs {len(metric_b)}")
        
    if len(metric_a) < 2:
        return {"paired_t_p": 1.0, "wilcoxon_p": 1.0}
        
    # Paired t-test
    try:
        t_stat, t_p = stats.ttest_rel(metric_a, metric_b)
    except Exception:
        t_p = 1.0
        
    # Wilcoxon signed-rank test
    try:
        w_stat, w_p = stats.wilcoxon(metric_a, metric_b)
    except Exception:
        # Wilcoxon can fail if differences are all identical or length is too small
        w_p = 1.0
        
    return {
        "paired_t_p": float(t_p) if not np.isnan(t_p) else 1.0,
        "wilcoxon_p": float(w_p) if not np.isnan(w_p) else 1.0
    }

def print_significance_report(agg_json_path: str):
    """
    Reads the aggregate JSON and prints a paired significance report 
    comparing the full system to other configurations.
    """
    import json
    import os
    
    if not os.path.exists(agg_json_path):
        return
        
    with open(agg_json_path, 'r') as f:
        results = json.load(f)
        
    if "full_system" not in results:
        print("❌ Cannot perform significance testing: 'full_system' config data missing.")
        return
        
    full_pieces = results["full_system"]
    full_scores = [np.mean([s.get("attempts", [{}])[-1].get("score", 1.0) for s in p.get("sections", [])]) for p in full_pieces]
    
    print("\n========================================")
    print("📊 SIGNIFICANCE TESTING vs FULL SYSTEM (Piece Quality)")
    print("========================================")
    
    for name, pieces in results.items():
        if name == "full_system" or not pieces:
            continue
            
        try:
            other_scores = [np.mean([s.get("attempts", [{}])[-1].get("score", 1.0) for s in p.get("sections", [])]) for p in pieces]
            comparison = compare_configs(full_scores, other_scores)
            
            p_t = comparison["paired_t_p"]
            p_w = comparison["wilcoxon_p"]
            
            sig_t = "★ Significant" if p_t < 0.05 else "Not Significant"
            sig_w = "★ Significant" if p_w < 0.05 else "Not Significant"
            
            print(f"\nConfiguration: {name}")
            print(f"   ↳ Paired t-test p-value: {p_t:.4f} ({sig_t})")
            print(f"   ↳ Wilcoxon p-value:       {p_w:.4f} ({sig_w})")
        except Exception as e:
            print(f"⚠️ Comparison with {name} failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="eval/results/aggregate_metrics.json")
    args = parser.parse_args()
    print_significance_report(args.input)
