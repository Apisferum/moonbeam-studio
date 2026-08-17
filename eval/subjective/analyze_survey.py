import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from scipy import stats

def analyze_survey_results(csv_path: str):
    """
    Parses survey response CSV, extracts ratings per configuration,
    and runs paired Wilcoxon signed-rank significance tests.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Survey responses file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"📊 Analyzing {len(df)} collected subjective ratings...")

    # Group ratings by configuration
    # For each response, we have ratings for config_a and config_b
    configs = set(df["config_a"].unique()) | set(df["config_b"].unique())
    
    metrics = ["musicality", "naturalness", "boundary", "blueprint", "coherence"]
    
    # Compile flat scores per configuration
    scores_by_config = {cfg: {m: [] for m in metrics} for cfg in configs}
    
    for _, row in df.iterrows():
        cfg_a = row["config_a"]
        cfg_b = row["config_b"]
        
        for m in metrics:
            scores_by_config[cfg_a][m].append(row[f"{m}_a"])
            scores_by_config[cfg_b][m].append(row[f"{m}_b"])

    print("\n========================================")
    print("📈 AVERAGE SUBJECTIVE RATINGS BY CONFIG (MOS)")
    print("========================================")
    for cfg in configs:
        print(f"\nConfiguration: {cfg}")
        for m in metrics:
            avg = np.mean(scores_by_config[cfg][m]) if scores_by_config[cfg][m] else 0.0
            print(f"   ↳ {m.capitalize()}: {avg:.2f}")

    # Paired Wilcoxon tests comparing full_system vs other configurations
    # Find responses where full_system was compared side-by-side with another config
    if "full_system" not in configs:
        return

    print("\n========================================")
    print("🎭 PAIRED WILCOXON SIGNED-RANK SIGNIFICANCE TESTS")
    print("========================================")
    
    for other_cfg in configs:
        if other_cfg == "full_system":
            continue
            
        print(f"\nComparing [full_system] vs [{other_cfg}]:")
        
        # Filter rows where A=full and B=other OR A=other and B=full
        comparison_rows = df[
            ((df["config_a"] == "full_system") & (df["config_b"] == other_cfg)) |
            ((df["config_a"] == other_cfg) & (df["config_b"] == "full_system"))
        ]
        
        if len(comparison_rows) < 2:
            print("   ↳ Not enough paired comparisons to run Wilcoxon test (requires >= 2).")
            continue
            
        for m in metrics:
            full_val = []
            other_val = []
            for _, row in comparison_rows.iterrows():
                if row["config_a"] == "full_system":
                    full_val.append(row[f"{m}_a"])
                    other_val.append(row[f"{m}_b"])
                else:
                    full_val.append(row[f"{m}_b"])
                    other_val.append(row[f"{m}_a"])
            
            try:
                stat, p_val = stats.wilcoxon(full_val, other_val)
                sig = "★ Significant" if p_val < 0.05 else "Not Significant"
                print(f"   ↳ {m.capitalize()}: Wilcoxon p = {p_val:.4f} ({sig})")
            except Exception as e:
                print(f"   ↳ {m.capitalize()}: Test failed (difference is likely zero)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="eval/results/survey_responses.csv")
    args = parser.parse_args()
    analyze_survey_results(args.csv)
