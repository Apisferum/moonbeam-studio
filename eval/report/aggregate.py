import json
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from typing import Dict, Any

def build_summary_table(agg_json_path: str, output_md_path: str):
    """
    Parses the evaluation results from agg_json_path and writes a summary
    table in markdown format comparing all runs.
    """
    if not os.path.exists(agg_json_path):
        print(f"❌ Aggregated metrics file not found: {agg_json_path}")
        return

    with open(agg_json_path, 'r') as f:
        results = json.load(f)

    summary = []
    
    # Metrics to aggregate
    headers = [
        "Configuration",
        "Chord Match Rate",
        "Harmony Adherence",
        "Voice Leading",
        "Motif Recurrence",
        "Density RMSE",
        "Density R^2",
        "Avg Attempts",
        "Avg Latency (s)"
    ]

    for config_name, pieces in results.items():
        if not pieces:
            continue
            
        chord_scores = []
        harmony_scores = []
        vl_scores = []
        motif_scores = []
        density_rmse_scores = []
        density_r2_scores = []
        attempts = []
        latencies = []
        
        for piece in pieces:
            metrics = piece.get("metrics", {})
            chord_scores.append(metrics.get("chord_match_rate", 1.0))
            harmony_scores.append(metrics.get("harmony_consistency", 1.0))
            vl_scores.append(metrics.get("voice_leading_score", 1.0))
            motif_scores.append(metrics.get("motif_recurrence", 1.0))
            density_rmse_scores.append(metrics.get("density_rmse", 1.0))
            density_r2_scores.append(metrics.get("density_r2", 0.0))
            
            efficiency = metrics.get("efficiency", {})
            attempts.append(efficiency.get("mean_attempts_per_section", 1.0))
            latencies.append(efficiency.get("total_latency_ms", 0.0) / 1000.0)
            
        row = [
            config_name,
            f"{np.mean(chord_scores)*100:.1f}%" if chord_scores else "N/A",
            f"{np.mean(harmony_scores)*100:.1f}%" if harmony_scores else "N/A",
            f"{np.mean(vl_scores)*100:.2f}" if vl_scores else "N/A",
            f"{np.mean(motif_scores)*100:.1f}%" if motif_scores else "N/A",
            f"{np.mean(density_rmse_scores):.4f}" if density_rmse_scores else "N/A",
            f"{np.mean(density_r2_scores):.4f}" if density_r2_scores else "N/A",
            f"{np.mean(attempts):.2f}" if attempts else "N/A",
            f"{np.mean(latencies):.2f}s" if latencies else "N/A"
        ]
        summary.append(row)

    # Generate Markdown Table
    md_content = []
    md_content.append("# Evaluation Metrics Summary Table\n")
    md_content.append("| " + " | ".join(headers) + " |")
    md_content.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary:
        md_content.append("| " + " | ".join(row) + " |")
        
    md_content.append("\n*Generated automatically by Moonbeam Evaluation aggregate parser.*")

    with open(output_md_path, 'w') as f:
        f.write("\n".join(md_content) + "\n")
        
    print(f"🏆 Successfully built summary table at {output_md_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="eval/results/aggregate_metrics.json")
    parser.add_argument("--output", type=str, default="eval/results/summary_table.md")
    args = parser.parse_args()
    build_summary_table(args.input, args.output)
