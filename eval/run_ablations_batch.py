import os
import sys
import subprocess

workspace_root = "/home/aashishbishow/ProjectX"
py_bin = "/home/aashishbishow/miniconda3/envs/moon/bin/python"

configs = [
    "no_planner",
    "single_adapter",
    "no_faiss",
    "no_soft",
    "no_hard",
    "vanilla_moonbeam"
]

print("=" * 65)
print("🚀 RUNNING FULL SYSTEM ABLATION MATRIX (TABLE 1)")
print("=" * 65, flush=True)

for cfg in configs:
    print(f"\n==================================================", flush=True)
    print(f"▶️ STARTING ABLATION: {cfg}", flush=True)
    print(f"==================================================", flush=True)
    
    cmd = [
        py_bin, "-u", "moonbeam-studio/eval/run_eval.py",
        "--prompts_limit", "1",
        "--max_attempts", "1",
        "--config", cfg
    ]
    
    try:
        subprocess.run(cmd, cwd=workspace_root, check=True)
        print(f"✅ Finished ablation: {cfg}", flush=True)
    except Exception as e:
        print(f"❌ Error during ablation {cfg}: {e}", flush=True)

print("\n" + "=" * 65)
print("🎉 ALL SYSTEM ABLATIONS FINISHED! CHECK eval/results/summary_table.md")
print("=" * 65, flush=True)
