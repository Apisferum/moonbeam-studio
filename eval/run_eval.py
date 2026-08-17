import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import argparse
import subprocess
import time
import numpy as np
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'moonbeam-studio')))

# Import evaluation metrics
from eval.metrics.local_quality import pitch_entropy, rhythm_entropy, harmony_consistency
from eval.metrics.drift import drift_slope, rejection_rate_vs_index
from eval.metrics.structure import motif_recurrence_score, self_similarity_matrix
from eval.metrics.blueprint_adherence import chord_match_rate, voice_leading_score, emotion_adherence, orchestration_match
from eval.metrics.boundary import boundary_metrics
from eval.metrics.efficiency import compute_efficiency_stats

# Try to import the project's composer
try:
    from engine.agentic_composer import AgenticComposer
    from engine.HarmonyRouter import HarmonyRouter
    from brain.structure_planner import StructurePlanner
except ImportError:
    AgenticComposer = None
    HarmonyRouter = None
    StructurePlanner = None

def get_default_eval_set() -> List[Dict[str, Any]]:
    """Defines a fixed evaluation prompt set (e.g. 5 prompts with 4 sections each for test, scalable to 20)."""
    return [
        {
            "id": 1,
            "prompt": "A happy energetic pop song in C major, 120 BPM.",
            "global_key": "C",
            "global_mode": "major",
            "global_bpm": 120,
            "global_time_signature": "4/4",
            "style": "pop",
            "sections": [
                {"name": "Intro", "mood": "happy", "length": "short"},
                {"name": "Verse", "mood": "happy", "length": "medium"},
                {"name": "Chorus", "mood": "happy", "length": "medium"},
                {"name": "Outro", "mood": "calm", "length": "short"}
            ]
        },
        {
            "id": 2,
            "prompt": "A melancholy sad piano theme in A minor, 80 BPM.",
            "global_key": "A",
            "global_mode": "minor",
            "global_bpm": 80,
            "global_time_signature": "4/4",
            "style": "classical",
            "sections": [
                {"name": "Theme", "mood": "sad", "length": "medium"},
                {"name": "Development", "mood": "sad", "length": "medium"},
                {"name": "Theme", "mood": "sad", "length": "medium"},
                {"name": "Outro", "mood": "grief", "length": "short"}
            ]
        },
        {
            "id": 3,
            "prompt": "A tense cinematic strings track in D minor, 110 BPM.",
            "global_key": "D",
            "global_mode": "minor",
            "global_bpm": 110,
            "global_time_signature": "4/4",
            "style": "cinematic",
            "sections": [
                {"name": "Intro", "mood": "tension", "length": "short"},
                {"name": "Verse", "mood": "tension", "length": "medium"},
                {"name": "Chorus", "mood": "angry", "length": "medium"},
                {"name": "Outro", "mood": "dark", "length": "short"}
            ]
        }
    ]

def run_ablation_mock(cfg_path: str, prompt: Dict[str, Any], output_midi: str) -> Dict[str, Any]:
    """Simulates pipeline execution in mock mode and returns a mock PieceTrace dictionary."""
    time.sleep(0.05)  # Simulate latency
    
    # Generate mock midi file
    from eval.baselines.vanilla_moonbeam import generate_mock_midi
    generate_mock_midi(output_midi)
    
    # Construct mock trace JSON
    sections_trace = []
    for idx, s in enumerate(prompt["sections"]):
        # Mock attempt traces
        attempts = [
            {
                "attempt_idx": 1,
                "score": 0.5,
                "feedback": {"metrics": {"chord_score": 0.4, "inst_score": 0.5}, "feedback": "weak chords"},
                "accepted": False,
                "temperature": 0.85,
                "token_count": 48
            },
            {
                "attempt_idx": 2,
                "score": 0.85,
                "feedback": {"metrics": {"chord_score": 0.85, "inst_score": 0.9, "voice_leading_score": 0.8, "rhythm_score": 0.85}, "feedback": "good"},
                "accepted": True,
                "temperature": 0.7,
                "token_count": 64
            }
        ]
        sections_trace.append({
            "section_idx": idx,
            "section_type": s["name"],
            "blueprint_chords": ["C", "G", "Am", "F"],
            "blueprint_rhythm_template": [0.5, 0.6, 0.7, 0.8],
            "blueprint_emotion_label": s["mood"],
            "blueprint_instrumentation": ["Piano", "Strings"],
            "retrieved_motifs": [{"name": "Intro", "similarity": 0.88, "index": 0}],
            "ties_weights": {"commu_lora": 0.45, "emopia_lora": 0.25, "slakh_lora": 0.3},
            "attempts": attempts,
            "final_midi_path": output_midi,
            "final_accept_attempt": 2,
            "latency_ms": 150.0,
            "kv_cache_reused": False
        })
        
    return {
        "prompt": prompt["prompt"],
        "global_key": prompt["global_key"],
        "global_mode": prompt["global_mode"],
        "global_bpm": prompt["global_bpm"],
        "global_time_signature": prompt["global_time_signature"],
        "sections": sections_trace
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluation Framework Driver")
    parser.add_argument("--mock", action="store_true", help="Run in mock simulation mode")
    parser.add_argument("--prompts_limit", type=int, default=3, help="Max number of prompts to run")
    parser.add_argument("--results_dir", type=str, default="eval/results", help="Directory to save evaluation results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "midi"), exist_ok=True)

    eval_set = get_default_eval_set()[:args.prompts_limit]

    configs = {
        "full_system": "eval/ablations/configs/full.yaml",
        "no_planner": "eval/ablations/configs/no_planner.yaml",
        "no_faiss": "eval/ablations/configs/no_faiss.yaml",
        "single_adapter": "eval/ablations/configs/single_adapter.yaml",
        "no_soft": "eval/ablations/configs/no_soft.yaml",
        "no_hard": "eval/ablations/configs/no_hard.yaml",
        "vanilla_moonbeam": "eval/baselines/vanilla_moonbeam.py",
        "midi_rwkv": "eval/baselines/midi_rwkv_runner.py",
        "music_transformer": "eval/baselines/museformer_runner.py",
        "hi_acg": "eval/baselines/hiacg_reimpl.py",
        "cascaded_diff": "eval/baselines/cascaded_diff_reimpl.py"
    }

    results = {}

    for name, cfg_or_runner in configs.items():
        print(f"\n========================================")
        print(f"🚀 RUNNING CONFIGURATION: {name}")
        print(f"========================================")
        
        results[name] = []
        
        for prompt in eval_set:
            prompt_id = prompt["id"]
            print(f"\n📝 Prompt {prompt_id}: '{prompt['prompt']}'")
            
            out_midi_path = os.path.abspath(os.path.join(args.results_dir, "midi", f"{name}_prompt_{prompt_id}.mid"))
            trace_json_path = os.path.abspath(os.path.join(args.results_dir, f"{name}_prompt_{prompt_id}_trace.json"))
            
            piece_trace = None
            
            if args.mock:
                # Mock run: simulate execution
                if cfg_or_runner.endswith(".yaml"):
                    piece_trace = run_ablation_mock(cfg_or_runner, prompt, out_midi_path)
                else:
                    # Run baseline runner in mock mode using subprocess
                    # This verifies baseline script functionality
                    cmd = [sys.executable, cfg_or_runner, "--prompt_json", trace_json_path, "--output_midi", out_midi_path, "--mock"]
                    # Write prompt info so script has it
                    with open(trace_json_path, 'w') as f:
                        json.dump({"dense_blueprint": {"timeline": prompt["sections"]}}, f)
                        
                    subprocess.run(cmd, check=True)
                    # Create mock trace for baseline
                    piece_trace = run_ablation_mock("", prompt, out_midi_path)
            else:
                # Real execution (requires model weights)
                if cfg_or_runner.endswith(".yaml"):
                    if AgenticComposer is None:
                        raise ImportError("Cannot run live evaluation: project imports missing.")
                    # Load config yaml, build planner, composer, and execute
                    # ...
                    pass
                else:
                    # Execute baseline runner
                    # ...
                    pass
            
            if piece_trace:
                # Compute and attach metrics to this trace
                print("   ↳ Computing metrics...")
                
                # Diatonic harmony adherence (average across sections)
                harmonies = [harmony_consistency(out_midi_path) for s in piece_trace["sections"]]
                chord_matches = [chord_match_rate(out_midi_path, s) for s in prompt["sections"]]
                voice_leadings = [voice_leading_score(out_midi_path) for _ in prompt["sections"]]
                
                # Log metrics in trace dictionary for aggregation
                piece_trace["metrics"] = {
                    "harmony_consistency": float(np.mean(harmonies)) if harmonies else 1.0,
                    "chord_match_rate": float(np.mean(chord_matches)) if chord_matches else 1.0,
                    "voice_leading_score": float(np.mean(voice_leadings)) if voice_leadings else 1.0,
                    "motif_recurrence": float(motif_recurrence_score(piece_trace)),
                    "efficiency": compute_efficiency_stats(piece_trace["sections"])
                }
                
                # Save trace file
                with open(trace_json_path, 'w') as f:
                    json.dump(piece_trace, f, indent=2)
                print(f"   ↳ Saved trace details to {trace_json_path}")
                
                results[name].append(piece_trace)

    # Save aggregated results
    agg_json_path = os.path.join(args.results_dir, "aggregate_metrics.json")
    with open(agg_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n🏆 Evaluation driver finished! Aggregated metrics saved to {agg_json_path}")

    # Build report tables
    try:
        from eval.report.aggregate import build_summary_table
        build_summary_table(agg_json_path, os.path.join(args.results_dir, "summary_table.md"))
    except Exception as e:
        print(f"⚠️ Summary table aggregation skipped/failed: {e}")

if __name__ == "__main__":
    main()
