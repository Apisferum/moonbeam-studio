import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import orjson
import json
import orjson
import argparse
import subprocess
import time
import glob
import yaml
import numpy as np
from typing import Dict, Any, List
from dotenv import load_dotenv

# Clean repository-level sibling path resolution
studio_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if studio_root not in sys.path:
    sys.path.insert(0, studio_root)

# Moonbeam Foundation Model repository
codebase_root = os.environ.get("MOONBEAM_CODEBASE_PATH")

if not codebase_root:
    codebase_root = os.path.abspath(os.path.join(studio_root, "..", "moonbeam-codebase"))

if not os.path.isdir(codebase_root):
    raise FileNotFoundError(
        f"Moonbeam codebase not found: {codebase_root}"
    )

if codebase_root not in sys.path:
    sys.path.insert(0, codebase_root)

print(f"Moonbeam codebase: {codebase_root}")
# Import evaluation metrics
from eval.metrics.local_quality import harmony_consistency
from eval.metrics.structure import motif_recurrence_score
from eval.metrics.blueprint_adherence import chord_match_rate, voice_leading_score, density_adherence_metrics
from eval.metrics.efficiency import compute_efficiency_stats

# Try to import the project's composer and router
try:
    from engine.agentic_composer import AgenticComposer
    from engine.HarmonyRouter import HarmonyRouter
    from brain.structure_planner import StructurePlanner
    from eval.instrumentation.hooks import TraceContext
except ImportError as e:
    import traceback
    print(f"Error: Project import failed: {e}")
    traceback.print_exc()
    raise


def _select_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        print("WARNING: CUDA requested but not available - falling back to CPU.")
        return "cpu"
    except ImportError:
        return "cpu"


def _autodiscover_checkpoints():
    """
    Autodiscover checkpoint paths. Searches environment variables first,
    then Kaggle paths, then local workspace directories.
    """
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    local_roots = [
        workspace_root,
        os.path.join(workspace_root, "Moonbeam Pretrained Weights"),
        os.path.join(workspace_root, "moonbeam_chunk_20260716_140713"),
        os.path.join(workspace_root, "moonbeam_chunk_20260817_155517"),
        os.path.join(workspace_root, "moonbeam-codebase"),
        os.path.join(workspace_root, "Moonbeam Multi-Task Data"),
        "d:/scmoe",
        "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model"
    ]

    targets = {
        "BASE_MODEL_PATH": "moonbeam_839M.pt",
        "LORA_DIR": "epoch_4",
        "CONFIG_PATH": "model_config_multi_task.json",
        "MASTER_DICT_PATH": "indexed_tokens_dict.json",
    }

    resolved = {}
    for env_key, filename in targets.items():
        if os.environ.get(env_key):
            resolved[env_key] = os.environ[env_key]
            continue

        found = None
        # 1. Search Kaggle roots
        search_roots = ["/kaggle/input", "/kaggle/working"]
        for root in search_roots:
            if os.path.isdir(root):
                matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
                if matches:
                    found = matches[0]
                    break

        if not found:
            # 2. Search local workspace roots
            for root in local_roots:
                if os.path.isdir(root):
                    matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
                    if matches:
                        found = matches[0]
                        break

        if found:
            resolved[env_key] = found
        else:
            # 3. Development fallback
            candidates = []
            if env_key == "BASE_MODEL_PATH":
                candidates = [
                    os.path.join(workspace_root, "Moonbeam Pretrained Weights", "moonbeam_839M.pt"),
                    os.path.join(workspace_root, "moonbeam_checkpoint", "moonbeam_839M.pt"),
                    "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/moonbeam_checkpoint/moonbeam_839M.pt"
                ]
            elif env_key == "LORA_DIR":
                candidates = [
                    os.path.join(workspace_root, "moonbeam_chunk_20260716_140713"),
                    os.path.join(workspace_root, "moonbeam_chunk_20260817_155517"),
                    os.path.join(workspace_root, "multi_task_lora"),
                    os.path.join(workspace_root, "moonbeam_checkpoint", "multi_task_lora"),
                    "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/moonbeam_checkpoint/multi_task_lora"
                ]
            elif env_key == "CONFIG_PATH":
                candidates = [
                    os.path.join(workspace_root, "moonbeam-codebase", "src", "llama_recipes", "configs", "model_config_multi_task.json"),
                    os.path.join(workspace_root, "src", "llama_recipes", "configs", "model_config_multi_task.json"),
                    "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/model_config_multi_task.json"
                ]
            elif env_key == "MASTER_DICT_PATH":
                candidates = [
                    os.path.join(workspace_root, "Moonbeam Multi-Task Data", "ComMU", "indexed_tokens_dict.json"),
                    os.path.join(workspace_root, "processed", "ComMU", "indexed_tokens_dict.json"),
                    "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/processed/ComMU/indexed_tokens_dict.json"
                ]
            
            # Pick first existing path, otherwise use hardcoded fallback
            local_guess = candidates[-1]
            for c in candidates:
                if os.path.exists(c):
                    local_guess = c
                    break
            resolved[env_key] = local_guess

    # Special logic for parent folder of epochs
    if not os.environ.get("LORA_DIR"):
        for root in local_roots:
            if os.path.isdir(root):
                epochs = glob.glob(os.path.join(root, "**", "epoch_*"), recursive=True)
                if epochs:
                    resolved["LORA_DIR"] = os.path.dirname(epochs[0])
                    break

    return resolved


def _require_paths(paths: dict) -> None:
    missing = {k: v for k, v in paths.items() if not os.path.exists(v)}
    if missing:
        lines = [f"  - {k}: {v}" for k, v in missing.items()]
        msg = (
            "Error: Missing required checkpoint/config path(s):
" + "\n".join(lines) +
            "\n\nPlease set environment variables (BASE_MODEL_PATH, LORA_DIR, CONFIG_PATH, MASTER_DICT_PATH) "
            "manually or run from the root containing Moonbeam weights."
        )
        raise FileNotFoundError(msg)


def get_default_eval_set() -> List[Dict[str, Any]]:
    """Defines a fixed evaluation prompt set."""
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


def build_fallback_blueprint(prompt_dict: dict, use_ties: bool, single_adapter_weights: dict = None) -> dict:
    """
    Constructs a basic timeline directly from the prompt metadata without running StructurePlanner.
    This bypasses Markov chord generation and chord-to-tone realization.
    Crucially, note_events = [] so that no notes are forced (free-sampling).
    """
    global_key = prompt_dict.get("global_key", "C")
    global_mode = prompt_dict.get("global_mode", "major")
    bpm = prompt_dict.get("global_bpm", 120)
    style = prompt_dict.get("style", "pop")
    time_signature = prompt_dict.get("global_time_signature", "4/4")
    beats_per_bar = 4
    if "/" in time_signature:
        try:
            beats_per_bar = int(time_signature.split("/")[0])
        except ValueError:
            pass

    dense_timeline = []
    for s in prompt_dict.get("sections", []):
        name = s["name"]
        mood = s["mood"]
        length_str = s["length"]

        bars = 8
        if length_str == "short":
            bars = 4
        elif length_str == "medium":
            bars = 8
        elif length_str == "long":
            bars = 16

        instruments = s.get("instruments", ["Piano", "Strings"])

        if not use_ties and single_adapter_weights:
            ties = single_adapter_weights.copy()
        else:
            ties = {"commu_lora": 0.35, "emopia_lora": 0.25, "slakh_lora": 0.40}

        section_entry = {
            "section_name": name,
            "bars": bars,
            "key": global_key,
            "mode": global_mode,
            "bpm": bpm,
            "style": style,
            "mood": mood,
            "energy_curve": [0.5] * bars,
            "riff_mode": False,
            "solo_instrument": None,
            "time_signature": time_signature,
            "beats_per_bar": beats_per_bar,
            "progression": [],
            "target_chords": [],
            "chord_timeline": [],
            "target_instruments": instruments,
            "note_events": [],  # Empty means free-sampling!
            "ties_weights": ties,
            "density_curve": [0.5] * bars,
            "max_tokens": bars * 16 * max(1, len(instruments) // 2),
            "motif_action": "generate_and_save_seed" if name == "Intro" else "retrieve_seed"
        }
        dense_timeline.append(section_entry)

    return {
        "global_settings": {
            "key": global_key,
            "mode": global_mode,
            "bpm": bpm,
            "style": style,
            "time_signature": time_signature,
        },
        "timeline": dense_timeline
    }


def run_ablation_mock(cfg_path: str, prompt: Dict[str, Any], output_midi: str) -> Dict[str, Any]:
    """Simulates pipeline execution in mock mode and returns a mock PieceTrace dictionary."""
    time.sleep(0.05)  # Simulate latency
    
    # Generate mock midi file
    from eval.baselines.vanilla_moonbeam import generate_mock_midi
    generate_mock_midi(output_midi)
    
    # Construct mock trace JSON
    sections_trace = []
    for idx, s in enumerate(prompt["sections"]):
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
    parser.add_argument("--config", type=str, default=None, help="Specific configuration name to run (e.g. full_system, no_planner)")
    args = parser.parse_args()

    load_dotenv()

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

    if args.config:
        if args.config not in configs:
            print(f"Error: Invalid config name: {args.config}. Must be one of: {list(configs.keys())}")
            sys.exit(1)
        configs = {args.config: configs[args.config]}

    # Discover and boot model if not running in mock mode
    harmony_router = None
    paths = {}
    if not args.mock:
        print("Discovering checkpoints...")
        paths = _autodiscover_checkpoints()
        print("Resolved checkpoint paths:")
        for k, v in paths.items():
            status = "[OK]" if os.path.exists(v) else "[MISSING]"
            print(f"   {status} {k} = {v}")
        _require_paths(paths)

        if HarmonyRouter is None:
            raise ImportError("Cannot run live evaluation: project imports missing.")

        device = _select_device("cuda")
        print(f"
Booting HarmonyRouter (single instance) on {device}...")
        harmony_router = HarmonyRouter(
            base_model_path=paths["BASE_MODEL_PATH"],
            lora_checkpoint_dir=paths["LORA_DIR"],
            model_config_path=paths["CONFIG_PATH"],
            master_dict_path=paths["MASTER_DICT_PATH"],
            device=device,
        )

    results = {}

    for name, cfg_or_runner in configs.items():
        cfg_or_runner = os.path.abspath(os.path.join(studio_root, cfg_or_runner))
        print(f"\n========================================")
        print(f"RUNNING CONFIGURATION: {name}")
        print(f"========================================")
        
        results[name] = []
        
        # Load yaml config parameters
        use_planner = True
        use_motif_memory = True
        use_ties = True
        use_soft_refiner = True
        use_hard_scorer = True
        
        if cfg_or_runner.endswith(".yaml"):
            try:
                with open(cfg_or_runner, 'r') as f:
                    cfg = yaml.safe_load(f)
                use_planner = cfg.get("use_planner", True)
                use_motif_memory = cfg.get("use_motif_memory", True)
                use_ties = cfg.get("use_ties", True)
                use_soft_refiner = cfg.get("use_soft_refiner", True)
                use_hard_scorer = cfg.get("use_hard_scorer", True)
            except Exception as e:
                print(f"Warning: Error reading config file {cfg_or_runner}: {e}")

        # Deterministic single adapter weights: commu only
        single_adapter_weights = {"commu_lora": 1.0, "emopia_lora": 0.0, "slakh_lora": 0.0}

        for prompt in eval_set:
            prompt_id = prompt["id"]
            print(f"
Prompt {prompt_id}: '{prompt['prompt']}'")
            
            out_midi_path = os.path.abspath(os.path.join(args.results_dir, "midi", f"{name}_prompt_{prompt_id}.mid"))
            trace_json_path = os.path.abspath(os.path.join(args.results_dir, f"{name}_prompt_{prompt_id}_trace.json"))
            
            piece_trace = None
            blueprint = None
            
            if args.mock:
                # Mock run: simulate execution
                if cfg_or_runner.endswith(".yaml"):
                    piece_trace = run_ablation_mock(cfg_or_runner, prompt, out_midi_path)
                else:
                    cmd = [sys.executable, cfg_or_runner, "--prompt_json", trace_json_path, "--output_midi", out_midi_path, "--mock"]
                    with open(trace_json_path, 'w') as f:
                        json.dump({"dense_blueprint": {"timeline": prompt["sections"]}}, f)
                        
                    subprocess.run(cmd, check=True)
                    piece_trace = run_ablation_mock("", prompt, out_midi_path)
            else:
                # Real execution (requires model weights)
                if cfg_or_runner.endswith(".yaml"):
                    if AgenticComposer is None:
                        raise ImportError("Cannot run live evaluation: project imports missing.")
                    
                    # Instantiate AgenticComposer with this config's parameters
                    composer = AgenticComposer(
                        harmonyrouter=harmony_router,
                        acceptance_threshold=0.75,
                        use_planner=use_planner,
                        use_motif_memory=use_motif_memory,
                        use_ties=use_ties,
                        use_soft_refiner=use_soft_refiner,
                        use_hard_scorer=use_hard_scorer
                    )
                    
                    # Ensure mock LLM is false for real eval
                    composer.llm.use_mock = False
                    
                    # Generate intent
                    llm_intent = composer.llm.generate_intent(prompt["prompt"])
                    
                    # Plan blueprint
                    if use_planner:
                        print(f"   ↳ Planning structure blueprint...")
                        blueprint = composer.planner.plan(llm_intent)
                        # Override ties weights if use_ties is False (single_adapter)
                        if not use_ties:
                            for sec in blueprint["timeline"]:
                                sec["ties_weights"] = single_adapter_weights.copy()
                    else:
                        print(f"   ↳ Generating fallback non-planner blueprint (free-sampling)...")
                        blueprint = build_fallback_blueprint(prompt, use_ties, single_adapter_weights)
                        
                    # Execute composition
                    print(f"   ↳ Composing full song (real)...")
                    final_song_midi = composer.compose_full_song(
                        blueprint["timeline"],
                        prompt=prompt["prompt"]
                    )
                    
                    # Save MIDI
                    final_song_midi.write(out_midi_path)
                    print(f"   ↳ Wrote MIDI to {out_midi_path}")
                    
                    # Retrieve the trace from TraceContext hooks
                    piece_trace = TraceContext.get_instance().trace.to_dict()
                    
                    # Clear HarmonyRouter ties cache to free memory
                    harmony_router.clear_ties_cache()
                else:
                    # Execute baseline runner in real mode
                    print(f"   ↳ Executing baseline runner: {cfg_or_runner}")
                    cmd = [
                        sys.executable, cfg_or_runner,
                        "--prompt_json", trace_json_path,
                        "--output_midi", out_midi_path,
                        "--base_model_path", paths["BASE_MODEL_PATH"],
                        "--model_config_path", paths["CONFIG_PATH"],
                        "--tokenizer_path", paths["MASTER_DICT_PATH"]
                    ]
                    # Write prompt info so script has it
                    with open(trace_json_path, 'w') as f:
                        json.dump({"dense_blueprint": {"timeline": prompt["sections"]}}, f)
                        
                    try:
                        subprocess.run(cmd, check=True)
                        if os.path.exists(trace_json_path):
                            with open(trace_json_path, 'r') as f:
                                piece_trace = json.load(f)
                        else:
                            piece_trace = {
                                "prompt": prompt["prompt"],
                                "global_key": prompt["global_key"],
                                "global_mode": prompt["global_mode"],
                                "global_bpm": prompt["global_bpm"],
                                "global_time_signature": prompt["global_time_signature"],
                                "sections": [{"section_idx": i, "section_type": s["name"], "attempts": [{"accepted": True}]} for i, s in enumerate(prompt["sections"])]
                            }
                    except Exception as e:
                        print(f"⚠️ Baseline execution failed: {e}")
                        piece_trace = None
            
            if piece_trace:
                # Compute and attach metrics to this trace
                print("   ↳ Computing metrics...")
                
                # Diatonic harmony adherence (average across sections)
                print("   ↳ Calculating harmony adherence...")
                harmonies = [harmony_consistency(out_midi_path) for s in piece_trace["sections"]]
                
                # Calculate chord match rates based on planned blueprint
                if blueprint and "timeline" in blueprint:
                    chord_matches = [chord_match_rate(out_midi_path, s) for s in blueprint["timeline"]]
                else:
                    chord_matches = []
                    
                voice_leadings = [voice_leading_score(out_midi_path) for _ in prompt["sections"]]
                
                # Calculate density adherence metrics across sections
                all_y_true = []
                all_y_pred = []
                
                current_offset_seconds = 0.0
                for idx, s_trace in enumerate(piece_trace["sections"]):
                    target_template = s_trace.get("blueprint_rhythm_template", [])
                    bpm = prompt.get("global_bpm", 120)
                    bars = 8
                    if blueprint and "timeline" in blueprint and idx < len(blueprint["timeline"]):
                        bp_sec = blueprint["timeline"][idx]
                        bpm = bp_sec.get("bpm", bpm)
                        bars = bp_sec.get("bars", bars)
                    else:
                        if idx < len(prompt["sections"]):
                            p_sec = prompt["sections"][idx]
                            length_str = p_sec.get("length", "medium")
                            raw_bars = {"short": 4, "medium": 8, "long": 16}.get(length_str, 8)
                            bars = max(4, int(raw_bars * 1.5))
                        
                    res_density = density_adherence_metrics(
                        out_midi_path, 
                        target_template, 
                        bpm=bpm, 
                        bars=bars, 
                        beats_per_bar=4, 
                        time_offset=current_offset_seconds
                    )
                    all_y_true.extend(res_density.get("y_true", []))
                    all_y_pred.extend(res_density.get("y_pred", []))
                    
                    # Accumulate offset for the next section
                    beat_duration = 60.0 / bpm
                    bar_duration = 4 * beat_duration
                    current_offset_seconds += bars * bar_duration
                
                # Calculate global density metrics across the entire song
                from eval.metrics.regression import calculate_rmse, calculate_mae, calculate_r2
                if all_y_true and all_y_pred:
                    global_rmse = calculate_rmse(all_y_true, all_y_pred)
                    global_mae = calculate_mae(all_y_true, all_y_pred)
                    global_r2 = calculate_r2(all_y_true, all_y_pred)
                else:
                    global_rmse = 1.0
                    global_mae = 1.0
                    global_r2 = 0.0
                
                # Log metrics in trace dictionary for aggregation
                piece_trace["metrics"] = {
                    "harmony_consistency": float(np.mean(harmonies)) if harmonies else 1.0,
                    "chord_match_rate": float(np.mean(chord_matches)) if chord_matches else 1.0,
                    "voice_leading_score": float(np.mean(voice_leadings)) if voice_leadings else 1.0,
                    "motif_recurrence": float(motif_recurrence_score(piece_trace)),
                    "efficiency": compute_efficiency_stats(piece_trace["sections"]),
                    "density_rmse": float(global_rmse),
                    "density_mae": float(global_mae),
                    "density_r2": float(global_r2)
                }
                
                with open(trace_json_path, "wb") as f:
                    f.write(
                        orjson.dumps(
                            piece_trace,
                            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
                        )
                    )
                print(f"   ↳ Saved trace details to {trace_json_path}")
                
                results[name].append(piece_trace)

    # Save aggregated results
    agg_json_path = os.path.join(args.results_dir, "aggregate_metrics.json")
    with open(agg_json_path, "wb") as f:
        f.write(
            orjson.dumps(
                results,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY
            )
        )
    print(f"
Evaluation driver finished! Aggregated metrics saved to {agg_json_path}")

    # Build report tables
    try:
        from eval.report.aggregate import build_summary_table
        build_summary_table(agg_json_path, os.path.join(args.results_dir, "summary_table.md"))
    except Exception as e:
        print(f"Warning: Summary table aggregation skipped/failed: {e}")

if __name__ == "__main__":
    main()