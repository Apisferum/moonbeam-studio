import json
import os
import sys
import glob
import argparse
import logging
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from engine.HarmonyRouter import HarmonyRouter
from engine.agentic_composer import AgenticComposer
from shared.music_theory_constants import logger


def _select_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("⚠️ CUDA requested but not available — falling back to CPU.")
        return "cpu"
    except ImportError:
        return "cpu"


def _autodiscover_checkpoints():
    """
    Search common roots for the checkpoint set, so you don't have to hardcode
    a Kaggle dataset slug that changes every time you re-upload or re-attach
    a dataset. Search order:
      1. Explicit env vars (BASE_MODEL_PATH etc.) — always wins if set.
      2. /kaggle/input/**  — any attached Kaggle Dataset.
      3. /kaggle/working/** — anything downloaded into the working dir this session.
      4. The original hardcoded local dev path (GIDEON) — last resort.
    Returns a dict of the four paths, best-effort. Missing entries stay None
    and get caught by _require_paths with a clear message, instead of
    silently falling through to a wrong machine's path.
    """
    search_roots = ["/kaggle/input", "/kaggle/working"]
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    targets = {
        "BASE_MODEL_PATH": "moonbeam_839M.pt",
        "LORA_DIR": "multi_task_lora",
        "CONFIG_PATH": "model_config_multi_task.json",
        "MASTER_DICT_PATH": "indexed_tokens_dict.json",
    }

    resolved = {}
    for env_key, filename in targets.items():
        # 1. Explicit env var always wins.
        if os.environ.get(env_key):
            resolved[env_key] = os.environ[env_key]
            continue

        found = None
        # 2 & 3. Search Kaggle roots recursively for the target file/dir.
        for root in search_roots:
            if not os.path.isdir(root):
                continue
            matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
            if matches:
                found = matches[0]
                break

        if found:
            resolved[env_key] = found
        else:
            # 4. Fall back to the original local dev path as a last resort,
            # checking dynamic sibling folders first.
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

    return resolved


def _require_paths(paths: dict) -> None:
    missing = {k: v for k, v in paths.items() if not os.path.exists(v)}
    if missing:
        lines = [f"  - {k}: {v}" for k, v in missing.items()]
        msg = (
            "❌ Missing required checkpoint/config path(s):\n" + "\n".join(lines) +
            "\n\nOn Kaggle: run `!ls /kaggle/input/` to see attached datasets, confirm "
            "the checkpoint files are actually inside one of them, or set the matching "
            "env var explicitly (BASE_MODEL_PATH, LORA_DIR, CONFIG_PATH, MASTER_DICT_PATH) "
            "before running this script."
        )
        logger.error(msg)
        sys.exit(msg)


def _chunk_and_render_ddsp(midi_path, ddsp_output_dir, env, chunk_seconds=20.0, overlap_seconds=1.0):
    """
    Render a MIDI file through DDSP in short time-windows to avoid OOM in
    ddsp.core.upsample_with_windows / overlap_and_add, whose intermediate
    tensor scales with total frame count (i.e. song length). Chunks are
    rendered separately then crossfaded back together.
    """
    import pretty_midi
    import subprocess
    import shutil
    import os
    from pydub import AudioSegment

    pm = pretty_midi.PrettyMIDI(midi_path)
    total_duration = pm.get_end_time()

    if total_duration <= chunk_seconds:
        # Short enough to render in one shot.
        subprocess.run(
            ["midi_ddsp_synthesize", "--midi_path", midi_path, "--output_dir", ddsp_output_dir],
            check=True, capture_output=True, text=True, timeout=600, env=env,
        )
        return

    chunk_dir = os.path.join(os.path.dirname(ddsp_output_dir) or ".", "_ddsp_chunks")
    if os.path.exists(chunk_dir):
        shutil.rmtree(chunk_dir)
    os.makedirs(chunk_dir, exist_ok=True)

    starts = []
    t = 0.0
    while t < total_duration:
        starts.append(t)
        t += chunk_seconds

    rendered_segments = []
    for i, start in enumerate(starts):
        end = min(start + chunk_seconds + overlap_seconds, total_duration)

        chunk_pm = pretty_midi.PrettyMIDI()
        for inst in pm.instruments:
            new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            for note in inst.notes:
                if note.start < end and note.end > start:
                    new_inst.notes.append(pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=max(0.0, note.start - start),
                        end=min(end, note.end) - start,
                    ))
            if new_inst.notes:
                chunk_pm.instruments.append(new_inst)

        if not chunk_pm.instruments:
            continue

        chunk_midi_path = os.path.join(chunk_dir, f"chunk_{i:03d}.mid")
        chunk_pm.write(chunk_midi_path)

        chunk_out_dir = os.path.join(chunk_dir, f"out_{i:03d}")
        os.makedirs(chunk_out_dir, exist_ok=True)

        print(f"      ↳ Rendering chunk {i+1}/{len(starts)} ({start:.1f}s–{end:.1f}s)...")
        subprocess.run(
            ["midi_ddsp_synthesize", "--midi_path", chunk_midi_path, "--output_dir", chunk_out_dir],
            check=True, capture_output=True, text=True, timeout=600, env=env,
        )

        chunk_wavs = sorted(f for f in os.listdir(chunk_out_dir) if f.endswith(".wav"))
        if not chunk_wavs:
            continue

        chunk_mix = AudioSegment.silent(duration=int(end - start) * 1000 + 1000)
        for f in chunk_wavs:
            chunk_mix = chunk_mix.overlay(AudioSegment.from_wav(os.path.join(chunk_out_dir, f)))

        rendered_segments.append((start, chunk_mix))

    if not rendered_segments:
        return

    os.makedirs(ddsp_output_dir, exist_ok=True)
    crossfade_ms = int(overlap_seconds * 1000)
    full_mix = rendered_segments[0][1]
    for start, seg in rendered_segments[1:]:
        full_mix = full_mix.append(seg, crossfade=min(crossfade_ms, len(full_mix), len(seg)))

    full_mix.export(os.path.join(ddsp_output_dir, "full_mix.wav"), format="wav")

    # Cleanup chunk directory
    if os.path.exists(chunk_dir):
        try:
            shutil.rmtree(chunk_dir)
        except Exception:
            pass


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    load_dotenv()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Moonbeam AI Record Label Pipeline")
    parser.add_argument("--prompt", type=str, default="Generate a 2-minute epic cinematic track.", help="The user prompt")
    parser.add_argument("--output", type=str, default="masterpiece.mid", help="Output MIDI filename")
    parser.add_argument("--mock_llm", action="store_true", help="Use mock LLM instead of live API tiers")
    parser.add_argument("--use_mock_llm", type=str2bool, nargs='?', const=True, default=False, help="Use mock LLM instead of live API tiers")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--continue_midi", type=str, default=None, help="Path to a seed MIDI file to continue generating from.")
    parser.add_argument("--skip_render", action="store_true", help="Skip the DDSP neural audio render, MIDI only")
    args = parser.parse_args()

    args.mock_llm = args.mock_llm or args.use_mock_llm
    device = _select_device(args.device)

    # Resolve checkpoint paths: explicit env vars > Kaggle auto-discovery > local dev fallback.
    paths = _autodiscover_checkpoints()
    print("📂 Resolved checkpoint paths:")
    for k, v in paths.items():
        status = "✅" if os.path.exists(v) else "❌"
        print(f"   {status} {k} = {v}")

    if not args.mock_llm:
        _require_paths(paths)

    print("\n🚀 Booting HarmonyRouter...")
    try:
        harmony_router = HarmonyRouter(
            base_model_path=paths["BASE_MODEL_PATH"],
            lora_checkpoint_dir=paths["LORA_DIR"],
            model_config_path=paths["CONFIG_PATH"],
            master_dict_path=paths["MASTER_DICT_PATH"],
            device=device,
            use_mock=args.mock_llm,
        )
    except Exception as e:
        logger.error(f"❌ Failed to boot HarmonyRouter: {e}")
        sys.exit(1)

    print("\n🎼 Initializing AgenticComposer...")
    composer = AgenticComposer(harmonyrouter=harmony_router, acceptance_threshold=0.75)
    composer.llm.use_mock = args.mock_llm

    print(f"\n🎤 USER PROMPT: '{args.prompt}'\n")
    if args.continue_midi:
        print(f"🎧 OUTPAINTING MODE: Continuing from seed '{args.continue_midi}'\n")

    try:
        llm_intent = composer.llm.generate_intent(args.prompt)
        blueprint = composer.planner.plan(llm_intent)
    except Exception as e:
        logger.error(f"❌ Failed during intent generation / planning: {e}")
        sys.exit(1)

    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    stem, _ = os.path.splitext(output_path)
    blueprint_filename = f"{stem}_blueprint.json"
    with open(blueprint_filename, "w") as f:
        json.dump({"llm_intent": llm_intent, "dense_blueprint": blueprint}, f, indent=2)
    print(f"📝 Song Blueprint saved to: {blueprint_filename}")

    try:
        final_song_midi = composer.compose_full_song(blueprint["timeline"], primer_midi_path=args.continue_midi)
    except Exception as e:
        logger.error(f"❌ Composition failed: {e}")
        sys.exit(1)

    final_song_midi.write(output_path)
    print(f"\n🏆 SUCCESS! Final masterpiece saved to: {output_path}")

    # --- 6. THE HYBRID NEURAL AUDIO RENDERER ---
    if args.skip_render:
        print("\n⏭️ Skipping neural audio render (--skip_render set). Done.")
        return

    print("\n🎧 INITIALIZING HYBRID NEURAL RENDERER...")
    wav_path = f"{stem}.wav"

    try:
        import subprocess
        import shutil
        from pydub import AudioSegment
        from pydub import effects
    except ImportError as e:
        print(f"⚠️ Neural renderer dependencies not installed ({e}). Skipping DDSP render.")
    else:
        try:
            ddsp_output_dir = "./ddsp_stems"
            if os.path.exists(ddsp_output_dir):
                shutil.rmtree(ddsp_output_dir)
            os.makedirs(ddsp_output_dir, exist_ok=True)

            print("   🎻 Preparing MIDI tracks for sequential DDSP GPU rendering...")
            env = os.environ.copy()
            env["TF_USE_LEGACY_KERAS"] = "1"
            env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Prevent TF from hogging entire GPU memory
            try:
                import nvidia.cudnn
                cudnn_file = getattr(nvidia.cudnn, "__file__", None)
                if cudnn_file is not None:
                    cudnn_lib_dir = os.path.join(os.path.dirname(cudnn_file), "lib")
                    if os.path.exists(cudnn_lib_dir):
                        env["LD_LIBRARY_PATH"] = cudnn_lib_dir + (f":{env['LD_LIBRARY_PATH']}" if "LD_LIBRARY_PATH" in env else "")
            except ImportError:
                pass

            import pretty_midi
            pm = pretty_midi.PrettyMIDI(output_path)
            supported_ddsp_programs = {40, 41, 42, 43, 73, 68, 71, 70, 56, 60, 57, 58}

            for inst in pm.instruments:
                if inst.is_drum or not inst.notes:
                    continue

                # Map instrument program if not supported by DDSP
                if inst.program not in supported_ddsp_programs:
                    prog = inst.program
                    if 32 <= prog <= 39:
                        inst.program = 43  # Double Bass
                    elif 48 <= prog <= 55:
                        inst.program = 40  # Violin
                    elif 56 <= prog <= 63:
                        inst.program = 56  # Trumpet
                    elif 64 <= prog <= 71:
                        inst.program = 71  # Clarinet
                    elif 72 <= prog <= 79:
                        inst.program = 73  # Flute
                    else:
                        inst.program = 40  # Default to Violin

            temp_midi_path = "./temp_mapped_input.mid"
            pm.write(temp_midi_path)

            print(f"   🎻 Running time-chunked DDSP GPU synthesis (chunk size: 20s)...")
            _chunk_and_render_ddsp(temp_midi_path, ddsp_output_dir, env, chunk_seconds=20.0, overlap_seconds=1.0)

            if os.path.exists(temp_midi_path):
                try:
                    os.remove(temp_midi_path)
                except Exception:
                    pass

            print("   🎛️ Mixing Neural Stems...")
            full_mix_path = os.path.join(ddsp_output_dir, "full_mix.wav")
            if os.path.exists(full_mix_path):
                shutil.copy(full_mix_path, wav_path)
                print(f"🏆 NEURAL MASTERPIECE RENDERED: {wav_path}")
            else:
                stem_files = sorted(f for f in os.listdir(ddsp_output_dir) if f.endswith(".wav"))
                if not stem_files:
                    print(f"⚠️ DDSP produced no .wav stems. Skipping mix.")
                else:
                    master_mix = AudioSegment.silent(duration=0)
                    for stem_file in stem_files:
                        stem_audio = AudioSegment.from_wav(os.path.join(ddsp_output_dir, stem_file))
                        master_mix = master_mix.overlay(stem_audio - 3)
                    master_mix = effects.normalize(master_mix)
                    master_mix.export(wav_path, format="wav")
                    print(f"🏆 NEURAL MASTERPIECE RENDERED: {wav_path}")

        except FileNotFoundError:
            print("⚠️ midi_ddsp_synthesize CLI not found in PATH.")
        except subprocess.TimeoutExpired:
            print("⚠️ DDSP synthesis timed out.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ DDSP rendering failed with exit code {e.returncode}.")
            print("--- DDSP Stdout ---")
            print(e.stdout)
            print("--- DDSP Stderr ---")
            print(e.stderr)
            print("-------------------")
        except Exception as e:
            print(f"⚠️ DDSP rendering failed ({e}).")


if __name__ == "__main__":
    main()