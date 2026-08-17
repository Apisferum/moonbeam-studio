import argparse
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import time

# Add root folder to sys.path so we can import from project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'moonbeam-studio')))

try:
    import pretty_midi
except ImportError:
    pretty_midi = None

def generate_mock_midi(output_path: str):
    """Generates a dummy MIDI file with simple notes for metric testing."""
    if pretty_midi is None:
        return
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # Grand Piano
    # Create simple melodic notes
    for i in range(8):
        note = pretty_midi.Note(
            velocity=80,
            pitch=60 + (i % 4) * 2,  # Melodic steps
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(output_path)

def main():
    parser = argparse.ArgumentParser(description="Vanilla Moonbeam Baseline Runner")
    parser.add_argument("--prompt_json", type=str, required=True, help="Path to prompt/blueprint JSON file")
    parser.add_argument("--output_midi", type=str, required=True, help="Path to output MIDI file")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--model_config_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    args = parser.parse_args()

    print(f"🎵 Running Vanilla Moonbeam baseline on prompt: {args.prompt_json}")

    if args.mock:
        print("💡 [Vanilla Moonbeam] Running in mock simulation mode...")
        time.sleep(0.1)
        generate_mock_midi(args.output_midi)
        print(f"🏆 Saved mock baseline MIDI to {args.output_midi}")
        return

    # Real baseline execution
    from recipes.inference.custom_music_generation.generation import MusicLlama
    from llama_recipes.datasets.music_tokenizer import MusicTokenizer
    import torch

    with open(args.prompt_json, 'r') as f:
        blueprint = json.load(f)

    # Instantiate base model (unconditional or default commu lora)
    # Vanilla Moonbeam runs base model directly, no active PEFT adapter merging or soft refiner
    print("--> Loading Vanilla Moonbeam Model...")
    generator = MusicLlama.build(
        ckpt_dir=args.base_model_path,
        model_config_path=args.model_config_path,
        tokenizer_path=args.tokenizer_path or "tokenizer.model",
        max_seq_len=1024,
        max_batch_size=1
    )

    # Translate blueprint timeline into a simple continuation prompt
    # Since there's no planner, we just run a simple prompt completion loop
    timeline = blueprint.get("dense_blueprint", {}).get("timeline", [])
    if not timeline:
        print("❌ Error: empty timeline in blueprint.")
        sys.exit(1)

    # For vanilla moonbeam, we generate a single continuous MIDI sequence
    # without section-based partitioning or FAISS motif memory primer injecting.
    total_bars = sum(s.get("bars", 8) for s in timeline)
    bpm = timeline[0].get("bpm", 120)
    
    # Simple unconditional starter prompt (SOS token)
    prompt_tokens = [[generator.tokenizer.sos_token_compound]]
    
    print("--> Commencing generation...")
    generation_tokens, _ = generator.generate(
        prompt_tokens=prompt_tokens,
        bpm_condition=[bpm],
        time_signature_condition=["4/4"],
        num_measures_condition=[total_bars],
        max_gen_len=512,
        temperature=0.8,
        top_p=0.9
    )

    # Convert tokens to MIDI
    print("--> Saving generated MIDI...")
    flat_tokens = [tok for group in generation_tokens[0] for tok in (group if isinstance(group[0], list) else [group])]
    raw_midi = generator.tokenizer.compound_to_midi(flat_tokens)
    
    buffer = io.BytesIO()
    raw_midi.save(file=buffer)
    buffer.seek(0)
    midi_obj = pretty_midi.PrettyMIDI(buffer)
    midi_obj.write(args.output_midi)
    
    print(f"🏆 Baseline MIDI successfully saved to {args.output_midi}")

if __name__ == "__main__":
    main()
