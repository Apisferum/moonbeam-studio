import argparse
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import time

studio_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(studio_root)
sys.path.append(os.path.abspath(os.path.join(studio_root, 'moonbeam-studio')))

codebase_root = os.environ.get("MOONBEAM_CODEBASE_PATH")
if not codebase_root:
    codebase_root = os.path.abspath(os.path.join(studio_root, "..", "moonbeam-codebase"))

if os.path.isdir(codebase_root):
    sys.path.insert(0, codebase_root)
    sys.path.insert(0, os.path.join(codebase_root, "src", "llama_recipes"))

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
    for i in range(8):
        note = pretty_midi.Note(
            velocity=65,
            pitch=60 + (i % 5) * 2,  # Melodic steps
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(output_path)

def main():
    parser = argparse.ArgumentParser(description="Hi-ACG Reimplementation Runner")
    parser.add_argument("--prompt_json", type=str, required=True, help="Path to prompt/blueprint JSON file")
    parser.add_argument("--output_midi", type=str, required=True, help="Path to output MIDI file")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--model_config_path", type=str, default=None)
    args = parser.parse_args()

    print(f"🎵 Running Hi-ACG reimplemented baseline on prompt: {args.prompt_json}")

    if args.mock:
        print("💡 [Hi-ACG] Running in mock simulation mode...")
        time.sleep(0.1)
        generate_mock_midi(args.output_midi)
        print(f"🏆 Saved mock baseline MIDI to {args.output_midi}")
        return

    # Real baseline execution
    from recipes.inference.custom_music_generation.generation import MusicLlama
    import torch

    with open(args.prompt_json, 'r') as f:
        blueprint = json.load(f)

    # Instantiate base model
    print("--> Loading Moonbeam model for Hi-ACG...")
    generator = MusicLlama.build(
        ckpt_dir=args.base_model_path,
        model_config_path=args.model_config_path,
        tokenizer_path="tokenizer.model",
        max_seq_len=1024,
        max_batch_size=1
    )

    timeline = blueprint.get("dense_blueprint", {}).get("timeline", [])
    if not timeline:
        print("❌ Error: empty timeline in blueprint.")
        sys.exit(1)

    # Hi-ACG generates sequentially, section-by-section, using the IMMEDIATE previous section
    # as the auto-conditioned primer (no semantic FAISS memory, no LoRA merging, no refinement).
    last_section_tokens = None
    all_midi_objects = []

    print("--> Generating sections sequentially (ACG paradigm)...")
    for idx, s in enumerate(timeline):
        print(f"Generating Section {idx+1}/{len(timeline)}: {s.get('section_name')}")
        
        # Build prompt: either SOS or the last section's generated tokens
        if last_section_tokens is None:
            prompt_tokens = [[generator.tokenizer.sos_token_compound]]
        else:
            prompt_tokens = [[generator.tokenizer.sos_token_compound] + last_section_tokens[-128:]] # Cap primer length
            
        generation_tokens, _ = generator.generate(
            prompt_tokens=prompt_tokens,
            bpm_condition=[s.get("bpm", 120)],
            time_signature_condition=["4/4"],
            num_measures_condition=[s.get("bars", 8)],
            max_gen_len=256,
            temperature=0.8,
            top_p=0.9
        )
        
        # Save generated tokens for next section's conditioning
        flat_tokens = [tok for group in generation_tokens[0] for tok in (group if isinstance(group[0], list) else [group])]
        last_section_tokens = flat_tokens
        
        # Decode current section to MIDI
        sect_midi = generator.tokenizer.compound_to_midi(flat_tokens)
        all_midi_objects.append(sect_midi)

    # Stitch all sections together
    import io
    final_pm = pretty_midi.PrettyMIDI()
    current_time_offset = 0.0
    for sect_midi_mido in all_midi_objects:
        buffer = io.BytesIO()
        sect_midi_mido.save(file=buffer)
        buffer.seek(0)
        sect_pm = pretty_midi.PrettyMIDI(buffer)
        
        for inst in sect_pm.instruments:
            # Simple stitching: find or create matching instrument channel in final
            new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
            for note in inst.notes:
                new_inst.notes.append(pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start + current_time_offset,
                    end=note.end + current_time_offset
                ))
            final_pm.instruments.append(new_inst)
            
        current_time_offset += sect_pm.get_end_time() + 0.2

    final_pm.write(args.output_midi)
    print(f"🏆 Hi-ACG MIDI successfully saved to {args.output_midi}")

if __name__ == "__main__":
    main()
