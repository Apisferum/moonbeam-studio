import argparse
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
            velocity=70,
            pitch=60 + (i % 2) * 4,  # Melodic steps
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(output_path)

def main():
    parser = argparse.ArgumentParser(description="Museformer Baseline Runner")
    parser.add_argument("--prompt_json", type=str, required=True, help="Path to prompt/blueprint JSON file")
    parser.add_argument("--output_midi", type=str, required=True, help="Path to output MIDI file")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    args = parser.parse_args()

    print(f"🎵 Running Museformer baseline on prompt: {args.prompt_json}")

    if args.mock:
        print("💡 [Museformer] Running in mock simulation mode...")
        time.sleep(0.1)
        generate_mock_midi(args.output_midi)
        print(f"🏆 Saved mock baseline MIDI to {args.output_midi}")
        return

    # Live execution placeholder (would clone microsoft/muzic)
    print("💡 [Museformer] Booting Museformer model...")
    try:
        raise NotImplementedError("Museformer model environment not initialized. Run with --mock for validation.")
    except Exception as e:
        print(f"❌ Error during Museformer execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
