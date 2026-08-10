import os
import glob
import argparse
try:
    import pretty_midi
except ImportError:
    pretty_midi = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.physics_failsafe import WavePINNModel

def get_emotion_from_path(file_path: str) -> list[float]:
    path_lower = file_path.lower()
    if "q1" in path_lower:
        return [0.8, 0.1, 0.2, 0.1]  # Happy / Q1
    elif "q2" in path_lower:
        return [0.1, 0.8, 0.1, 0.8]  # Tense / Q2
    elif "q3" in path_lower:
        return [0.1, 0.2, 0.8, 0.1]  # Sad / Q3
    elif "q4" in path_lower:
        return [0.2, 0.1, 0.2, 0.8]  # Calm / Q4
    return [0.5, 0.2, 0.3, 0.3]      # Default fallback

def midi_to_tensors(file_path: str) -> list[torch.Tensor]:
    """
    Convert MIDI file -> list of tensors, one for each instrument/program.
    Each tensor contains note data: [frequency, amplitude, duration, happy, sad, calm, tense, instrument_program/127.0]
    """
    if pretty_midi is None:
        print(f"   ⚠️ Cannot process MIDI '{file_path}': pretty_midi is not installed.")
        return None
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
    except Exception as e:
        print(f"   ⚠️ Failed to parse MIDI '{file_path}': {e}")
        return None
        
    emotion = get_emotion_from_path(file_path)
    tensors = []

    # Get non-drum instruments
    instruments = [inst for inst in pm.instruments if not getattr(inst, "is_drum", False)]
    for inst in instruments:
        if not inst.notes:
            continue
            
        # Ensure notes are sorted chronologically
        notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
        notes_data = []
        for note in notes:
            freq = 440.0 * (2.0 ** ((note.pitch - 69.0) / 12.0))
            amp = note.velocity / 127.0
            dur = max(0.0, note.end - note.start)
            data = [freq, amp, dur] + emotion + [inst.program / 127.0]
            notes_data.append(data)
            
        if len(notes_data) >= 2:
            tensors.append(torch.tensor(notes_data, dtype=torch.float32))
            
    return tensors

def npy_to_tensors(file_path: str) -> list[torch.Tensor]:
    """
    Convert preprocessed .npy token file -> list of tensors, one for each instrument/program.
    """
    try:
        import numpy as np
        tokens = np.load(file_path)
    except Exception as e:
        print(f"   ⚠️ Failed to parse numpy file '{file_path}': {e}")
        return None

    if len(tokens.shape) != 2 or tokens.shape[1] < 6:
        print(f"   ⚠️ Invalid token shape {tokens.shape} in '{file_path}'")
        return None

    emotion = get_emotion_from_path(file_path)
    tensors = []

    # Filter out drums (program 128)
    non_drum = tokens[tokens[:, 4] != 128]
    unique_insts = np.unique(non_drum[:, 4])
    
    for inst_prog in unique_insts:
        inst_tokens = non_drum[non_drum[:, 4] == inst_prog]
        if len(inst_tokens) < 2:
            continue
            
        # Sort notes chronologically: onset (col 0) first, then pitch (octave * 12 + pitch_class)
        pitches = inst_tokens[:, 2] * 12 + inst_tokens[:, 3]
        sort_indices = np.lexsort((pitches, inst_tokens[:, 0]))
        sorted_tokens = inst_tokens[sort_indices]
        sorted_pitches = pitches[sort_indices]

        notes_data = []
        for idx, token in enumerate(sorted_tokens):
            pitch = int(sorted_pitches[idx])
            freq = 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))
            amp = token[5] / 127.0
            dur = max(0.0, token[1] / 100.0)
            data = [freq, amp, dur] + emotion + [inst_prog / 127.0]
            notes_data.append(data)
            
        if len(notes_data) >= 2:
            tensors.append(torch.tensor(notes_data, dtype=torch.float32))
            
    return tensors

def main():
    parser = argparse.ArgumentParser(description="Train the SCMoE Physics Failsafe Model")
    parser.add_argument("--dataset-dir", default="d:/scmoe/Moonbeam Multi-Task Data", help="Directory containing training MIDI files")
    parser.add_argument("--output-path", default="models/wave_pinn.pt", help="Path to save the trained model checkpoint")
    parser.add_argument("--instrument", type=int, default=None, help="MIDI program number to filter for training (default: train on all instruments)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # Search for all training files recursively
    print(f"🔍 Searching for training files in '{args.dataset_dir}'...")
    midi_files = glob.glob(os.path.join(args.dataset_dir, "**", "*.mid"), recursive=True)
    midi_files += glob.glob(os.path.join(args.dataset_dir, "**", "*.midi"), recursive=True)
    npy_files = glob.glob(os.path.join(args.dataset_dir, "**", "*.npy"), recursive=True)
    # Exclude chord metadata files
    npy_files = [f for f in npy_files if not f.endswith("_bar_beat_chord.npy")]
    all_files = midi_files + npy_files
    print(f"   ↳ Found {len(midi_files)} MIDI files and {len(npy_files)} numpy files (total: {len(all_files)} files).")

    dataset = []
    skipped = 0
    
    print("📦 Preprocessing files to training tensors...")
    for idx, f in enumerate(all_files):
        if idx > 0 and idx % 200 == 0:
            print(f"   ↳ Processed {idx}/{len(all_files)} files...")
        
        if f.lower().endswith(".npy"):
            tensors = npy_to_tensors(f)
        else:
            if pretty_midi is None:
                skipped += 1
                continue
            tensors = midi_to_tensors(f)
            
        if not tensors:
            skipped += 1
            continue
            
        # Optional instrument filter
        if args.instrument is not None:
            tensors = [t for t in tensors if abs(t[0, -1].item() * 127.0 - args.instrument) < 1e-4]
            if not tensors:
                skipped += 1
                continue

        for tensor in tensors:
            if tensor.shape[0] < 2:
                continue
            dataset.append(tensor)

    print(f"   ↳ Done. Loaded {len(dataset)} valid sequences, skipped {skipped}.")
    if len(dataset) == 0:
        raise RuntimeError("No training sequences could be extracted. Check your dataset directory and filters.")

    # Create dataset sequences (predicting next note)
    X_train = torch.cat([d[:-1, :] for d in dataset], dim=0).float()
    Y_train = torch.cat([d[1:, :3] for d in dataset], dim=0)
    Y_train = torch.cat([Y_train, X_train[:, 3:]], dim=1).float()

    print(f"\n📊 Training Data Shapes:\n   X_train: {X_train.shape}\n   Y_train: {Y_train.shape}")

    # Initialize model
    model = WavePINNModel(input_dim=8, wave_dim=32, latent_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    print(f"\n🚀 Starting training on {device} (Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr})...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            num_batches += 1
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(1, num_batches)
            print(f"   ↳ Epoch {epoch+1:02d}/{args.epochs} | Mean MSE Loss: {avg_loss:.5e}")

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Save trained checkpoint
    torch.save(model.state_dict(), args.output_path)
    print(f"\n💾 Model weights successfully saved to: {args.output_path}")
    print("🎉 Training Complete!")

if __name__ == "__main__":
    main()
