import os
import glob
import argparse
import pretty_midi
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

def midi_to_tensor(file_path: str, instrument: int = 25) -> torch.Tensor:
    """
    Convert MIDI file -> tensor of [frequency, amplitude, duration, happy, sad, calm, tense, instrument]
    """
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
    except Exception as e:
        print(f"   ⚠️ Failed to parse MIDI '{file_path}': {e}")
        return None
        
    notes_data = []
    emotion = get_emotion_from_path(file_path)

    # Filter tracks: prefer guitar or requested instrument
    non_drum = [inst for inst in pm.instruments if not getattr(inst, "is_drum", False)]
    if instrument is None:
        candidates = non_drum
    else:
        candidates = [inst for inst in non_drum if inst.program == instrument]
        if len(candidates) == 0:
            candidates = non_drum

    if len(candidates) == 0:
        return None

    # Pick the track with the most notes
    inst = max(candidates, key=lambda i: len(i.notes) if i.notes is not None else 0)
    if not inst.notes:
        return None

    # Ensure notes are sorted chronologically
    notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
    for note in notes:
        freq = pretty_midi.note_number_to_hz(note.pitch)
        amp = note.velocity / 127.0
        dur = max(0.0, note.end - note.start)
        data = [freq, amp, dur] + emotion + [1.0]
        notes_data.append(data)

    if len(notes_data) == 0:
        return None
    return torch.tensor(notes_data, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Train the SCMoE Physics Failsafe Model")
    parser.add_argument("--dataset-dir", default="d:/scmoe/Moonbeam Multi-Task Data", help="Directory containing training MIDI files")
    parser.add_argument("--output-path", default="models/wave_pinn.pt", help="Path to save the trained model checkpoint")
    parser.add_argument("--instrument", type=int, default=25, help="MIDI program number to filter for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # Search for all midi files recursively
    print(f"🔍 Searching for MIDI files in '{args.dataset_dir}'...")
    midi_files = glob.glob(os.path.join(args.dataset_dir, "**", "*.mid"), recursive=True)
    print(f"   ↳ Found {len(midi_files)} MIDI files.")

    dataset = []
    skipped = 0
    
    print("📦 Preprocessing MIDI files to training tensors...")
    for idx, f in enumerate(midi_files):
        if idx > 0 and idx % 200 == 0:
            print(f"   ↳ Processed {idx}/{len(midi_files)} files...")
        tensor = midi_to_tensor(f, instrument=args.instrument)
        if tensor is None or tensor.shape[0] < 2:
            skipped += 1
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
