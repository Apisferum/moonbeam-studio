import os
import glob
import argparse
import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.physics_failsafe import WavePINNModel

# Default GM Acoustic Guitar program
GUITAR_PROGRAM = 25 

def midi_to_tensor(file_path: str, instrument_program: int = GUITAR_PROGRAM, emotion=None):
    """
    Converts a MIDI file into a tensor of shape (num_notes, 8) representing:
    [frequency, amplitude, duration, happy, sad, calm, tense, instrument]
    """
    if emotion is None:
        emotion = [0.8, 0.1, 0.2, 0.1] # Default emotion vector
        
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
    except Exception as e:
        # Skip corrupted MIDI files
        return None
        
    notes_data = []

    # Get non-drum tracks
    non_drum = [inst for inst in pm.instruments if not getattr(inst, "is_drum", False)]
    if instrument_program is None:
        candidates = non_drum
    else:
        candidates = [inst for inst in non_drum if inst.program == instrument_program]
        if len(candidates) == 0:
            candidates = non_drum

    if len(candidates) == 0:
        return None

    # Choose track with the most notes
    inst = max(candidates, key=lambda i: len(i.notes) if i.notes is not None else 0)
    if not inst.notes:
        return None

    # Sort notes by start time
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
    parser = argparse.ArgumentParser(description="Train SCMoE WavePINN Failsafe Model")
    parser.add_argument("--dataset-dir", default="../Moonbeam Multi-Task Data", help="Path to MIDI dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-path", default="models/wave_pinn.pt", help="Path to save the trained model weights")
    args = parser.parse_args()

    print(f"🔍 Searching for MIDI files in {args.dataset_dir}...")
    midi_files = glob.glob(os.path.join(args.dataset_dir, "**", "*.mid"), recursive=True)
    print(f"Found {len(midi_files)} MIDI files.")

    print("📦 Processing MIDI files into tensors...")
    dataset = []
    skipped = 0
    for idx, f in enumerate(midi_files):
        tensor = midi_to_tensor(f)
        if tensor is None or tensor.shape[0] < 2:
            skipped += 1
            continue
        dataset.append(tensor)
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(midi_files)} files...")

    if len(dataset) == 0:
        print("❌ Error: No training sequences extracted. Check your dataset folder or program filter.")
        return

    print(f"✅ Extracted {len(dataset)} sequences (skipped {skipped} empty/invalid files).")

    # X: current note features, Y: next note features (predicting [freq, amp, dur] and keeping emotion/inst)
    X_train = torch.cat([d[:-1, :] for d in dataset], dim=0).float()
    Y_train = torch.cat([d[1:, :3] for d in dataset], dim=0)
    Y_train = torch.cat([Y_train, X_train[:, 3:]], dim=1).float()

    print(f"Training Data Shapes: X={X_train.shape}, Y={Y_train.shape}")

    # Model and training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    model = WavePINNModel(input_dim=8, wave_dim=32, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )

    print("🚀 Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            
            # MSE loss on note features
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += float(loss.item())
            num_batches += 1
            
        avg_loss = epoch_loss / max(1, num_batches)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{args.epochs} | Loss: {avg_loss:.6f}")

    # Create models output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"💾 Saving model weights to {args.output_path}...")
    torch.save(model.state_dict(), args.output_path)
    print("🎉 Training complete!")

if __name__ == "__main__":
    main()
