import os
import numpy as np
try:
    import pretty_midi
except ImportError:
    pretty_midi = None

def get_chroma_vector(notes: list) -> np.ndarray:
    """Computes a 12-dimensional normalized chroma vector from a list of notes."""
    chroma = np.zeros(12)
    for n in notes:
        chroma[n.pitch % 12] += 1.0
    sum_val = np.sum(chroma)
    return chroma / sum_val if sum_val > 0 else chroma

def boundary_metrics(midi_path: str, boundary_time: float, window: float = 4.0) -> dict:
    """
    Computes boundary transition statistics for a given MIDI file at boundary_time.
    Slices the MIDI into [boundary_time - window, boundary_time] and [boundary_time, boundary_time + window].
    """
    default_res = {
        "harmonic_jump": 0.0,
        "onset_density_delta": 0.0,
        "silence_gap_sec": 0.0
    }
    
    if pretty_midi is None or not os.path.exists(midi_path):
        return default_res
        
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        all_notes = []
        for inst in pm.instruments:
            if not inst.is_drum:
                all_notes.extend(inst.notes)
                
        if not all_notes:
            return default_res
            
        # Slices
        t_start_before = max(0.0, boundary_time - window)
        notes_before = [n for n in all_notes if t_start_before <= n.start < boundary_time]
        notes_after = [n for n in all_notes if boundary_time <= n.start <= boundary_time + window]
        
        # 1. Onset Density Delta
        density_before = len(notes_before) / window if window > 0 else 0
        density_after = len(notes_after) / window if window > 0 else 0
        density_delta = density_after - density_before
        
        # 2. Harmonic Jump (measured via cosine distance between the chroma vectors of the boundary boundaries)
        # We take notes in the immediate vicinity (last 2 seconds of before, first 2 seconds of after)
        border_before = [n for n in notes_before if n.start >= boundary_time - 2.0]
        border_after = [n for n in notes_after if n.start <= boundary_time + 2.0]
        
        chroma_before = get_chroma_vector(border_before)
        chroma_after = get_chroma_vector(border_after)
        
        norm_b = np.linalg.norm(chroma_before)
        norm_a = np.linalg.norm(chroma_after)
        
        if norm_b > 0 and norm_a > 0:
            cosine_sim = np.dot(chroma_before, chroma_after) / (norm_b * norm_a)
            harmonic_jump = float(1.0 - cosine_sim)
        else:
            harmonic_jump = 0.0
            
        # 3. Silence Gap
        # Find the last note offset in notes_before and first note onset in notes_after
        last_offset = max([n.end for n in notes_before]) if notes_before else boundary_time - window
        first_onset = min([n.start for n in notes_after]) if notes_after else boundary_time + window
        
        silence_gap = max(0.0, first_onset - last_offset) if (notes_before and notes_after) else 0.0
        
        return {
            "harmonic_jump": float(harmonic_jump),
            "onset_density_delta": float(density_delta),
            "silence_gap_sec": float(silence_gap)
        }
        
    except Exception:
        return default_res
