import os
import numpy as np
try:
    import pretty_midi
except ImportError:
    pretty_midi = None

def extract_embedding(midi_path):
    """
    Extracts the normalized 28D semantic embedding (12D Chroma + 12D Intervals + 4D Stats)
    from a MIDI file, mirroring the math in MotifMemoryFAISS.
    """
    if pretty_midi is None or not os.path.exists(midi_path):
        return np.zeros(28, dtype='float32')

    try:
        midi_obj = pretty_midi.PrettyMIDI(midi_path)
        chroma = np.zeros(12)
        intervals = np.zeros(12)
        total_notes = 0
        pitch_sum, vel_sum, dur_sum = 0.0, 0.0, 0.0

        all_notes = [note for inst in midi_obj.instruments if not inst.is_drum for note in inst.notes]
        all_notes.sort(key=lambda x: x.start)

        prev_pitch = -1
        for note in all_notes:
            chroma[note.pitch % 12] += 1
            pitch_sum += note.pitch
            vel_sum += note.velocity
            dur_sum += (note.end - note.start)

            if prev_pitch != -1:
                interval = (note.pitch - prev_pitch) % 12
                intervals[interval] += 1
            prev_pitch = note.pitch
            total_notes += 1

        if total_notes > 0:
            chroma = chroma / np.sum(chroma)
            if np.sum(intervals) > 0:
                intervals = intervals / np.sum(intervals)
            stats = [
                total_notes / 100.0,
                (pitch_sum / total_notes) / 127.0,
                (vel_sum / total_notes) / 127.0,
                dur_sum / total_notes
            ]
        else:
            stats = [0.0, 0.0, 0.0, 0.0]

        emb = np.concatenate([chroma, intervals, stats]).astype('float32')
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    except Exception:
        return np.zeros(28, dtype='float32')

def motif_recurrence_score(piece_trace: dict) -> float:
    """
    Measures the similarity of repeating sections to their retrieved FAISS motif templates.
    """
    hits = 0
    total = 0
    sections = piece_trace.get("sections", [])
    
    for section in sections:
        section_name = section.get("section_type", "").lower()
        # Check if this section is a repeat (e.g. Chorus 2, Verse 2, or contains numbers)
        is_repeat = any(char.isdigit() for char in section_name) or "coda" in section_name
        
        if is_repeat and section.get("retrieved_motifs"):
            # Compare current generated section MIDI to the retrieved template MIDI
            gen_midi = section.get("final_midi_path")
            
            # Find a saved midi path for the retrieved motif in previously generated files
            retrieved_name = section["retrieved_motifs"][0].get("name", "").lower()
            
            # Search previous sections to find the MIDI path of the matching motif
            motif_midi = None
            for s in sections:
                if s.get("section_type", "").lower() == retrieved_name:
                    motif_midi = s.get("final_midi_path")
                    break
            
            if gen_midi and motif_midi and os.path.exists(gen_midi) and os.path.exists(motif_midi):
                total += 1
                target_emb = extract_embedding(motif_midi)
                gen_emb = extract_embedding(gen_midi)
                
                # Cosine similarity
                sim = np.dot(target_emb, gen_emb)
                if sim > 0.7:
                    hits += 1
                    
    return hits / max(1, total) if total > 0 else 1.0

def self_similarity_matrix(midi_paths: list[str]) -> np.ndarray:
    """
    Computes a section-level self-similarity matrix over generated section MIDIs.
    """
    n = len(midi_paths)
    matrix = np.zeros((n, n))
    embeddings = [extract_embedding(path) for path in midi_paths]
    
    for i in range(n):
        for j in range(n):
            if np.linalg.norm(embeddings[i]) > 0 and np.linalg.norm(embeddings[j]) > 0:
                matrix[i, j] = np.dot(embeddings[i], embeddings[j])
            else:
                matrix[i, j] = 0.0
    return matrix
