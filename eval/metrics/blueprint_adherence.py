import os
from typing import List, Dict, Any, Tuple
try:
    import pretty_midi
except ImportError:
    pretty_midi = None

# Try to import the existing HardScorer from the studio engine
try:
    from critic.hard_scorer import HardScorer
except ImportError:
    HardScorer = None

def chord_match_rate(midi_path: str, blueprint: dict) -> float:
    """Estimates the chord adherence rate using the HardScorer logic."""
    if pretty_midi is None or not os.path.exists(midi_path):
        return 0.0
    if HardScorer is None:
        return 0.5  # Fallback
        
    try:
        midi_obj = pretty_midi.PrettyMIDI(midi_path)
        scorer = HardScorer()
        
        bpm = blueprint.get("bpm", 120)
        bars = blueprint.get("bars", 8)
        beats_per_bar = blueprint.get("beats_per_bar", 4) or 4
        beat_duration = 60.0 / bpm
        section_end_time = bars * beats_per_bar * beat_duration
        chord_timeline = blueprint.get("chord_timeline", [])
        target_chords = blueprint.get("target_chords", [])
        
        if not target_chords and not chord_timeline:
            return 1.0
            
        score, fb = scorer._score_chord_and_bass(
            midi_obj, target_chords, chord_timeline, section_end_time, beat_duration, beats_per_bar
        )
        return float(score)
    except Exception:
        return 0.0

def voice_leading_score(midi_path: str) -> float:
    """Computes the voice leading score using the HardScorer logic."""
    if pretty_midi is None or not os.path.exists(midi_path):
        return 0.0
    if HardScorer is None:
        return 0.7  # Fallback
        
    try:
        midi_obj = pretty_midi.PrettyMIDI(midi_path)
        scorer = HardScorer()
        score, fb = scorer._score_voice_leading(midi_obj)
        return float(score)
    except Exception:
        return 0.0

def rule_based_emotion_classifier(midi_path: str) -> str:
    """
    A rule-based heuristic classifier mapping MIDI properties to the 4 EMOPIA quadrants:
    Q1: High valence, High arousal (Happy, energetic)
    Q2: Low valence, High arousal (Tense, angry)
    Q3: Low valence, Low arousal (Sad, melancholy)
    Q4: High valence, Low arousal (Calm, peaceful)
    """
    if pretty_midi is None or not os.path.exists(midi_path):
        return "emo_q4"
        
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for inst in pm.instruments:
            if not inst.is_drum:
                notes.extend(inst.notes)
                
        if not notes:
            return "emo_q4"
            
        avg_velocity = np.mean([n.velocity for n in notes])
        note_density = len(notes) / max(0.1, pm.get_end_time())
        
        # Heuristic valence/arousal scoring
        # Arousal is associated with tempo and velocity/density
        arousal = (avg_velocity / 127.0) * 0.5 + min(1.0, note_density / 5.0) * 0.5
        
        # Valence heuristic is major key vs minor key (major -> positive valence)
        # Simple major/minor key heuristic from note distributions
        pcs = [n.pitch % 12 for n in notes]
        major_matches = sum(1 for pc in pcs if pc in [0, 2, 4, 5, 7, 9, 11])  # C major scale
        minor_matches = sum(1 for pc in pcs if pc in [0, 2, 3, 5, 7, 8, 10])  # C minor scale
        valence = 0.7 if major_matches >= minor_matches else 0.3
        
        if valence >= 0.5 and arousal >= 0.5:
            return "emo_q1"
        elif valence < 0.5 and arousal >= 0.5:
            return "emo_q2"
        elif valence < 0.5 and arousal < 0.5:
            return "emo_q3"
        else:
            return "emo_q4"
    except Exception:
        return "emo_q4"

def emotion_adherence(midi_path: str, prompted_label: str, emopia_classifier=None) -> float:
    """
    Returns 1.0 if the predicted emotion matches the prompted emotion, else 0.0.
    Falls back to a robust rule-based heuristic classifier if no neural head is loaded.
    """
    # Normalize labels
    prompted = prompted_label.lower().replace("mood_", "").replace("_lora", "").strip()
    
    # Map synonyms to quadrants (emo_q1 to emo_q4)
    quadrant_map = {
        "happy": "emo_q1", "heroic": "emo_q1", "excited": "emo_q1", "q1": "emo_q1", "emo_q1": "emo_q1",
        "tension": "emo_q2", "angry": "emo_q2", "dark": "emo_q2", "q2": "emo_q2", "emo_q2": "emo_q2",
        "sad": "emo_q3", "melancholy": "emo_q3", "grief": "emo_q3", "q3": "emo_q3", "emo_q3": "emo_q3",
        "calm": "emo_q4", "peaceful": "emo_q4", "tender": "emo_q4", "q4": "emo_q4", "emo_q4": "emo_q4"
    }
    
    target_quadrant = quadrant_map.get(prompted, "emo_q4")
    
    if emopia_classifier is not None:
        try:
            pred_label = emopia_classifier.predict(midi_path)
            pred_quadrant = quadrant_map.get(pred_label.lower(), "emo_q4")
            return float(pred_quadrant == target_quadrant)
        except Exception:
            pass
            
    # Heuristic fallback
    pred_quadrant = rule_based_emotion_classifier(midi_path)
    return float(pred_quadrant == target_quadrant)

def orchestration_match(midi_path: str, blueprint_instruments: List[str]) -> Dict[str, float]:
    """
    Computes instrument precision and recall.
    """
    if pretty_midi is None or not os.path.exists(midi_path):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    try:
        from shared.gm_instruments import get_program
    except ImportError:
        def get_program(name):
            # Fallback mappings for standard names
            mapping = {"piano": 0, "strings": 48, "bass": 33, "brass": 61, "flute": 73}
            return mapping.get(name.lower(), 0)

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        used_programs = {inst.program for inst in pm.instruments if not inst.is_drum}
        planned_programs = {get_program(name) for name in blueprint_instruments}
        
        if not used_programs and not planned_programs:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not used_programs or not planned_programs:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        intersection = used_programs & planned_programs
        precision = len(intersection) / len(used_programs)
        recall = len(intersection) / len(planned_programs)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    except Exception:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def density_adherence_metrics(midi_path: str, blueprint_rhythm_template: List[float], bpm: float, bars: int, beats_per_bar: int = 4, time_offset: float = 0.0) -> Dict[str, float]:
    """
    Computes MSE, RMSE, MAE, and R^2 between the blueprint's target density/rhythm template (y_true)
    and the actual generated MIDI's note density per bar (y_pred).
    """
    default_vals = {"mse": 1.0, "rmse": 1.0, "mae": 1.0, "r2": 0.0}
    if pretty_midi is None or not os.path.exists(midi_path) or not blueprint_rhythm_template:
        return default_vals
        
    try:
        from eval.metrics.regression import calculate_mse, calculate_rmse, calculate_mae, calculate_r2
        import numpy as np
        
        pm = pretty_midi.PrettyMIDI(midi_path)
        beat_duration = 60.0 / bpm
        bar_duration = beats_per_bar * beat_duration
        
        all_notes = []
        for inst in pm.instruments:
            if not inst.is_drum:
                all_notes.extend(inst.notes)
                
        if not all_notes:
            return default_vals
            
        # Segment actual notes into bars
        actual_densities = []
        n_steps = len(blueprint_rhythm_template)
        # Calculate step duration (matching each element of the rhythm template)
        step_duration = (bars * bar_duration) / max(1, n_steps)
        
        for i in range(n_steps):
            start = time_offset + i * step_duration
            end = time_offset + (i + 1) * step_duration
            notes_in_step = sum(1 for n in all_notes if start <= n.start < end)
            
            # Normalize actual density: notes per beat, capped at 4.0 notes/beat (16th notes density)
            notes_per_beat = notes_in_step / max(0.1, step_duration / beat_duration)
            normalized_density = min(1.0, notes_per_beat / 4.0)
            actual_densities.append(normalized_density)
            
        mse = calculate_mse(blueprint_rhythm_template, actual_densities)
        rmse = calculate_rmse(blueprint_rhythm_template, actual_densities)
        mae = calculate_mae(blueprint_rhythm_template, actual_densities)
        r2 = calculate_r2(blueprint_rhythm_template, actual_densities)
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
    except Exception:
        return default_vals

