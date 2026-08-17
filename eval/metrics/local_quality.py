import math
from collections import Counter
import numpy as np

# Try to import external libraries, with fallback definitions
try:
    import muspy
except ImportError:
    muspy = None

try:
    import pretty_midi
except ImportError:
    pretty_midi = None

try:
    from music21 import converter, analysis
except ImportError:
    converter = None

def get_notes_from_midi(midi_path):
    """Utility to parse notes using pretty_midi with fallback."""
    if pretty_midi is None:
        return []
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for inst in pm.instruments:
            if not inst.is_drum:
                notes.extend(inst.notes)
        notes.sort(key=lambda x: x.start)
        return notes
    except Exception:
        return []

def pitch_entropy(midi_path):
    """Calculates the Shannon entropy of note pitches."""
    if muspy is not None:
        try:
            m = muspy.read_midi(midi_path)
            return float(muspy.pitch_entropy(m))
        except Exception:
            pass
            
    notes = get_notes_from_midi(midi_path)
    if not notes:
        return 0.0
    pitches = [n.pitch for n in notes]
    total = len(pitches)
    counts = Counter(pitches)
    entropy = 0.0
    for pitch, count in counts.items():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)

def rhythm_entropy(midi_path):
    """Calculates groove consistency or duration-histogram entropy."""
    if muspy is not None:
        try:
            m = muspy.read_midi(midi_path)
            # muspy's groove consistency returns NaN if rhythm is perfectly uniform or empty
            val = muspy.groove_consistency(m)
            if not math.isnan(val):
                return float(val)
        except Exception:
            pass
            
    notes = get_notes_from_midi(midi_path)
    if len(notes) < 2:
        return 0.0
    
    # Calculate Inter-Onset Intervals (IOIs)
    iois = []
    for i in range(len(notes) - 1):
        ioi = round(notes[i+1].start - notes[i].start, 2)
        if ioi > 0:
            iois.append(ioi)
            
    if not iois:
        return 0.0
        
    total = len(iois)
    counts = Counter(iois)
    entropy = 0.0
    for ioi, count in counts.items():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)

def harmony_consistency(midi_path):
    """Computes the fraction of notes aligning with the analyzed diatonic key."""
    if converter is not None:
        try:
            score = converter.parse(midi_path)
            key = score.analyze('key')
            # Get list of pitch classes in key
            pcs = [p.pitchClass for p in key.pitches]
            notes = [n.pitch.pitchClass for n in score.flatten().notes if n.isNote]
            if not notes:
                return 1.0
            matches = sum(1 for pc in notes if pc in pcs)
            return float(matches / len(notes))
        except Exception:
            pass
            
    # Simple Krumhansl-Schmuckler fallback or major scale heuristic key detection
    notes = get_notes_from_midi(midi_path)
    if not notes:
        return 1.0
        
    pitch_classes = [n.pitch % 12 for n in notes]
    # Simple major scale templates (diatonic scale is 7 notes)
    # Major scale intervals relative to tonic: 0, 2, 4, 5, 7, 9, 11
    major_scales = {}
    for root in range(12):
        major_scales[root] = {(root + interval) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]}
        
    # Find the major scale that contains the maximum number of pitch classes
    best_root = 0
    best_match_count = -1
    for root, scale in major_scales.items():
        matches = sum(1 for pc in pitch_classes if pc in scale)
        if matches > best_match_count:
            best_match_count = matches
            best_root = root
            
    # Return ratio of matching notes to total notes
    return float(best_match_count / len(pitch_classes))
