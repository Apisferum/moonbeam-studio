import bisect
import random
import pretty_midi
from typing import Optional
from shared.music_theory_constants import logger, NOTE_MAP, CHORD_INTERVALS, ADAPTER_PROFILES

class MIDIPolisher:
    def _determine_active_profile(self, ties_weights: Optional[dict]) -> dict:
        weights = ties_weights or {"commu_lora": 0.6, "emopia_lora": 0.2, "slakh_lora": 0.2}
        dominant = max(weights, key=weights.get)
        name = dominant.replace("_lora", "")
        if weights[dominant] >= 0.6:
            return ADAPTER_PROFILES.get(name, ADAPTER_PROFILES["hybrid"])
        return ADAPTER_PROFILES["hybrid"]

    def _get_chord_pcs(self, chord_symbol: str) -> list:
        if not chord_symbol or chord_symbol in ("N", "s"): return list(range(12))
        
        if len(chord_symbol) >= 2 and chord_symbol[1] in ('#', 'b'):
            root_str, quality = chord_symbol[:2], chord_symbol[2:]
        else:
            root_str, quality = chord_symbol[0], chord_symbol[1:]
            
        quality = quality.replace("minor", "min").replace("major", "maj")
        if quality in ("", "M"): quality = "maj"
        if quality == "m": quality = "min"
            
        intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["maj"])
        root_pc = NOTE_MAP.get(root_str, 0)
        return [(root_pc + interval) % 12 for interval in intervals]

    def _build_chord_index(self, chord_timeline: list):
        """O(1) setup for O(log n) lookups using bisect."""
        self._chord_starts = [e["start"] for e in chord_timeline]
        self._chord_timeline = chord_timeline

    def _get_active_chord(self, current_beat: float) -> str:
        if not hasattr(self, '_chord_starts'): return "N"
        idx = bisect.bisect_right(self._chord_starts, current_beat) - 1
        if idx < 0: return "N"
        
        event = self._chord_timeline[idx]
        if current_beat < event["end"]:
            return event["chord"]
        return "N"

    def polish(self, midi_obj: pretty_midi.PrettyMIDI, bpm: float, 
               chord_timeline: list, ties_weights: dict = None) -> pretty_midi.PrettyMIDI:
        
        # TIES-AWARE: Dynamically adjust strengths based on the active adapter!
        profile = self._determine_active_profile(ties_weights)
        quant_strength = profile.get("quant_strength", 0.65)
        pitch_strength = profile.get("pitch_strength", 0.80)

        self._build_chord_index(chord_timeline)
        
        beat_duration = 60.0 / bpm
        sixteenth_grid = beat_duration / 4.0
        
        for inst in midi_obj.instruments:
            if inst.is_drum: continue
            
            for note in inst.notes:
                original_duration = note.end - note.start
                if original_duration <= 0: continue
                
                # 1. SOFT QUANTIZATION
                grid_time = round(note.start / sixteenth_grid) * sixteenth_grid
                note.start = note.start + quant_strength * (grid_time - note.start)
                note.end = note.start + original_duration
                
                # 2. TIMELINE-AWARE CHORD CORRECTION
                current_beat = note.start / beat_duration
                current_chord = self._get_active_chord(current_beat)
                
                valid_pcs = self._get_chord_pcs(current_chord)
                current_pc = note.pitch % 12
                
                if current_pc not in valid_pcs:
                    distances = [(min(abs(current_pc - vpc), 12 - abs(current_pc - vpc)), vpc) for vpc in valid_pcs]
                    min_dist = min(d[0] for d in distances)
                    best_pcs = [d[1] for d in distances if d[0] == min_dist]
                    nearest_pc = random.choice(best_pcs) # Random tie-breaking
                    
                    diff = nearest_pc - current_pc
                    if diff > 6: diff -= 12
                    elif diff < -6: diff += 12
                    
                    # Use standard random for scalar, and clamp to 0-127 MIDI range!
                    if random.random() < pitch_strength:
                        note.pitch = max(0, min(127, note.pitch + diff))
                        
        logger.debug(f"[MIDIPolisher] Polished MIDI with quant={quant_strength:.2f}, pitch={pitch_strength:.2f}")
        return midi_obj