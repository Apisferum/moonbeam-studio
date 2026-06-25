import random
from typing import Optional
# Import the centralized constants
from shared.music_theory_constants import (
    logger, NOTE_MAP, CHORD_INTERVALS, ADAPTER_PROFILES, 
    SCALE_MODES, DRUM_INSTRUMENT_IDS
)

class SoftRefiner:
    def __init__(self):
        # No more hardcoded dictionaries! We use the centralized constants.
        self.drum_ids = DRUM_INSTRUMENT_IDS

    def _determine_active_profile(self, ties_weights: Optional[dict]) -> dict:
        weights = ties_weights or {"commu_lora": 0.6, "emopia_lora": 0.2, "slakh_lora": 0.2}
        dominant = max(weights, key=weights.get)
        name = dominant.replace("_lora", "")
        if weights[dominant] >= 0.6:
            return ADAPTER_PROFILES.get(name, ADAPTER_PROFILES["hybrid"])
        return ADAPTER_PROFILES["hybrid"]

    def _parse_chord(self, chord_str: str):
        if not chord_str or chord_str in ("s", "N"): return None, None
        if len(chord_str) >= 2 and chord_str[1] in ('#', 'b'):
            root_str, quality = chord_str[:2], chord_str[2:]
        else:
            root_str, quality = chord_str[0], chord_str[1:]
            
        quality = quality.replace("minor", "min").replace("major", "maj")
        if quality in ("", "M"): quality = "maj"
        if quality == "m": quality = "min"
        return NOTE_MAP.get(root_str, 0), quality

    def refine_tokens(self, tokens: list, root_note: str = "C", mode: str = "major", 
                      current_chord: str = None, ties_weights: dict = None) -> list:
        
        profile = self._determine_active_profile(ties_weights)
        
        # 1. Get Global Scale (Using centralized SCALE_MODES)
        root_pc = NOTE_MAP.get(root_note, 0)
        # Fallback to "major" if the user passes an unsupported mode string
        intervals = SCALE_MODES.get(mode, SCALE_MODES["major"])
        scale_pcs = [(root_pc + i) % 12 for i in intervals]
        
        # 2. Get Local Chord
        chord_pcs = None
        if current_chord:
            r_pc, quality = self._parse_chord(current_chord)
            if r_pc is not None:
                chord_intervals = CHORD_INTERVALS.get(quality)
                if chord_intervals:
                    chord_pcs = [(r_pc + i) % 12 for i in chord_intervals]
                else:
                    chord_pcs = scale_pcs 
        
        valid_pcs = chord_pcs if chord_pcs else scale_pcs
        strength = profile["pitch_strength"]
        allow_accidentals = profile["allow_accidentals"]
        
        snapped_count = 0
        refined = []
        
        for tok in tokens:
            # Moonbeam tokens: [ts, dur, oct, pitch_class, inst, vel]
            if len(tok) == 6 and 0 <= tok[3] <= 11:
                inst_id = tok[4]
                pc = tok[3]
                
                # Use centralized DRUM_INSTRUMENT_IDS
                if inst_id in self.drum_ids:
                    refined.append(tok)
                    continue
                
                if pc not in valid_pcs:
                    if allow_accidentals:
                        refined.append(tok)
                    else:
                        if random.random() < strength:
                            # Find nearest valid PC with random tie-breaking
                            min_dist = 12
                            best_pcs = []
                            for vpc in valid_pcs:
                                dist = min(abs(pc - vpc), 12 - abs(pc - vpc))
                                if dist < min_dist:
                                    min_dist = dist
                                    best_pcs = [vpc]
                                elif dist == min_dist:
                                    best_pcs.append(vpc)
                            
                            best_pc = random.choice(best_pcs)
                            
                            tok = list(tok)
                            tok[3] = best_pc
                            snapped_count += 1
                            refined.append(tok)
                        else:
                            refined.append(tok)
                else:
                    refined.append(tok)
            else:
                refined.append(tok)
                
        logger.debug(f"[SoftRefiner] Snapped {snapped_count} tokens | Chord: {current_chord}")
        return refined