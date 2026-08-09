import random
from typing import Optional, List, Set
from shared.music_theory_constants import (
    logger, NOTE_MAP, CHORD_INTERVALS, ADAPTER_PROFILES,
    SCALE_MODES, DRUM_INSTRUMENT_IDS
)

class SoftRefiner:
    """
    The Token-Level Gatekeeper (Pre-MIDI Sanitizer).
    Operates on locally-scoped ABSOLUTE onsets AFTER generation but BEFORE MIDI compilation.

    EXPLICIT-CONTENT FIX: refine_tokens now accepts `protected_pcs` — a set
    of pitch classes that must NEVER be snapped to the declared scale, even
    if they fall outside it. Without this, any deliberately chromatic note
    coming from an explicit_chords entry (secondary dominants, borrowed
    chords) or an explicit_melody token (passing tones, blue notes,
    accidentals) would get silently "corrected" back toward the nearest
    diatonic pitch — quietly undoing exactly the thing the user asked for.
    Algorithmically-generated (Markov) chords never needed this, since
    they're already built from the declared scale by construction; this
    only matters for user-supplied harmonic/melodic content.
    """
    def __init__(self):
        self.drum_ids = DRUM_INSTRUMENT_IDS
        self.valid_drum_pitches = [36, 38, 42, 46, 45, 48, 50] 
        
        self.emopia_velocity_bounds = {
            "emo_q1": (70, 110), "emo_q2": (80, 127),
            "emo_q3": (30, 70),  "emo_q4": (40, 80),
        }
        self.quadrant_map = {
            "happy": "emo_q1", "heroic": "emo_q1", "excited": "emo_q1",
            "tension": "emo_q2", "angry": "emo_q2", "dark": "emo_q2",
            "sad": "emo_q3", "melancholy": "emo_q3", "grief": "emo_q3",
            "calm": "emo_q4", "peaceful": "emo_q4", "tender": "emo_q4"
        }

    def _determine_active_profile(self, ties_weights: Optional[dict]) -> dict:
        weights = ties_weights or {"commu_lora": 0.4, "emopia_lora": 0.3, "slakh_lora": 0.3}
        dominant = max(weights, key=weights.get)
        name = dominant.replace("_lora", "")
        if weights[dominant] >= 0.5:
            return ADAPTER_PROFILES.get(name, ADAPTER_PROFILES["hybrid"])
        return ADAPTER_PROFILES["hybrid"]

    def refine_tokens(self, tokens: list, root_note: str = "C", mode: str = "major", 
                      max_section_ticks: int = 1600, mood: str = "calm",
                      ties_weights: dict = None,
                      protected_pcs: Optional[Set[int]] = None) -> list:
        
        profile = self._determine_active_profile(ties_weights)
        root_pc = NOTE_MAP.get(root_note, 0)
        intervals = SCALE_MODES.get(mode, SCALE_MODES["major"])
        scale_pcs = [(root_pc + i) % 12 for i in intervals]
        protected = set(protected_pcs or [])
        
        valid_pcs = scale_pcs
        strength = profile["pitch_strength"]
        allow_accidentals = profile["allow_accidentals"]
        
        q = self.quadrant_map.get(mood.lower(), "emo_q4")
        vel_min, vel_max = self.emopia_velocity_bounds.get(q, (40, 100))

        snapped_pitch = clipped_dur = dropped_notes = clamped_vel = sanitized_drums = protected_kept = 0
        refined = []
        
        for tok in tokens:
            if len(tok) == 6:
                abs_time, dur, oct, pc, inst, vel = tok # tok[0] is LOCAL ABSOLUTE time
                
                if abs_time >= max_section_ticks:
                    dropped_notes += 1
                    continue 
                    
                if abs_time + dur > max_section_ticks:
                    dur = max(1, max_section_ticks - abs_time)
                    clipped_dur += 1
                
                if inst in self.drum_ids or inst == 128:
                    if pc not in self.valid_drum_pitches:
                        pc = 36 if random.random() < 0.5 else 38 
                        sanitized_drums += 1
                elif pc not in valid_pcs:
                    if pc in protected:
                        # Deliberately chosen by the user (explicit chord
                        # tone or explicit melody note) — leave it alone
                        # even though it's outside the declared scale.
                        protected_kept += 1
                    elif not allow_accidentals and random.random() < strength:
                        min_dist = 12
                        best_pcs = []
                        for vpc in valid_pcs:
                            dist = min(abs(pc - vpc), 12 - abs(pc - vpc))
                            if dist < min_dist: min_dist = dist; best_pcs = [vpc]
                            elif dist == min_dist: best_pcs.append(vpc)
                        pc = random.choice(best_pcs)
                        snapped_pitch += 1
                
                if vel < vel_min: vel = vel_min; clamped_vel += 1
                elif vel > vel_max: vel = vel_max; clamped_vel += 1
                
                refined.append([abs_time, dur, oct, pc, inst, vel])
            else:
                refined.append(tok)
                
        logger.debug(f"[SoftRefiner] Dropped {dropped_notes} | Clipped {clipped_dur} | Snapped {snapped_pitch} | "
                     f"Clamped {clamped_vel} | Drums {sanitized_drums} | Protected-kept {protected_kept}")
        return refined