import bisect
import random
import pretty_midi
from typing import Optional, List

from shared.music_theory_constants import logger, NOTE_MAP, CHORD_INTERVALS, ADAPTER_PROFILES
from shared.gm_instruments import BASS_PROGRAMS, infer_role  # infer_role added for the performance layer below

# GM's "Sound Effects 2" (120-127: guitar fret noise through gunshot) ranges 
# are never a legitimate musical choice for any genre this pipeline targets. 
# A note landing here (e.g. the "Percussion" → program 128 → clamped-to-127 
# "Gunshot" bug) gets remapped to a cohesive active instrument. 
# Synth FX (96-103) are preserved as they are musical synth pads.
_JUNK_PROGRAM_RANGES = set(range(120, 128))
_JUNK_FALLBACK_PROGRAM = 0  # Acoustic Grand Piano

# ---------------------------------------------------------------------
# PERFORMANCE LAYER TABLES
# ---------------------------------------------------------------------
# These are deliberately simple, hand-tuned heuristic tables — not learned,
# not exhaustive. They're a first pass at "does this mood/role combination
# sound more legato or more detached, more rubato or more metronomic" and
# are very much tunable by ear once you're listening to actual output.

# Base note-length ratio applied to a note's ORIGINAL generated duration,
# before mood adjustment. lead ~= near-legato; bass shortened for
# rhythmic punch; harmony keeps its written sustain (it's usually already
# a held pad/voicing, doesn't need shortening or lengthening).
_ARTICULATION_BY_ROLE = {
    "lead": 0.92,
    "harmony": 1.0,
    "bass": 0.75,
}

# Multiplies the role ratio above. >1 = more legato/sustained,
# <1 = more detached/marcato. Unlisted moods default to 1.0 (no change).
_ARTICULATION_MOOD_ADJUST = {
    "heroic": 0.92, "tension": 0.90, "angry": 0.85, "dark": 0.90,
    "sad": 1.05, "melancholy": 1.05, "grief": 1.08, "tender": 1.08,
    "calm": 1.05, "peaceful": 1.08,
}

# How much expressive micro-timing deviation (rubato) a mood gets, as a
# fraction of a sixteenth-note grid unit. Melancholy/tender/calm moods —
# exactly the register this whole codebase keeps building cinematic
# ballads in — get real rubato; heroic/tension/angry moods get almost
# none, since those want to feel driving and metronomic instead.
_RUBATO_BY_MOOD = {
    "sad": 0.35, "melancholy": 0.35, "grief": 0.40, "tender": 0.30,
    "calm": 0.20, "peaceful": 0.20,
    "heroic": 0.05, "tension": 0.05, "angry": 0.0, "excited": 0.05, "happy": 0.10, "dark": 0.10,
}
_DEFAULT_RUBATO = 0.15

# How much a note's velocity swings with the section's density curve, per
# role. lead gets the full expressive range; bass stays comparatively
# grounded (a bass line surging in volume every bar reads as a mixing
# mistake, not expression).
_DYNAMIC_SWING_BY_ROLE = {
    "lead": 0.5,
    "harmony": 0.3,
    "bass": 0.15,
}


class MIDIPolisher:
    def _determine_active_profile(self, ties_weights: Optional[dict]) -> dict:
        weights = ties_weights or {"commu_lora": 0.4, "emopia_lora": 0.3, "slakh_lora": 0.3}
        dominant = max(weights, key=weights.get)
        name = dominant.replace("_lora", "")
        if weights[dominant] >= 0.5:
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
        self._chord_starts = [e["start"] for e in chord_timeline]
        self._chord_timeline = chord_timeline

    def _get_active_chord(self, current_time_sec: float) -> str:
        if not hasattr(self, '_chord_starts'): return "N"
        idx = bisect.bisect_right(self._chord_starts, current_time_sec) - 1
        if idx < 0: return "N"
        event = self._chord_timeline[idx]
        if current_time_sec < event["end"]:
            return event["chord"]
        return "N"

    def polish(self, midi_obj: pretty_midi.PrettyMIDI, bpm: float,
               chord_timeline: list, ties_weights: dict = None,
               density_curve: Optional[List[float]] = None, mood: str = "calm",
               bars: int = 8, beats_per_bar: int = 4) -> pretty_midi.PrettyMIDI:

        profile = self._determine_active_profile(ties_weights)
        quant_strength = profile.get("quant_strength", 0.65)
        pitch_strength = profile.get("pitch_strength", 0.80)

        self._build_chord_index(chord_timeline)
        beat_duration = 60.0 / bpm
        sixteenth_grid = beat_duration / 4.0
        bar_duration = max(1e-6, beats_per_bar * beat_duration)

        mood_key = (mood or "calm").lower()
        rubato_amount = _RUBATO_BY_MOOD.get(mood_key, _DEFAULT_RUBATO)
        articulation_mood_mult = _ARTICULATION_MOOD_ADJUST.get(mood_key, 1.0)

        # Get list of all non-junk programs in the MIDI object to use as cohesive fallback
        active_clean_programs = [i.program for i in midi_obj.instruments if i.program not in _JUNK_PROGRAM_RANGES and not i.is_drum]
        fallback_program = active_clean_programs[0] if active_clean_programs else _JUNK_FALLBACK_PROGRAM

        for inst in midi_obj.instruments:
            if inst.is_drum: continue

            # JUNK INSTRUMENT REMAP — see _JUNK_PROGRAM_RANGES above.
            if inst.program in _JUNK_PROGRAM_RANGES:
                logger.warning(f"⚠️ [Polish] Instrument program {inst.program} is in GM's Sound Effects "
                                f"range — remapping to cohesive fallback program ({fallback_program}).")
                inst.program = fallback_program

            # Use functional BASS_PROGRAMS to protect Tuba (58) and Contrabass (43)
            is_bass = inst.program in BASS_PROGRAMS
            # PERFORMANCE LAYER: role drives articulation/dynamics/rubato
            # shaping below. Deliberately separate from is_bass — is_bass
            # only gates the existing mud-prevention octave shift and stays
            # untouched, so nothing about that logic's behavior changes.
            role = infer_role(inst.program)
            articulation_ratio = max(0.3, min(1.15, _ARTICULATION_BY_ROLE.get(role, 1.0) * articulation_mood_mult))
            dynamic_swing = _DYNAMIC_SWING_BY_ROLE.get(role, 0.3)

            # Sorting isn't strictly required for the passes below (each
            # note is handled independently), but keeps behavior
            # deterministic and makes any future "look at the neighboring
            # note" articulation logic safe to add without re-deriving order.
            inst.notes.sort(key=lambda n: n.start)

            for note in inst.notes:
                original_duration = note.end - note.start
                if original_duration <= 0: continue
                
                # 1. SOFT QUANTIZATION (Fixes Rhythmic Jitter without regeneration)
                grid_time = round(note.start / sixteenth_grid) * sixteenth_grid
                note.start = note.start + quant_strength * (grid_time - note.start)

                # 1b. RUBATO — a small expressive deviation AWAY from the
                # grid, applied AFTER quantization so it doesn't fight the
                # pocket-tightening pass, and role-gated to lead/harmony
                # only (bass and drums stay tight — the brief's own
                # instruction was rubato in melody/strings phrasing, not in
                # the rhythm section). Amount scales with mood: melancholy/
                # tender ballads get real push-and-pull; heroic/tension
                # sections stay closer to metronomic.
                if role in ("lead", "harmony") and rubato_amount > 0:
                    jitter = random.uniform(-1, 1) * rubato_amount * sixteenth_grid * 0.5
                    note.start = max(0.0, note.start + jitter)

                # 1c. ARTICULATION — reshape note length by role + mood
                # instead of always keeping the raw generated duration.
                # This replaces (not stacks with) step 1/1b's naive
                # start+original_duration, applying the ratio to the
                # ORIGINAL duration so articulation and rubato don't
                # compound unpredictably.
                note.end = note.start + max(0.05, original_duration * articulation_ratio)

                # 2. TIMELINE-AWARE CHORD CORRECTION
                current_chord = self._get_active_chord(note.start)
                valid_pcs = self._get_chord_pcs(current_chord)
                current_pc = note.pitch % 12
                
                if current_pc not in valid_pcs:
                    distances = [(min(abs(current_pc - vpc), 12 - abs(current_pc - vpc)), vpc) for vpc in valid_pcs]
                    min_dist = min(d[0] for d in distances)
                    best_pcs = [d[1] for d in distances if d[0] == min_dist]
                    nearest_pc = random.choice(best_pcs)
                    diff = nearest_pc - current_pc
                    if diff > 6: diff -= 12
                    elif diff < -6: diff += 12
                    if random.random() < pitch_strength:
                        note.pitch = max(0, min(127, note.pitch + diff))
                        
                # 3. AUTOMATIC MUD PREVENTION (Fixes Register Separation)
                # 🚀 FIX: Lowered threshold to C2 (36) so Cellos/bass strings aren't incorrectly transposed
                if not is_bass and note.pitch < 36:
                    note.pitch += 12

                # 4. DYNAMICS ENVELOPE — scale velocity by this section's
                # density curve at the note's bar position, so instruments
                # actually swell and recede with the arrangement's energy
                # instead of sitting at one flat velocity for the whole
                # section (SoftRefiner's mood-based min/max bounds still
                # apply upstream; this modulates WITHIN that range rather
                # than overriding it).
                if density_curve:
                    bar_idx = int(note.start / bar_duration)
                    bar_idx = max(0, min(bar_idx, len(density_curve) - 1))
                    density_here = density_curve[bar_idx]
                    dynamic_scale = 1.0 + (density_here - 0.5) * dynamic_swing
                    note.velocity = max(1, min(127, int(round(note.velocity * dynamic_scale))))
                        
        return midi_obj