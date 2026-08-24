import pretty_midi
import numpy as np
from typing import Dict, List, Tuple, Optional
from shared.music_theory_constants import logger, NOTE_MAP, CHORD_INTERVALS, INSTRUMENT_FAMILIES
from shared.gm_instruments import get_program, LOW_REGISTER_SAFE_PROGRAMS


class HardScorer:
    """
    The 'Grammy Mastering Engineer'.
    Grades the polished MIDI against the JSON Blueprint, checking not just
    harmonic correctness, but voice leading, rhythmic pocket, and mix clarity.

    SYMPHONIC MODE: `bars * 4` (section length, and the no-chord-timeline
    downbeat fallback) now reads `beats_per_bar` off the blueprint instead of
    assuming 4/4.

    BUGFIX (inst_score): `_score_instruments_and_register` now checks BOTH
    the INSTRUMENT_FAMILIES bucket AND the exact get_program() match for
    every target instrument, instead of only falling back to the exact
    match when the family bucket was empty. Previously, target "Strings"
    (get_program -> 48, String Ensemble) was scored against
    INSTRUMENT_FAMILIES["Strings"] = range(40,48) = programs 40-47, which
    does NOT include 48 — so a correctly-generated Strings part was scored
    as missing on every section that used it, since "Strings" is by far the
    most common non-piano target instrument in this pipeline.

    BUGFIX (mud penalty): now uses LOW_REGISTER_SAFE_PROGRAMS from
    shared.gm_instruments (BASS_PROGRAMS plus Cello/Bassoon — the same set
    MIDIPolisher's protection logic is built on) instead of the narrower
    INSTRUMENT_FAMILIES["Bass"] range, and caps the penalty so a busy
    orchestral texture can't wipe out the whole instrument score.

    IMPROVEMENT (voice leading): now scores every non-drum instrument
    instead of only a narrow bass+lead allowlist. That allowlist made sense
    when harmony instruments all played an identical repeated chord stack
    (no real "voice" to grade); now that ChordRealizer spreads harmony
    across register-separated single-tone parts, those parts have genuine
    voice-leading behavior worth scoring.
    """

    def score(self, midi_obj: pretty_midi.PrettyMIDI, blueprint: dict,
              primer_midi: Optional[pretty_midi.PrettyMIDI] = None,
              section_name: str = "verse") -> Tuple[float, dict]:

        metrics = {}
        feedback_strs = []

        bpm = blueprint.get("bpm", 120)
        bars = blueprint.get("bars", 8)
        beats_per_bar = blueprint.get("beats_per_bar", 4) or 4
        beat_duration = 60.0 / bpm
        section_end_time = bars * beats_per_bar * beat_duration
        chord_timeline = blueprint.get("chord_timeline", [])

        section_type = section_name.lower()
        is_intro_or_outro = "intro" in section_type or "outro" in section_type

        if "target_chords" in blueprint and blueprint["target_chords"]:
            chord_score, chord_fb = self._score_chord_and_bass(
                midi_obj, blueprint["target_chords"], chord_timeline, section_end_time, beat_duration, beats_per_bar,
                is_intro_or_outro=is_intro_or_outro
            )
            metrics["chord_score"] = chord_score
            feedback_strs.append(chord_fb)

        if "target_instruments" in blueprint and blueprint["target_instruments"]:
            inst_score, inst_fb = self._score_instruments_and_register(
                midi_obj, blueprint["target_instruments"], is_intro_or_outro=is_intro_or_outro
            )
            metrics["inst_score"] = inst_score
            feedback_strs.append(inst_fb)

        vl_score, vl_fb = self._score_voice_leading(midi_obj)
        metrics["voice_leading_score"] = vl_score
        feedback_strs.append(vl_fb)

        rhythm_score, rhythm_fb = self._score_rhythmic_pocket(midi_obj, beat_duration)
        metrics["rhythm_score"] = rhythm_score
        feedback_strs.append(rhythm_fb)

        section_type = section_name.lower()
        if "chorus" in section_type:
            weights = {"chord": 0.3, "inst": 0.3, "voice_leading": 0.1, "rhythm": 0.3}
        elif "bridge" in section_type:
            weights = {"chord": 0.3, "inst": 0.2, "voice_leading": 0.3, "rhythm": 0.2}
        elif "intro" in section_type or "outro" in section_type:
            weights = {"chord": 0.2, "inst": 0.2, "voice_leading": 0.3, "rhythm": 0.3}
        else:
            weights = {"chord": 0.4, "inst": 0.2, "voice_leading": 0.2, "rhythm": 0.2}

        final_score = (
            metrics.get("chord_score", 1.0) * weights["chord"] +
            metrics.get("inst_score", 1.0) * weights["inst"] +
            metrics.get("voice_leading_score", 1.0) * weights["voice_leading"] +
            metrics.get("rhythm_score", 1.0) * weights["rhythm"]
        )

        return final_score, {"metrics": metrics, "feedback": " | ".join(feedback_strs)}

    # ------------------------------------------------------------------
    # CHORD & BASS ADHERENCE
    # ------------------------------------------------------------------
    def _score_chord_and_bass(self, midi_obj: pretty_midi.PrettyMIDI, target_chords: List[str],
                               chord_timeline: List[dict], section_end: float, beat_dur: float,
                               beats_per_bar: int = 4, is_intro_or_outro: bool = False) -> Tuple[float, str]:
        all_notes = []
        for inst in midi_obj.instruments:
            if not inst.is_drum:
                valid_notes = [n for n in inst.notes if n.start < section_end]
                all_notes.extend(valid_notes)

        if not all_notes:
            return 0.0, "Error: No notes generated."

        matches = 0
        total_checked = 0

        if chord_timeline:
            for event in chord_timeline:
                root_pc = self._get_root_pc(event.get("chord", "N"))
                if root_pc == -1:
                    continue
                downbeat_time = event["start"]
                # 🚀 FIX: Widen window to catch rubato/strummed chords
                active_notes = [n for n in all_notes if n.start <= downbeat_time + 0.3 and n.end > downbeat_time - 0.1]
                
                # In intros/outros, we allow silence or fade outs/ins, so if no notes are active, we skip scoring this step
                if is_intro_or_outro and not active_notes:
                    continue
                    
                total_checked += 1
                if active_notes:
                    if is_intro_or_outro:
                        # Lenient match: check if root note is present anywhere in the chord stack
                        if any(n.pitch % 12 == root_pc for n in active_notes):
                            matches += 1
                    else:
                        lowest_note = min(active_notes, key=lambda n: n.pitch)
                        if lowest_note.pitch % 12 == root_pc:
                            matches += 1
        else:
            root_pcs = [self._get_root_pc(c) for c in target_chords if self._get_root_pc(c) != -1]
            for i, target_root in enumerate(root_pcs):
                downbeat_time = i * beats_per_bar * beat_dur
                active_notes = [n for n in all_notes if n.start <= downbeat_time + 0.3 and n.end > downbeat_time - 0.1]
                
                # In intros/outros, we allow silence or fade outs/ins, so if no notes are active, we skip scoring this step
                if is_intro_or_outro and not active_notes:
                    continue
                    
                total_checked += 1
                if active_notes:
                    if is_intro_or_outro:
                        # Lenient match: check if root note is present anywhere in the chord stack
                        if any(n.pitch % 12 == target_root for n in active_notes):
                            matches += 1
                    else:
                        lowest_note = min(active_notes, key=lambda n: n.pitch)
                        if lowest_note.pitch % 12 == target_root:
                            matches += 1

        score = matches / total_checked if total_checked > 0 else 1.0
        return score, f"Chord/Bass Match: {score*100:.0f}%"

    def _get_root_pc(self, chord_symbol: str) -> int:
        """
        Returns the pitch class the BASS/lowest voice is actually expected
        to land on. For a plain chord this is the nominal root. For a slash
        chord like "F/A", it's A — the specified bass note — NOT F.

        BUGFIX: previously this always returned the pre-slash root
        regardless of inversion notation, so a correctly-realized inversion
        (ChordRealizer deliberately puts the specified bass note in the
        bass voice, not the chord root — see bass_override_pc in
        chord_realizer.py) was scored as a complete chord/bass mismatch.
        This only ever affected explicit/user-specified progressions, since
        Markov-generated chords never carry inversions.
        """
        if not chord_symbol or chord_symbol in ("N", "s"):
            return -1
        symbol = chord_symbol
        if "/" in symbol:
            _, bass_str = symbol.split("/", 1)
            bass_str = bass_str.strip()
            bass_root = bass_str[:2] if len(bass_str) >= 2 and bass_str[1] in ('#', 'b') else bass_str[:1]
            return NOTE_MAP.get(bass_root, -1)
        root_str = symbol[:2] if len(symbol) >= 2 and symbol[1] in ('#', 'b') else symbol[0]
        return NOTE_MAP.get(root_str, -1)

    # ------------------------------------------------------------------
    # INSTRUMENT DENSITY & REGISTER SEPARATION
    # ------------------------------------------------------------------
    def _score_instruments_and_register(self, midi_obj: pretty_midi.PrettyMIDI, target_instruments: List[str], is_intro_or_outro: bool = False) -> Tuple[float, str]:
        generated_programs = [inst.program for inst in midi_obj.instruments if not inst.is_drum]
        has_drum_track = any(inst.is_drum for inst in midi_obj.instruments)

        found_families = 0
        total_families = len(target_instruments)
        mud_penalty = 0.0

        for target in target_instruments:
            t_lower = target.lower()
            # 🚀 DRUM FIX: Explicitly check for drum tracks
            if t_lower in ["drums", "percussion", "drum"]:
                if has_drum_track:
                    found_families += 1
                continue

            # BUGFIX: check the family bucket AND the exact get_program()
            # match, always — not "family bucket, falling back to exact
            # match only if the bucket was empty." A non-empty-but-wrong
            # bucket (e.g. "Strings" -> range(40,48), which excludes the 48
            # that get_program("Strings") actually returns) used to mask a
            # correct generation as a miss.
            valid_programs = set(INSTRUMENT_FAMILIES.get(target, []))
            valid_programs.add(get_program(target))

            if any(prog in valid_programs for prog in generated_programs):
                found_families += 1

        for inst in midi_obj.instruments:
            if not inst.is_drum and inst.program not in LOW_REGISTER_SAFE_PROGRAMS:
                low_notes = sum(1 for n in inst.notes if n.pitch < 48)
                if low_notes > 5:
                    mud_penalty += 0.1
        # Cap so a dense orchestral texture with several genuinely-low parts
        # can't drag the instrument score to zero on penalty alone.
        mud_penalty = min(mud_penalty, 0.5)

        base_score = found_families / total_families if total_families > 0 else 1.0
        if is_intro_or_outro and found_families > 0:
            # Give a leniency boost to sparse arrangements
            base_score = 0.5 + (base_score * 0.5)

        final_score = max(0.0, base_score - mud_penalty)
        return final_score, f"Inst: {found_families}/{total_families} (Mud Penalty: {mud_penalty:.1f})"

    # ------------------------------------------------------------------
    # VOICE LEADING SMOOTHNESS
    # ------------------------------------------------------------------
    def _score_voice_leading(self, midi_obj: pretty_midi.PrettyMIDI) -> Tuple[float, str]:
        total_leaps = 0
        severe_leaps = 0

        for inst in midi_obj.instruments:
            if inst.is_drum:
                continue

            # IMPROVEMENT: previously restricted to a narrow
            # bass+lead-ish-programs allowlist, so harmony-role instruments
            # (Strings, French Horn, Choir, etc.) were never checked at all.
            # Now every non-drum part is scored — meaningful now that
            # register-spread harmony voices carry real independent lines
            # rather than repeating an identical chord stack every beat.
            notes = sorted(inst.notes, key=lambda n: n.start)
            for i in range(1, len(notes)):
                leap = abs(notes[i].pitch - notes[i - 1].pitch)
                total_leaps += 1
                if leap > 12:
                    severe_leaps += 1

        if total_leaps == 0:
            return 1.0, "VL: N/A"

        leap_ratio = severe_leaps / total_leaps
        score = max(0.0, 1.0 - (leap_ratio * 2))
        return score, f"Voice Leading: {score*100:.0f}%"

    # ------------------------------------------------------------------
    # RHYTHMIC POCKET (Grid vs Jitter)
    # ------------------------------------------------------------------
    def _score_rhythmic_pocket(self, midi_obj: pretty_midi.PrettyMIDI, beat_duration: float) -> Tuple[float, str]:
        sixteenth_grid = beat_duration / 4.0
        if sixteenth_grid <= 0:
            return 1.0, "Rhythm: N/A"

        deviations = []
        for inst in midi_obj.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                nearest_grid = round(note.start / sixteenth_grid) * sixteenth_grid
                dev = abs(note.start - nearest_grid) / sixteenth_grid
                deviations.append(min(dev, 0.5))

        if not deviations:
            return 1.0, "Rhythm: N/A"

        avg_deviation = sum(deviations) / len(deviations)
        score = max(0.0, 1.0 - (avg_deviation * 2))
        return score, f"Rhythmic Pocket: {score*100:.0f}%"