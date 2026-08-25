import math
import re
import random
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

# Import the Single Source of Truth
from shared.music_theory_constants import (
    logger, NOTE_MAP, PC_TO_NAME, SCALE_MODES,
    HARMONIC_FUNCTIONS_MAJOR, HARMONIC_FUNCTIONS_MINOR, MACRO_SKELETONS,
    MAJOR_QUALITIES, MINOR_QUALITIES, ARRANGEMENT_PROFILES, TIES_PRESETS,
    # SYMPHONIC MODE: new names — see music_theory_constants_symphonic_additions.py
    TIME_SIGNATURES, DEFAULT_TIME_SIGNATURE, CLASSICAL_MACRO_SKELETONS,
    ORCHESTRAL_ARRANGEMENT_PROFILES, ORCHESTRAL_DEFAULT_VOICES,
)
from brain.chord_realizer import ChordRealizer, _chord_tones


class StructurePlanner:
    """
    The 'Algorithmic Composer' (Hybrid Edition).
    Combines Harmonic Function Skeletons (Macro) with Markov Creativity (Micro),
    while generating the rich metadata required by the HarmonyRouter and Critic.

    Also runs ChordRealizer per section, attaching a `note_events` list
    (octave/pitch_class/instrument per note, chord-tone-correct) to each
    timeline entry. This is additive - every key that existed before still
    exists with the same meaning, so tasks.py / agentic_composer.py callers
    that only read the old keys are unaffected.

    SYMPHONIC MODE: plan() now resolves a `global_time_signature` (default
    "4/4") into a `beats_per_bar` int that replaces every previously
    hardcoded `* 4` in the beat math below, and every section_entry now
    carries `time_signature` / `beats_per_bar` so downstream consumers
    (ChordRealizer, HardScorer, AgenticComposer) stay in sync instead of
    silently assuming 4/4. Section names that match CLASSICAL_MACRO_SKELETONS
    (e.g. "Exposition", "Development", "Theme") now resolve to real harmonic
    skeletons instead of collapsing to the generic ["T","S","D","T"] default.
    style == "orchestral"/"symphonic"/"classical" now pulls instrumentation
    from ORCHESTRAL_ARRANGEMENT_PROFILES instead of the pop-oriented
    ARRANGEMENT_PROFILES bucket.
    """
    def __init__(self, octave_vocab_size: int = 9, pitch_class_vocab_size: int = 12,
                 instrument_vocab_size: int = 130):
        # Vocab sizes should be passed in from the live model config
        # (llama_config.octave_vocab_size etc.) wherever this planner is
        # constructed, so realized notes never collide with sos/eos ids.
        # Defaults here match MusicTokenizer's fallback defaults.
        self.chord_realizer = ChordRealizer(
            octave_vocab_size=octave_vocab_size,
            pitch_class_vocab_size=pitch_class_vocab_size,
            instrument_vocab_size=instrument_vocab_size,
        )

    def _resolve_time_signature(self, raw_ts) -> Tuple[str, int]:
        """Returns (normalized_ts_string, beats_per_bar). Falls back to 4/4
        for anything unrecognized rather than crashing the whole plan()."""
        ts = (raw_ts or DEFAULT_TIME_SIGNATURE).strip()
        entry = TIME_SIGNATURES.get(ts)
        if entry is None:
            logger.warning(f"⚠️ [Planner] Unknown time signature '{raw_ts}', defaulting to {DEFAULT_TIME_SIGNATURE}.")
            ts = DEFAULT_TIME_SIGNATURE
            entry = TIME_SIGNATURES[DEFAULT_TIME_SIGNATURE]
        return ts, entry["beats_per_bar"]

    def _get_relative_key(self, root_pc: int, mode: str) -> tuple:
        """Calculates the Relative Major/Minor (shares the exact same key signature)."""
        if mode == "minor":
            new_root = (root_pc + 3) % 12
            return PC_TO_NAME[new_root], "major"
        else:
            new_root = (root_pc + 9) % 12
            return PC_TO_NAME[new_root], "minor"

    def _degree_to_chord(self, degree: str, key_pc: int, scale_intervals: list, is_minor: bool) -> Tuple[str, str, int]:
        clean = degree.replace("°", "")
        roman_to_idx = {"i": 0, "I": 0, "ii": 1, "iii": 2, "III": 2, "iv": 3, "IV": 3, "v": 4, "V": 4, "vi": 5, "VI": 5, "vii": 6, "VII": 6}
        idx = roman_to_idx.get(clean, 0)

        interval = scale_intervals[idx] if idx < len(scale_intervals) else scale_intervals[-1]
        chord_root_pc = (key_pc + interval) % 12
        root_name = PC_TO_NAME.get(chord_root_pc, "C")

        qualities = MINOR_QUALITIES if is_minor else MAJOR_QUALITIES
        quality = qualities.get(degree, "maj7")
        if "°" in degree:
            quality = "dim"

        # Also return root_pc so ChordRealizer doesn't have to re-derive
        # it by re-parsing the symbol string.
        return f"{root_name}{quality}", quality, chord_root_pc

    def _parse_explicit_chord(self, symbol: str) -> Tuple[str, str, int, Optional[int]]:
        """
        Parses a user-supplied chord symbol (e.g. 'F#m7', 'Cmaj7', 'G7',
        'F/A') into (symbol, quality, root_pc, bass_override_pc) — the same
        shape _degree_to_chord produces, plus an inversion bass note when a
        slash chord is given. 'F/A' means "F chord, with A in the bass":
        the part before '/' is parsed exactly as any other chord; the part
        after '/' is parsed as a bare note name and returned separately,
        for ChordRealizer's bass voice to emphasize on the downbeat instead
        of the chord root. bass_override_pc is None for plain (non-slash)
        chords, in which case the bass just plays the root as before —
        nothing changes for those. This is the whole trick that makes
        explicit chords "just work" everywhere else in the pipeline: once a
        progression entry has {symbol, quality, root_pc, bass_override_pc,
        duration_beats, start_beat, end_beat}, ChordRealizer, HardScorer,
        and _build_chord_timeline can't tell (and don't need to know)
        whether it came from the Markov selector or straight from the user.
        """
        s = symbol.strip()
        bass_override_pc = None
        if "/" in s:
            s, bass_str = s.split("/", 1)
            bass_str = bass_str.strip()
            bass_root = bass_str[:2] if len(bass_str) >= 2 and bass_str[1] in ('#', 'b') else bass_str[:1]
            if bass_root in NOTE_MAP:
                bass_override_pc = NOTE_MAP[bass_root]

        root_str = s[:2] if len(s) >= 2 and s[1] in ('#', 'b') else s[:1]
        quality = s[2:] if len(s) >= 2 and s[1] in ('#', 'b') else s[1:]
        root_pc = NOTE_MAP.get(root_str, 0)
        # Defensive: strip parens even though the LLM is instructed to
        # write "Dmadd9" rather than "Dm(add9)" — a literal "(" breaks
        # contiguous substring matching in _chord_tones (e.g. "m(add9)"
        # doesn't contain "madd9" as a substring), silently losing the
        # extension. Stripping here costs nothing and doesn't depend on
        # the LLM actually following that instruction.
        quality = quality.replace("(", "").replace(")", "").strip().lower() or "maj"
        return symbol.strip(), quality, root_pc, bass_override_pc

    def _build_explicit_progression(self, chords: List[str], durations: Optional[List[float]],
                                     bars: int, beats_per_bar: int) -> List[Dict[str, Any]]:
        """
        Builds a progression directly from a user-supplied chord list instead
        of the Markov function-skeleton selector. If `durations` (in beats)
        is given and matches `chords` in length, each chord holds for its
        specified length and the whole (chord, duration) pattern loops to
        fill the section — so a user can say "hold the first chord twice as
        long" and have that shape repeat every cycle rather than only on the
        first pass. Without explicit durations, one chord fills one bar,
        looping (or truncating) to fit the section's actual bar count —
        e.g. a 4-chord progression handed to a 16-bar section loops 4 times.
        """
        total_beats = bars * beats_per_bar
        n = len(chords)
        if n == 0 or total_beats <= 0:
            return []

        if durations and len(durations) == n and sum(durations) > 0:
            pattern = list(zip(chords, durations))
            reps = max(1, math.ceil(total_beats / sum(durations)))
            tiled = pattern * reps
        else:
            reps = math.ceil(bars / n)
            tiled = [(c, beats_per_bar) for c in (chords * reps)[:bars]]

        progression = []
        total = 0.0
        for symbol, dur in tiled:
            if total >= total_beats:
                break
            dur = max(0.5, float(dur))
            if total + dur > total_beats:
                dur = total_beats - total
            # BUGFIX: this was unpacking only 3 values (norm_symbol,
            # quality, root_pc) against a function that returns 4 —
            # _parse_explicit_chord was updated to also return
            # bass_override_pc (for slash-chord inversion support) but
            # this, its only call site, was never updated to match. That's
            # exactly "too many values to unpack (expected 3)". Separately:
            # bass_override_pc was also never actually being stored on the
            # progression entry below, meaning even once this unpack is
            # fixed, ChordRealizer's `chord.get("bass_override_pc")` would
            # still always see None — the inversion feature has been dead
            # code since it was introduced. Both fixed together here.
            norm_symbol, quality, root_pc, bass_override_pc = self._parse_explicit_chord(symbol)
            progression.append({
                "symbol": norm_symbol, "degree": "user", "quality": quality, "root_pc": root_pc,
                "bass_override_pc": bass_override_pc,
                "duration_beats": dur, "start_beat": total, "end_beat": total + dur,
            })
            total += dur
        return progression

    _MELODY_DURATION_CODES = {
        "w": 4.0, "h": 2.0, "q": 1.0, "e": 0.5, "s": 0.25,
        "w.": 6.0, "h.": 3.0, "q.": 1.5, "e.": 0.75, "s.": 0.375,
    }
    _MELODY_NOTE_RE = re.compile(r'^([A-Ga-g])([#b]?)(-?\d+):(.+)$')
    _MELODY_REST_RE = re.compile(r'^[Rr]:(.+)$')

    def _parse_melody_duration(self, token: str) -> Optional[float]:
        token = token.strip()
        if token in self._MELODY_DURATION_CODES:
            return self._MELODY_DURATION_CODES[token]
        try:
            val = float(token)
            return val if val > 0 else None
        except ValueError:
            return None

    def _parse_melody_token(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Parses one explicit_melody token, e.g. 'C4:q', 'F#3:e.', or 'R:h',
        into {'is_rest', 'octave', 'pitch_class', 'duration_beats'}.

        Octave follows scientific pitch notation (C4 = middle C = MIDI 60).
        NOTE: this assumes the model's internal octave index uses that same
        convention (octave index N == scientific-pitch octave N, i.e.
        MIDI = (N+1)*12 + pitch_class). Every indirect clue in this codebase
        points that way — DEFAULT_OCTAVE_BY_ROLE (bass=2 -> ~C2/C1 register,
        harmony=4 -> C4, lead=5 -> C5, all musically sensible defaults), and
        MIDIPolisher's mud-prevention threshold at pitch 36 (= C2, matching
        octave=2 under this formula) — but it hasn't been confirmed directly
        against MusicTokenizer.octave_dict, since that file isn't something
        I've seen. Worth a one-time sanity check: force a single C4 note
        through the pipeline and confirm the output MIDI pitch is actually
        60 before trusting this at scale.
        """
        s = str(raw).strip()
        rest_match = self._MELODY_REST_RE.match(s)
        if rest_match:
            dur = self._parse_melody_duration(rest_match.group(1))
            if dur is None:
                return None
            return {"is_rest": True, "duration_beats": dur}

        m = self._MELODY_NOTE_RE.match(s)
        if not m:
            return None
        letter, accidental, octave_str, dur_token = m.groups()
        root_str = letter.upper() + accidental
        if root_str not in NOTE_MAP:
            return None
        dur = self._parse_melody_duration(dur_token)
        if dur is None:
            return None
        try:
            octave = int(octave_str)
        except ValueError:
            return None
        return {"is_rest": False, "pitch_class": NOTE_MAP[root_str], "octave": octave, "duration_beats": dur}

    def _build_melody_pattern(self, tokens: List[str], bars: int, beats_per_bar: int) -> List[Dict[str, Any]]:
        """
        Parses and tiles an explicit_melody token list across the section,
        the same "loop a short pattern to fill the section" philosophy
        _build_explicit_progression uses for chords — a short riff handed to
        a long section repeats to fill it rather than playing once and
        leaving the rest of the section's lead voice silent. Rests advance
        the timeline without emitting a note. Returns absolute-timed events
        (start_beat/duration_beats/octave/pitch_class) ready for
        ChordRealizer to drop straight in as the lead voice, decoupled from
        the chord/bar grid entirely (a melody doesn't have to phrase on
        chord boundaries).
        """
        total_beats = bars * beats_per_bar
        parsed = []
        for tok in tokens:
            p = self._parse_melody_token(tok)
            if p is None:
                logger.warning(f"⚠️ [Planner] Dropping unparseable melody token: '{tok}'")
                continue
            parsed.append(p)
        if not parsed:
            return []

        pattern_len = sum(p["duration_beats"] for p in parsed)
        if pattern_len <= 0:
            return []
        reps = max(1, math.ceil(total_beats / pattern_len))
        tiled = parsed * reps

        events = []
        total = 0.0
        for p in tiled:
            if total >= total_beats:
                break
            dur = p["duration_beats"]
            if total + dur > total_beats:
                dur = total_beats - total
            if not p["is_rest"] and dur > 0:
                events.append({
                    "start_beat": total, "duration_beats": dur,
                    "octave": p["octave"], "pitch_class": p["pitch_class"],
                })
            total += dur
        return events

    def _generate_function_markov_chords(self, key: str, mode: str, section_name: str, bars: int,
                                          beats_per_bar: int = 4) -> List[Dict[str, Any]]:
        """Generates chords using a Harmonic Function Skeleton + Markov Selection."""
        key_pc = NOTE_MAP.get(key, 0)
        scale_intervals = SCALE_MODES.get(mode, SCALE_MODES["major"])
        is_minor = (mode == "minor")
        func_groups = HARMONIC_FUNCTIONS_MINOR if is_minor else HARMONIC_FUNCTIONS_MAJOR

        # SYMPHONIC MODE: classical section names (Exposition/Development/
        # Theme/etc.) resolve here before falling through to the generic
        # tonic-subdominant-dominant-tonic default.
        base_skeleton = (
            MACRO_SKELETONS.get(section_name)
            or CLASSICAL_MACRO_SKELETONS.get(section_name)
            or ["T", "S", "D", "T"]
        )
        total_bar_beats = bars * beats_per_bar
        skeleton = (base_skeleton * math.ceil(bars / len(base_skeleton)))[:bars]

        if len(skeleton) >= 2:
            skeleton[-2] = "D"
            skeleton[-1] = "T"

        progression = []
        total_beats = 0

        for i, func in enumerate(skeleton):
            degree = random.choice(func_groups[func])
            symbol, quality, root_pc = self._degree_to_chord(degree, key_pc, scale_intervals, is_minor)

            # Was hardcoded 4 / 2 (a full bar / half a bar in 4/4). Now
            # scales with the actual meter: a full bar for non-dominant
            # function, half a bar for dominant (the harmonic-rhythm
            # speedup at cadences still applies, just proportional to
            # whatever beats_per_bar actually is).
            duration_beats = beats_per_bar if func != "D" else max(1, beats_per_bar // 2)

            if total_beats + duration_beats > total_bar_beats:
                duration_beats = total_bar_beats - total_beats

            progression.append({
                "symbol": symbol, "degree": degree, "quality": quality, "root_pc": root_pc,
                "duration_beats": duration_beats, "start_beat": total_beats, "end_beat": total_beats + duration_beats
            })
            total_beats += duration_beats
            if total_beats >= total_bar_beats:
                break

        return progression

    def _generate_density_curve(self, section_name: str, bars: int, energy_curve: str = "build") -> List[float]:
        if bars <= 0:
            return []
        x = np.linspace(0, 1, bars)

        if section_name == "Intro":
            base = 0.2 + 0.3 * x
        elif section_name == "Chorus":
            base = 0.7 + 0.3 * np.sin(x * np.pi)
        elif section_name == "Outro":
            base = 0.6 * (1 - x)
        else:
            base = np.full(bars, 0.5)

        if energy_curve == "build":
            modifier = 0.5 + 0.5 * x
            base = base * modifier
        elif energy_curve == "fade":
            modifier = 1.0 - 0.5 * x
            base = base * modifier

        return np.clip(base, 0.1, 1.0).tolist()

    def _build_chord_timeline(self, progression: List[Dict], bpm: float) -> List[Dict]:
        beat_duration = 60.0 / bpm
        return [{"start": c["start_beat"] * beat_duration, "end": c["end_beat"] * beat_duration, "chord": c["symbol"]} for c in progression]

    def _get_arrangement(self, sections: List[str], mood: str, style: str = "cinematic") -> Dict[str, List[str]]:
        # SYMPHONIC MODE: orchestral/symphonic/classical style pulls from a
        # dedicated profile with orchestral default voices instead of the
        # pop-oriented "Piano"/"Strings" fallback.
        if style in ("orchestral", "symphonic", "classical"):
            profile = ORCHESTRAL_ARRANGEMENT_PROFILES.get("orchestral_tutti", {})
            default_voices = ORCHESTRAL_DEFAULT_VOICES
        else:
            arrangement_style = "build" if mood in ("heroic", "tension") else "sparse_to_dense"
            profile = ARRANGEMENT_PROFILES.get(arrangement_style, ARRANGEMENT_PROFILES["build"])
            default_voices = ["Piano", "Strings"]
        # Copying here prevents later `.insert()` calls (lead-instrument
        # injection) from mutating ARRANGEMENT_PROFILES itself, which would
        # otherwise leak instruments across every future request for the
        # lifetime of the process.
        return {s: list(profile.get(s, default_voices)) for s in sections}

    def _get_ties_weights(self, section_name: str, mood: str, instruments: List[str]) -> Dict[str, float]:
        # FIX: .get("balanced") returned None (and crashed on .copy()) if the
        # key were ever missing/renamed in TIES_PRESETS. .get(..., {}) fails
        # loudly-but-gracefully (empty weights) instead of an AttributeError
        # deep in a request path.
        base = TIES_PRESETS.get("balanced", {}).copy()
        if section_name == "Chorus":
            base["slakh_lora"] = base.get("slakh_lora", 0.0) + 0.15
        if mood in ("sad", "melancholy"):
            base["emopia_lora"] = base.get("emopia_lora", 0.0) + 0.2
        if len(instruments) >= 4:
            base["slakh_lora"] = base.get("slakh_lora", 0.0) + 0.1
        total = sum(base.values())
        if total <= 0:
            return base
        return {k: round(v / total, 3) for k, v in base.items()}

    def plan(self, llm_output: dict) -> dict:
        raw_key = llm_output.get("global_key") or "C"
        # FIX: raw_key could be "" (schema allows empty string), which made
        # "".split()[0] throw IndexError. Falling back to "C" up front closes
        # that gap without needing a schema change.
        global_key = (raw_key.split() or ["C"])[0].replace("minor", "").replace("major", "").strip() or "C"
        global_mode = llm_output.get("global_mode", "major")
        bpm = llm_output.get("global_bpm", 120)
        style = llm_output.get("style", "cinematic")
        energy_curve = llm_output.get("energy_curve", "build")
        lead_inst = llm_output.get("lead_instrument", "Violin") or "Violin"
        modulations = llm_output.get("modulations", [])

        # SYMPHONIC MODE: resolve time signature once per song. beats_per_bar
        # flows into every section below instead of a hardcoded 4.
        time_signature, beats_per_bar = self._resolve_time_signature(llm_output.get("global_time_signature"))

        sections_input = llm_output.get("sections") or [{"name": "Verse", "mood": "happy", "length": "medium"}]
        # FIX: `llm_output.get("sections", [default])` only falls back when the
        # KEY is missing, not when it's present but an empty list — an LLM
        # tier could legally return sections: [] and pass Pydantic validation,
        # crashing sections_input[0] below. `or [...]` catches both cases.

        mod_map = {m["target_section"]: m for m in modulations}

        section_names = [s.get("name", "Verse") for s in sections_input]
        primary_mood = sections_input[0].get("mood", "default")
        arrangement = self._get_arrangement(section_names, primary_mood, style)

        dense_timeline = []

        current_key = global_key
        current_mode = global_mode

        for section in sections_input:
            name = section.get("name", "Verse")
            mood = section.get("mood", primary_mood)

            raw_bars = {"short": 4, "medium": 8, "long": 16}.get(section.get("length", "medium"), 8)
            # NOTE: previously `max(8, int(raw_bars * 1.5))` clamped "short"
            # (4*1.5=6) up to the same floor as "medium" (8*1.5=12 -> but the
            # floor was 8), making short/medium indistinguishable at the low
            # end. Floor lowered to 4 so "short" sections can actually come
            # out shorter than "medium"/"long". Raise this back to 8 if
            # ChordRealizer or the model's max_tokens sizing assumes a hard
            # minimum bar count.
            bars = max(4, int(raw_bars * 1.5))

            if name in mod_map:
                mod_intent = mod_map[name]
                target_key = mod_intent.get("target_key")
                target_mode = mod_intent.get("target_mode")

                if not target_key or not target_mode:
                    current_key_pc = NOTE_MAP.get(current_key, 0)
                    current_key, current_mode = self._get_relative_key(current_key_pc, current_mode)
                else:
                    current_key = target_key
                    current_mode = target_mode

                logger.info(f"🎼 Modulating to {current_key} {current_mode} for the {name}!")

            explicit_chords = section.get("explicit_chords") or []
            explicit_durations = section.get("explicit_chord_durations") or None
            if explicit_chords:
                progression = self._build_explicit_progression(explicit_chords, explicit_durations, bars, beats_per_bar)
                harmony_source = "explicit"
                logger.info(f"🎼 Section '{name}' using user-specified chord progression ({len(explicit_chords)} chords, "
                            f"{'custom' if explicit_durations else 'one-bar-each'} durations).")
            else:
                progression = self._generate_function_markov_chords(current_key, current_mode, name, bars, beats_per_bar)
                harmony_source = "generated"
            chord_timeline = self._build_chord_timeline(progression, bpm)

            explicit_melody = section.get("explicit_melody") or []
            melody_pattern = self._build_melody_pattern(explicit_melody, bars, beats_per_bar) if explicit_melody else []
            if melody_pattern:
                logger.info(f"🎵 Section '{name}' using user-specified melody ({len(explicit_melody)} tokens, "
                            f"tiled to {len(melody_pattern)} notes).")

            # Notes the user deliberately chose — explicit chord tones and/or
            # explicit melody pitches — bypass SoftRefiner's diatonic
            # snapping later. Algorithmically-generated chords are already
            # built from the declared scale by construction, so they never
            # needed this; explicit content might intentionally include
            # chromaticism (secondary dominants, borrowed chords, blue
            # notes, passing tones) that would otherwise get "corrected"
            # right back out of the piece.
            protected_pcs = set()
            if harmony_source == "explicit":
                for c in progression:
                    for iv in _chord_tones(c.get("quality", "")):
                        protected_pcs.add((c["root_pc"] + iv) % 12)
            for ev in melody_pattern:
                protected_pcs.add(ev["pitch_class"])

            # PREFER LLM'S DYNAMIC INSTRUMENTS IF PROVIDED.
            # FIX: both branches now COPY rather than alias the source list.
            # Before: `instruments = arrangement.get(name, ...)` handed back
            # the *same* list object for every section sharing a name (e.g.
            # two sections both named "Chorus"), so `.insert()` on one
            # mutated the other's target_instruments too. And
            # `instruments = llm_instruments` aliased the caller's own
            # llm_output dict, so `.insert()` silently mutated the caller's
            # original data out from under them.
            llm_instruments = section.get("instruments", [])
            if llm_instruments:
                instruments = list(llm_instruments)
            else:
                instruments = list(arrangement.get(name, ["Violin"]))

            if lead_inst not in instruments:
                instruments.insert(0, lead_inst)

            # SOLO SUPPORT: ChordRealizer already treats instruments[0] as
            # the "lead" role — so featuring a solo instrument is just a
            # matter of moving it to index 0 before ChordRealizer ever sees
            # the list. No changes needed there for role assignment; the
            # backing-thinning behavior during a solo is handled inside
            # ChordRealizer separately (triggered by solo_instrument being
            # present at all, not by list position).
            solo_instrument = section.get("solo_instrument")
            if solo_instrument and solo_instrument in instruments and instruments[0] != solo_instrument:
                instruments.remove(solo_instrument)
                instruments.insert(0, solo_instrument)
            elif solo_instrument and solo_instrument not in instruments:
                logger.warning(f"⚠️ [Planner] solo_instrument '{solo_instrument}' isn't in section "
                                f"'{name}''s instrument list {instruments} — ignoring.")
                solo_instrument = None

            ties = self._get_ties_weights(name, mood, instruments)

            density_curve = self._generate_density_curve(name, bars, energy_curve)

            section_entry = {
                "section_name": name, "bars": bars,
                "key": current_key, "mode": current_mode,
                "bpm": bpm, "style": style,
                # BUGFIX: mood and energy_curve were never actually stored
                # on section_entry before, despite being computed right
                # here — meaning AgenticComposer's `section.get("mood",
                # "calm")` (used for the emotion-quadrant conditioning
                # token AND for SoftRefiner's velocity bounds) was silently
                # defaulting to "calm" for every section regardless of what
                # this planner actually decided. Same story for
                # energy_curve, needed by the phrase-melody generator below.
                "mood": mood, "energy_curve": energy_curve,
                "riff_mode": bool(section.get("riff", False)),
                "solo_instrument": solo_instrument,
                # SYMPHONIC MODE: carried through so ChordRealizer, HardScorer,
                # and AgenticComposer all read the same meter instead of each
                # independently assuming 4/4.
                "time_signature": time_signature, "beats_per_bar": beats_per_bar,
                "progression": progression,
                "target_chords": [c["symbol"] for c in progression],
                "chord_timeline": chord_timeline,
                "target_instruments": instruments,
                "harmony_source": harmony_source,
                "melody_pattern": melody_pattern,
                "protected_pcs": sorted(protected_pcs),
                "ties_weights": ties,
                "density_curve": density_curve,
                # Placeholder — overwritten below once the real note_events
                # count is known. Kept as a floor so a ChordRealizer failure
                # (caught below) still leaves a sane, non-zero budget.
                "max_tokens": bars * 16 * max(1, len(instruments) // 2),
                "motif_action": "generate_and_save_seed" if name == "Intro" else "retrieve_seed"
            }

            # Note-level realization for forced octave/pitch_class/instrument
            # generation. Additive key only - nothing above this line changed shape.
            try:
                section_entry["note_events"] = self.chord_realizer.realize_section(section_entry)
                # BUGFIX: max_tokens was previously ONLY a heuristic guess
                # (bars * 16 * instrument-count-factor) made before this
                # section's actual notes were ever realized. If the real
                # forced note_event count exceeds that guess — increasingly
                # likely now that register-spread harmony and full-phrase
                # melody generation both produce more simultaneous notes
                # than the old per-chord tone-cycling did — generation runs
                # out of token budget before covering the section's full
                # intended musical length, and the output comes out shorter
                # than the prompt implied. Resizing from the REAL count
                # (now that we have it) with 15% headroom for the model's
                # own connective/free-sampled content fixes this at the
                # source instead of relying on the duration-rescale safety
                # net in AgenticComposer to paper over it after the fact.
                real_note_count = len(section_entry["note_events"])
                if real_note_count > 0:
                    section_entry["max_tokens"] = max(
                        section_entry["max_tokens"],
                        int(real_note_count * 1.15) + 16
                    )
            except Exception as e:
                logger.error(f"❌ ChordRealizer failed for section '{name}': {e}. "
                              f"Falling back to empty note_events (model will free-sample as before).")
                section_entry["note_events"] = []

            dense_timeline.append(section_entry)

        return {
            "global_settings": {
                "key": global_key, "mode": global_mode, "bpm": bpm, "style": style,
                "time_signature": time_signature,
            },
            "timeline": dense_timeline
        }