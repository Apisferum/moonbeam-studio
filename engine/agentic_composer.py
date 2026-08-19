import os
import logging
import torch
import gc
import io
import copy
import traceback
import pretty_midi
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

from brain.llm_wrapper import LLMWrapper
from brain.structure_planner import StructurePlanner
from critic.hard_scorer import HardScorer
from critic.midi_polisher import MIDIPolisher
from critic.soft_refiner import SoftRefiner
from engine.motif_memory import MotifMemoryFAISS
from shared.music_theory_constants import logger, TICKS_PER_SECOND

try:
    from eval.instrumentation.hooks import (
        hook_initialize_piece, hook_start_section,
        hook_log_ties_weights, hook_log_attempt, hook_end_section
    )
except ImportError:
    def hook_initialize_piece(*args, **kwargs): pass
    def hook_start_section(*args, **kwargs): pass
    def hook_log_ties_weights(*args, **kwargs): pass
    def hook_log_attempt(*args, **kwargs): pass
    def hook_end_section(*args, **kwargs): pass

def _cuda_available() -> bool:
    try: return torch.cuda.is_available()
    except Exception: return False

class AgenticComposer:
    def __init__(self, harmonyrouter, acceptance_threshold: float = 0.55,
                 max_primer_tokens: Optional[int] = None,
                 use_planner: bool = True,
                 use_motif_memory: bool = True,
                 use_ties: bool = True,
                 use_soft_refiner: bool = True,
                 use_hard_scorer: bool = True):
        """
        max_primer_tokens: hard cap on how many compound tokens from a seed
        MIDI's primer are ever handed to HarmonyRouter.generate(). None
        (default) preserves the old behavior — the ENTIRE primer, however
        long, gets fed in with no truncation. That's fine for short seeds
        but has no safety net for long ones: nothing here or in
        HarmonyRouter currently checks primer length against the model's
        actual context window before calling music_completion(). On Kaggle
        T4x16GB you have real headroom over a 6GB card, but you still don't
        have infinite context — set this explicitly once you know your
        model's max sequence length, e.g.:

            composer = AgenticComposer(
                router,
                max_primer_tokens=int(router.config.max_position_embeddings * 0.5)
            )

        (leaving the other half of context for the generation itself). When
        the primer is longer than this cap, only the MOST RECENT tokens are
        kept — i.e. continuation is anchored to the end of the seed clip,
        which is usually what "continue this" means anyway.
        """
        self.harmonyrouter = harmonyrouter
        self.tokenizer = harmonyrouter.tokenizer
        self.master_dict = getattr(harmonyrouter, 'master_dict', {})
        self.llm = LLMWrapper()
        
        self.use_planner = use_planner
        self.use_motif_memory = use_motif_memory
        self.use_ties = use_ties
        self.use_soft_refiner = use_soft_refiner
        self.use_hard_scorer = use_hard_scorer

        # BUGFIX: StructurePlanner (and the ChordRealizer it owns) was
        # being constructed with ZERO arguments, meaning it silently used
        # its own class defaults (octave_vocab_size=9, pitch_class_vocab_
        # size=12, instrument_vocab_size=130) regardless of what the
        # ACTUAL loaded model's config specifies. StructurePlanner's own
        # docstring already said this needed to happen — it just never
        # did, at the only place that constructs it. A mismatch here means
        # forced octave/pitch_class tokens could reference indices that
        # don't mean what we think they mean in the real model's
        # vocabulary — a plausible contributor to the "data byte must be
        # in range 0..127" crash and the wildly unstable generation
        # lengths in the log (some attempts near-total failures). Worth
        # confirming directly: print harmonyrouter.config.octave_vocab_size
        # / .pitch_class_vocab_size / .instrument_vocab_size and compare
        # against 9/12/130 — if they differ, this was silently wrong the
        # entire time. getattr(..., default) guards against the config
        # object not having one of these attributes under a slightly
        # different name.
        self.planner = StructurePlanner(
            octave_vocab_size=getattr(harmonyrouter.config, "octave_vocab_size", 9),
            pitch_class_vocab_size=getattr(harmonyrouter.config, "pitch_class_vocab_size", 12),
            instrument_vocab_size=getattr(harmonyrouter.config, "instrument_vocab_size", 130),
        )
        self.scorer = HardScorer()
        self.polisher = MIDIPolisher()
        self.refiner = SoftRefiner()
        self.motif_memory = MotifMemoryFAISS()
        self.acceptance_threshold = acceptance_threshold
        self.max_attempts = 3
        self.max_primer_tokens = max_primer_tokens

    def _map_mood_to_emo_token(self, mood: str) -> int:
        quadrant_map = {
            "happy": "emo_q1", "heroic": "emo_q1", "excited": "emo_q1",
            "tension": "emo_q2", "angry": "emo_q2", "dark": "emo_q2",
            "sad": "emo_q3", "melancholy": "emo_q3", "grief": "emo_q3",
            "calm": "emo_q4", "peaceful": "emo_q4", "tender": "emo_q4"
        }
        q_token = quadrant_map.get(mood.lower(), "emo_q4")
        for key, val in self.master_dict.items():
            if q_token in key.lower(): return val
        return -4

    def _build_forced_streams(self, note_events: List[Dict], bpm: float = 120, primer_offset_ticks: int = 0) -> List[deque]:
        """
        TIMING FORCING: previously this only forced octave/pitch_class/
        instrument — the note's WHICH and its position in sequence, never
        WHEN. Onset timing and duration were always left to the model's own
        free sampling, which is why forced content could still land far
        from its intended beat position (a major contributor to the
        chord/timing mismatches seen across many generation runs). Now each
        queued item also carries target_tick (this event's intended
        absolute onset, in the tokenizer's TICKS_PER_SECOND=100 real-time
        convention) and target_duration_ticks (its intended length in the
        same units), computed from start_beat/duration_beats using the
        section's actual bpm. MusicLlama.generate() uses these to compute a
        dynamic timeshift delta against its own running onset position —
        see generation.py for why this can't be a precomputed static value.

        BUGFIX (primer_offset_ticks): target_tick used to be computed as
        purely section-local (0-based) regardless of whether a primer was
        present. But when continuing from a primer, the model's own
        running onset (tokens[:, cur_pos-1, 0] inside generate()) starts at
        the PRIMER's last absolute onset, not 0 — the primer's tokens are
        loaded into the sequence verbatim, unshifted. With a section-local
        target_tick compared against a primer-scale running onset, the
        computed delta was negative for the ENTIRE section whenever the
        primer's last onset was already comparable to (or larger than) the
        new section's own max_section_ticks — confirmed directly from a
        real run: primer_last_onset=4589 against max_section_ticks=4658,
        meaning literally every forced note in that section failed its
        timing check and the model's free-run onset just kept climbing on
        the primer's original scale while pitch/octave/instrument kept
        forcing anyway, landing every note far outside the section's
        window. Passing the primer's own last onset here and adding it to
        every target_tick puts them back on the same scale the model is
        actually continuing from.
        If an event has no duration_beats (shouldn't happen now that every
        ChordRealizer emission path sets one, but defensive regardless),
        target_duration_ticks falls back to a quarter-beat's worth of ticks
        rather than being omitted, since MusicLlama.generate() gracefully
        skips forcing when a key is absent — better to force a sane default.
        """
        beat_duration_ticks = (60.0 / bpm) * TICKS_PER_SECOND
        stream = deque()
        for event in note_events:
            target_tick = primer_offset_ticks + int(round(event.get('start_beat', 0) * beat_duration_ticks))
            duration_beats = event.get('duration_beats', 0.25)
            target_duration_ticks = max(1, int(round(duration_beats * beat_duration_ticks)))
            stream.append({
                "target_duration_ticks": target_duration_ticks,
            })
        return [stream]

    def _generate_and_score(self, section: Dict[str, Any], section_name: str,
                             metadata_ids: List[int], primer_tokens, temperature: float
                             ) -> Optional[Tuple[pretty_midi.PrettyMIDI, float, dict, list]]:
        try:
            note_events = section.get("note_events", [])
            # Computed here (not just later, at the post-hoc shift step) so
            # it can also be used to offset target_tick BEFORE generation —
            # see the BUGFIX note in _build_forced_streams above.
            primer_offset_ticks = 0
            if primer_tokens:
                six_d_primer_toks_for_offset = [t for t in primer_tokens if len(t) == 6]
                if six_d_primer_toks_for_offset:
                    primer_offset_ticks = six_d_primer_toks_for_offset[-1][0]
            forced_streams = self._build_forced_streams(
                note_events, bpm=section.get("bpm", 120), primer_offset_ticks=primer_offset_ticks
            ) if note_events else None

            # HARD CEILING CHECK: total_len = min(config.max_len, max_gen_len
            # + prompt_len) inside generate() — config.max_len (e.g. 1024)
            # caps PROMPT + GENERATED note-positions COMBINED, regardless of
            # what max_tokens requests. No amount of budget tuning on our
            # side can exceed this; it's architectural. Two things worth
            # knowing separately: (1) whether OUR requested budget alone
            # already exceeds it (a max_tokens sizing problem we could fix),
            # and (2) whether the section's actual forced note_events count
            # exceeds it BY ITSELF (a problem no budget can fix — the
            # section would need to be split into multiple chunks internally
            # to ever fully realize its intended length; not implemented
            # here, flagging for visibility so it isn't a silent mystery).
            config_max_len = getattr(getattr(self.harmonyrouter, "config", None), "max_len", None)
            requested_len = section.get("max_tokens", 256)
            primer_len = len(primer_tokens) if primer_tokens else 1
            if config_max_len is not None:
                if requested_len + primer_len > config_max_len:
                    logger.warning(
                        f"⚠️ [MaxLen] '{section_name}' requested max_tokens={requested_len} "
                        f"(+{primer_len} primer) = {requested_len + primer_len}, exceeding this "
                        f"model's max_len={config_max_len}. Generation WILL be silently clamped down "
                        f"to {max(0, config_max_len - primer_len)} note-positions regardless of budget."
                    )
                if note_events and len(note_events) > config_max_len:
                    logger.warning(
                        f"⚠️ [MaxLen] '{section_name}' needs {len(note_events)} forced note events, which "
                        f"ALONE exceeds max_len={config_max_len} — this section cannot fully realize its "
                        f"intended length no matter what max_tokens is set to. Reducing instrument/voice "
                        f"count for this section, or splitting it into smaller chunks, are the only real "
                        f"fixes; the budget itself isn't the bottleneck here."
                    )

            # DIAGNOSTIC: forced_streams[b] deques are mutated in-place by
            # generate() (via .popleft() each time an instrument gets
            # forced) — so comparing length before/after the call tells us
            # something the timing-offset fix alone can't: whether the
            # model consumed our WHOLE forced queue (remaining==0, meaning
            # content forcing worked and any runaway timing is in
            # free-generation AFTER the queue was exhausted) or stopped
            # early, before even using all of it (remaining>0, meaning
            # something is cutting generation off — plausibly related to
            # the separate "No notes generated" failures also showing up
            # on primer-continued sections, which the offset fix wasn't            initial_queue_len = len(forced_streams[0]) if forced_streams else 0
            all_generated_tokens = []
            current_primer = primer_tokens
            chunk_count = 0
            max_chunks = 3

            while True:
                chunk_count += 1
                logger.info(
                    f"   ↳ [SlidingWindow] Generating chunk {chunk_count} for '{section_name}' "
                    f"(queue size: {len(forced_streams[0]) if forced_streams else 0})"
                )

                raw_tokens = self.harmonyrouter.generate(
                    metadata_ids=metadata_ids, primer_tokens=current_primer,
                    max_gen_len=section.get("max_tokens", 256), temperature=temperature, top_p=0.9,
                    bpm=section.get("bpm", 120), num_measures=section.get("bars", 8),
                    forced_token_streams=forced_streams
                )

                safe_tokens = raw_tokens.tolist() if hasattr(raw_tokens, 'tolist') else raw_tokens

                # 🚀 MERGE GROUPED TOKENS (workaround for instrument splitting)
                if (safe_tokens and isinstance(safe_tokens, list) and len(safe_tokens) > 0
                        and isinstance(safe_tokens[0], list) and len(safe_tokens[0]) > 0
                        and isinstance(safe_tokens[0][0], list)):
                    merged_tokens = []
                    for group in safe_tokens:
                        merged_tokens.extend(group)
                    safe_tokens = merged_tokens

                if not safe_tokens or len(safe_tokens) == 0:
                    logger.warning(f"   ↳ [SlidingWindow] Chunk {chunk_count} generated empty tokens. Stopping chunk loop.")
                    break

                all_generated_tokens.extend(safe_tokens)

                # Check if there are still notes remaining in the forced queue
                remaining_in_queue = len(forced_streams[0]) if forced_streams else 0
                if remaining_in_queue == 0:
                    logger.info("   ↳ [SlidingWindow] All forced note events consumed. Stopping chunk loop.")
                    break
                if chunk_count >= max_chunks:
                    logger.warning(f"   ↳ [SlidingWindow] Reached maximum chunk limit of {max_chunks}. Stopping chunk loop.")
                    break

                # Prepare the primer for the next chunk:
                # The running context is the original primer plus the generated tokens so far.
                running_context = (primer_tokens or []) + all_generated_tokens
                # Keep only the last 255 tokens as the new primer, prepending the SOS token
                sos_token = self.tokenizer.sos_token_compound
                current_primer = [sos_token] + running_context[-255:]

            safe_tokens = all_generated_tokens

            if forced_streams:
                remaining_queue_len = len(forced_streams[0])
                logger.info(
                    f"   ↳ [ForceQueue] '{section_name}': {initial_queue_len} notes queued, "
                    f"{initial_queue_len - remaining_queue_len} consumed, {remaining_queue_len} left unused."
                )

            if not safe_tokens or len(safe_tokens) == 0:
                raise ValueError("Model generated empty tokens.")

            # 🚀 CRITICAL ARCHITECTURAL FIX: SHIFT (strip removed — see below)
            prompt_len = len(primer_tokens) if primer_tokens else 1

            # BUGFIX (double-strip): this used to do
            # `safe_tokens = safe_tokens[prompt_len:]` here — but
            # MusicLlama.generate() ALREADY strips the primer/condition
            # tokens before ever returning, via condition_token_lengths:
            #     out_tokens_no_cond_tokens.append(out_tokens[i][condition_token_length:])
            # (condition_token_length is exactly len(primer_tokens), the
            # same value as `prompt_len` here). So safe_tokens arriving at
            # this point is ALREADY the generated-only portion — slicing
            # off prompt_len AGAIN was removing prompt_len tokens from the
            # FRONT of the real new generation a second time. Two
            # consequences, confirmed directly from real runs:
            # (1) Whenever max_gen_len < prompt_len (small sections with a
            #     large FAISS primer — exactly 'Coda' every single time,
            #     deterministically, since this is pure arithmetic, not
            #     stochastic generation): the second strip empties the list
            #     completely regardless of what was actually generated.
            # (2) Whenever max_gen_len > prompt_len (the "successful" cases):
            #     the first prompt_len tokens of the ACTUAL new generation —
            #     often containing the section's opening chord/downbeat
            #     content — were silently discarded every time, which is a
            #     very plausible contributor to the chord/timing scores
            #     staying mediocre across primer-continued sections all
            #     through this conversation, even after every generation-
            #     side timing fix.
            # The empty-check below is kept as a genuine sanity check (the
            # model can still legitimately generate nothing), just no
            # longer following a redundant, destructive slice.
            if not safe_tokens or len(safe_tokens) == 0:
                raise ValueError(
                    f"Model generated no new tokens for this attempt (prompt_len={prompt_len}) — "
                    f"nothing left to convert to MIDI."
                )

            # 3. Shift the continuation's absolute onsets down to local section time (0-indexed)
            # Reuses primer_offset_ticks computed earlier (now that it's
            # also needed before generation, for target_tick) instead of
            # recomputing the same value from primer_tokens a second time.
            primer_last_onset = primer_offset_ticks if primer_tokens else None  # kept for the diagnostic error message below
            if primer_tokens is not None and primer_offset_ticks:
                for tok in safe_tokens:
                    if len(tok) == 6:
                        tok[0] = max(0, tok[0] - primer_offset_ticks)

            # 🚀 INJECT SOFT REFINER (The Token-Level Gatekeeper)
            bpm = section.get("bpm", 120)
            bars = section.get("bars", 8)
            # SYMPHONIC MODE: was hardcoded `bars * 4`, assuming 4/4. Now
            # reads beats_per_bar off the section (set by StructurePlanner
            # from global_time_signature).
            beats_per_bar = section.get("beats_per_bar", 4) or 4
            seconds = (bars * beats_per_bar) / (bpm / 60.0)
            max_ticks = int(seconds * TICKS_PER_SECOND * 1.10)

            # DIAGNOSTIC: captured before refine_tokens can drop anything, so
            # if it drops everything, the exception below can report the
            # actual abs_time range instead of just the bare fact that it
            # happened. This has recurred specifically on primer-continued
            # sections (never on the very first, primer-less section) —
            # the goal here is ground truth on the next run rather than
            # another guess at the cause.
            pre_refine_abs_times = [t[0] for t in safe_tokens if len(t) == 6]

            if self.use_soft_refiner:
                safe_tokens = self.refiner.refine_tokens(
                    safe_tokens, root_note=section.get("key", "C"), mode=section.get("mode", "major"),
                    max_section_ticks=max_ticks, mood=section.get("mood", "calm"),
                    ties_weights=section.get("ties_weights"),
                    protected_pcs=set(section.get("protected_pcs", []))
                )

                # BUGFIX: the previous empty-token check (right after stripping
                # the primer) only caught the model generating literally zero
                # new tokens. It DIDN'T catch this: refine_tokens legitimately
                # drops every token whose abs_time >= max_section_ticks (see
                # SoftRefiner's `dropped_notes` counter). If the primer-onset
                # shift a few lines up left every generated token's onset past
                # the section boundary — plausible with a long/self-similar
                # FAISS primer, which is exactly what happened here (Cosine Sim
                # 1.00) — refine_tokens can legitimately return an EMPTY list,
                # and nothing re-checked for that before compound_to_midi([])
                # produced a MIDI file with zero note events, which THEN
                # crashed three calls later inside pretty_midi's own
                # constructor ("max() iterable argument is empty") — the same
                # symptom as before, but reached through refine_tokens instead
                # of the strip. Checking again here, right after the step that
                # can actually cause it this time.
                if not safe_tokens or len(safe_tokens) == 0:
                    abs_range = f"min={min(pre_refine_abs_times)} max={max(pre_refine_abs_times)}" if pre_refine_abs_times else "no 6-tuple tokens at all"
                    raise ValueError(
                        f"SoftRefiner dropped every token for this attempt (all onsets fell past "
                        f"max_section_ticks={max_ticks}) — nothing left to convert to MIDI. "
                        f"Pre-refine abs_time range: {abs_range}. primer_last_onset={primer_last_onset}. "
                        f"primer_tokens length={len(primer_tokens) if primer_tokens else 0}."
                    )

            # CORRECTION: compound_to_midi does not accept a bpm/tempo
            # parameter — confirmed from MusicTokenizer's actual source.
            # It doesn't need one: `abs_time` ticks are already pure
            # real-time units (TIME_RESOLUTION=100 ticks/second), and
            # ticks_per_beat is hardcoded to 50 specifically so that MIDI's
            # own default tempo (120 BPM, applied when no set_tempo
            # message is present) reconstructs the correct real seconds —
            # 50 ticks/beat * 2 beats/sec at 120 BPM = 100 ticks/sec. This
            # is self-consistent regardless of the section's actual
            # musical bpm; nothing to fix here.
            raw_midi_mido = self.tokenizer.compound_to_midi(safe_tokens)
            buffer = io.BytesIO()
            raw_midi_mido.save(file=buffer)
            buffer.seek(0)
            raw_midi = pretty_midi.PrettyMIDI(buffer)

            # DURATION SELF-CHECK: `seconds` above is the section's true
            # intended duration. If the generated MIDI falls meaningfully
            # short of it, the most likely cause isn't a tempo mismatch
            # (ruled out above) — it's max_tokens/max_gen_len running out
            # before the model finishes covering the section, which
            # StructurePlanner now sizes from the actual forced note_event
            # count rather than a rough bars/instrument-count guess (see
            # structure_planner.py). This rescale is a safety net for
            # whatever residual gap remains, not the primary fix.
            actual_end = raw_midi.get_end_time()
            if actual_end > 0.05:
                ratio = seconds / actual_end
                if abs(ratio - 1.0) > 0.15:
                    logger.warning(
                        f"⚠️ [Timing] '{section_name}' generated at {actual_end:.2f}s but should span "
                        f"{seconds:.2f}s at {bpm} BPM ({abs(ratio - 1) * 100:.0f}% off) — rescaling note "
                        f"times to correct. If this fires often, max_tokens is likely still undersized "
                        f"for this section's actual note density."
                    )
                    for inst in raw_midi.instruments:
                        for note in inst.notes:
                            note.start *= ratio
                            note.end *= ratio

            polished_midi = self.polisher.polish(
                raw_midi, bpm=section["bpm"], chord_timeline=section["chord_timeline"],
                ties_weights=section.get("ties_weights"),
                density_curve=section.get("density_curve"), mood=section.get("mood", "calm"),
                bars=bars, beats_per_bar=beats_per_bar
            )

            if self.use_hard_scorer:
                score, feedback = self.scorer.score(polished_midi, section, section_name=section_name)
            else:
                score = 1.0
                feedback = {"feedback": "HardScorer disabled", "metrics": {}}
            return polished_midi, score, feedback, safe_tokens

        except Exception as e:
            # Was just str(e) — enough to know THAT something failed, not
            # WHERE. "max() iterable argument is empty" / "data byte must
            # be in range 0..127" style errors are unpinnable without a
            # traceback; this costs nothing and turns the next one of
            # these into an actual fixable bug report instead of a guess.
            logger.error(f"   ⚠️ Generation attempt failed for '{section_name}': {e}")
            logger.error(traceback.format_exc())
            if _cuda_available(): torch.cuda.empty_cache()
            gc.collect()
            return None

    def compose_full_song(self, timeline: List[Dict[str, Any]],
                          primer_midi_path: Optional[str] = None,
                          prompt: Optional[str] = None) -> pretty_midi.PrettyMIDI:

        # Reset motif memory at the start of each new song generation to avoid memory leakage and context contamination
        self.motif_memory.clear()

        primer_tokens = None
        primer_midi_obj = None
        primer_duration_sec = 0.0

        if primer_midi_path:
            if not os.path.exists(primer_midi_path):
                logger.error(f"❌ Primer MIDI not found: {primer_midi_path}. Proceeding without it.")
            else:
                primer_midi_obj = pretty_midi.PrettyMIDI(primer_midi_path)
                primer_duration_sec = primer_midi_obj.get_end_time()
                raw_primer_tokens = self.tokenizer.midi_to_compound(primer_midi_path)
                primer_tokens = [self.tokenizer.sos_token_compound] + raw_primer_tokens
                logger.info(f"🎧 [Primer] Loaded {primer_duration_sec:.2f}s seed MIDI ({len(primer_tokens)} tokens).")

                # IMPROVEMENT: opt-in tail-windowing. Previously the ENTIRE
                # primer — however long — was handed to generate() with no
                # length check against the model's context window. A long
                # seed clip (several minutes) could silently misbehave or
                # exceed context. When max_primer_tokens is set, keep only
                # the most recent tokens (continuation anchors to the END
                # of the seed, which is what "continue this" usually means)
                # while always preserving the SOS token at index 0.
                if self.max_primer_tokens and len(primer_tokens) > self.max_primer_tokens:
                    sos = primer_tokens[0]
                    tail = primer_tokens[-(self.max_primer_tokens - 1):]
                    original_len = len(primer_tokens)
                    primer_tokens = [sos] + tail
                    logger.warning(
                        f"⚠️ [Primer] Truncated from {original_len} to {len(primer_tokens)} tokens "
                        f"(max_primer_tokens={self.max_primer_tokens}) to stay within model context. "
                        f"The earliest material in this seed clip will not influence generation — "
                        f"only its tail does. Raise max_primer_tokens if your context window allows more."
                    )

        # BUGFIX: was `pretty_midi.PrettyMIDI()` with no arguments, which
        # defaults initial_tempo=120 regardless of what tempo the song was
        # actually composed at — cosmetic on its own (notes already carry
        # correct absolute-second times), but wrong/misleading if anything
        # downstream (a DAW, a tempo-aware renderer) reads this file's own
        # declared tempo instead of just trusting raw seconds.
        song_bpm = timeline[0].get("bpm", 120) if timeline else 120
        
        global_key = timeline[0].get("key", "C") if timeline else "C"
        global_mode = timeline[0].get("mode", "major") if timeline else "major"
        global_ts = timeline[0].get("time_signature", "4/4") if timeline else "4/4"
        hook_initialize_piece(
            prompt or getattr(self, "current_prompt", "Unknown Prompt"),
            global_key,
            global_mode,
            song_bpm,
            global_ts
        )

        final_midi = pretty_midi.PrettyMIDI(initial_tempo=song_bpm)
        current_time_offset = 0.0
        last_accepted_midi = None

        for section_idx, section in enumerate(timeline):
            section_name = section["section_name"]
            logger.info(f"🎼 Composing Section: {section_name}...")

            hook_start_section(section_idx, section_name, section)

            metadata_ids = []
            mood = section.get("mood", "calm")
            metadata_ids.append(self._map_mood_to_emo_token(mood))
            while len(metadata_ids) < 11: metadata_ids.append(-4)

            active_primer = None
            if section_idx == 0 and primer_tokens is not None:
                active_primer = primer_tokens
                logger.info("   ↳ Injecting user primer for outpainting continuation.")
            elif last_accepted_midi is not None and self.use_motif_memory:
                active_primer = self.motif_memory.retrieve_primer(last_accepted_midi)
                if active_primer: logger.info("   ↳ Injecting FAISS Primer.")

            best_midi = None
            best_score = -1.0
            best_feedback = {}
            current_temp = 0.85
            accepted = False

            # Initialize weights for this section (Bugfix: start fresh for each section)
            weights = section.get("ties_weights", {"commu_lora": 0.35, "emopia_lora": 0.25, "slakh_lora": 0.40}).copy()
            self.harmonyrouter.set_weights(weights)

            for attempt in range(1, self.max_attempts + 1):
                hook_log_ties_weights(weights)
                result = self._generate_and_score(
                    section, section_name, metadata_ids, active_primer, current_temp
                )

                if result is None:
                    hook_log_attempt(attempt, current_temp, 0.0, {"metrics": {}, "feedback": "Attempt crashed/failed"}, False, 0)
                    continue

                polished_midi, score, feedback, gen_tokens = result
                logger.info(f"   ↳ Attempt {attempt} | Score: {score:.2f} | {feedback.get('feedback', '')}")

                token_count = len(gen_tokens) if gen_tokens else 0
                hook_log_attempt(attempt, current_temp, score, feedback, score >= self.acceptance_threshold, token_count)

                if score >= self.acceptance_threshold:
                    best_midi = polished_midi
                    best_score = score
                    last_accepted_midi = polished_midi
                    if self.use_motif_memory:
                        self.motif_memory.save_section(polished_midi, section, tokens=gen_tokens)
                    accepted = True
                    break
                else:
                    if score > best_score: best_score = score; best_midi = polished_midi; best_feedback = feedback
                    metrics = feedback.get("metrics", {})
                    if self.use_ties:
                        weights = section.get("ties_weights", {"commu_lora": 0.35, "emopia_lora": 0.25, "slakh_lora": 0.40}).copy()
                        # IMPROVEMENT: this was an elif chain, so only the
                        # single highest-priority weak metric got corrected per
                        # retry even when several were bad at once — e.g. a
                        # section with both a low inst_score AND a low
                        # chord_score would only ever get the inst_score fix,
                        # leaving the chord problem unaddressed for the whole
                        # retry budget. Each weak metric now gets its own
                        # independent nudge, and temperature reduction can
                        # happen alongside the weight adjustments rather than
                        # only when nothing else was already flagged.
                        if metrics.get("inst_score", 1.0) < 0.75:
                            weights["slakh_lora"] = weights.get("slakh_lora", 0.0) + 0.15
                        if metrics.get("chord_score", 1.0) < 0.80:
                            weights["commu_lora"] = weights.get("commu_lora", 0.0) + 0.15
                        if metrics.get("voice_leading_score", 1.0) < 0.70:
                            weights["emopia_lora"] = weights.get("emopia_lora", 0.0) + 0.15
                        total = sum(weights.values())
                        if total > 0: weights = {k: v / total for k, v in weights.items()}
                        self.harmonyrouter.set_weights(weights)
                    
                    if metrics.get("rhythm_score", 1.0) < 0.70:
                        current_temp = max(0.4, current_temp - 0.15)

            if best_midi is None:
                logger.error(f"⚠️ Section '{section_name}' failed. Skipping.")
                hook_end_section(
                    final_midi_path="",
                    final_accept_attempt=-1,
                    kv_cache_reused=False
                )
                current_time_offset += 2.0
                continue

            # BUGFIX: previously, a section that never reached
            # acceptance_threshold in max_attempts tries still got appended
            # to final_midi below with NO warning and NO update to
            # last_accepted_midi — meaning the below-threshold section
            # shipped in the output silently, AND the next section's FAISS
            # primer retrieval would skip straight past it to whatever
            # earlier section actually succeeded, as if this one never
            # happened. Now it's logged explicitly, and last_accepted_midi
            # is still updated so downstream sections stay anchored to what
            # actually plays immediately before them.
            if not accepted:
                logger.warning(
                    f"⚠️ Section '{section_name}' never reached acceptance_threshold "
                    f"({self.acceptance_threshold}) in {self.max_attempts} attempts — "
                    f"using best-of-{self.max_attempts} (score={best_score:.2f}) instead."
                )
                last_accepted_midi = best_midi

            for inst in best_midi.instruments:
                new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
                for note in inst.notes:
                    new_inst.notes.append(pretty_midi.Note(
                        velocity=note.velocity, pitch=note.pitch,
                        start=note.start + current_time_offset, end=note.end + current_time_offset,
                    ))
                final_midi.instruments.append(new_inst)

            actual_end_time = 0.0
            for inst in best_midi.instruments:
                if inst.notes:
                    actual_end_time = max(actual_end_time, max(n.end for n in inst.notes))

            # BUGFIX (schedule-blowout cap): a single stray note with an
            # anomalously large end time — a residual artifact even after
            # every generation-side timing fix, e.g. from the model's
            # completely unconstrained free-generation tail once the
            # forced queue empties — used to blow up actual_end_time
            # arbitrarily. Since current_time_offset accumulates across
            # EVERY section, one bad outlier note in one section pushes
            # every SUBSEQUENT section's start time forward by the same
            # huge amount. This is exactly what a real assembled song
            # showed: large stretches of near-total silence between
            # sections, with a later section's content all crammed into a
            # few seconds far later than it should have started. Capping
            # actual_end_time at a generous multiple of the section's own
            # INTENDED duration (already known precisely) bounds how much
            # damage one outlier note can do to the rest of the song's
            # schedule, without discarding any of the section's own notes
            # — this only affects the OFFSET calculation for what comes
            # after, not the notes actually written into final_midi above.
            bpm = section.get("bpm", 120)
            bars = section.get("bars", 8)
            beats_per_bar = section.get("beats_per_bar", 4) or 4
            intended_seconds = (bars * beats_per_bar) / (bpm / 60.0)
            max_reasonable_end = max(intended_seconds * 2.0, 5.0)
            if actual_end_time > max_reasonable_end:
                logger.warning(
                    f"⚠️ [Assembly] '{section_name}' actual_end_time={actual_end_time:.2f}s far exceeds "
                    f"its intended duration ({intended_seconds:.2f}s) — capping to {max_reasonable_end:.2f}s "
                    f"so this section's own timing drift doesn't push every later section's schedule "
                    f"forward by the same huge amount. The section's actual notes are unaffected; only "
                    f"where the NEXT section starts is."
                )
            current_time_offset += actual_end_time + 0.2

            # Clean up TIES cache, GPU VRAM, and trigger garbage collection between section runs
            self.harmonyrouter.clear_ties_cache()
            if _cuda_available():
                torch.cuda.empty_cache()
            gc.collect()

            final_attempt_idx = attempt if accepted else -1
            hook_end_section(
                final_midi_path=f"section_{section_idx}.mid",
                final_accept_attempt=final_attempt_idx,
                kv_cache_reused=False
            )

        if primer_midi_obj is not None:
            logger.info("🧵 Stitching user primer with generated continuation...")
            for inst in final_midi.instruments:
                for note in inst.notes: note.start += primer_duration_sec; note.end += primer_duration_sec
            for inst in primer_midi_obj.instruments: final_midi.instruments.append(inst)

        return final_midi