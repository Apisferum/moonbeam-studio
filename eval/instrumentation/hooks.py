import os
import time
import dataclasses
from typing import List, Dict, Any, Optional
from eval.instrumentation.trace_schema import PieceTrace, SectionTrace, AttemptTrace

class TraceContext:
    _instance: Optional['TraceContext'] = None

    @classmethod
    def get_instance(cls) -> 'TraceContext':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls, prompt: str = "", global_key: str = "C", global_mode: str = "major", global_bpm: float = 120.0, global_time_signature: str = "4/4"):
        cls._instance = cls(prompt, global_key, global_mode, global_bpm, global_time_signature)
        return cls._instance

    def __init__(self, prompt: str = "", global_key: str = "C", global_mode: str = "major", global_bpm: float = 120.0, global_time_signature: str = "4/4"):
        self.trace = PieceTrace(
            prompt=prompt,
            global_key=global_key,
            global_mode=global_mode,
            global_bpm=global_bpm,
            global_time_signature=global_time_signature
        )
        self.current_section: Optional[SectionTrace] = None
        self.section_start_time: float = 0.0

    def start_section(self, section_idx: int, section_type: str, blueprint: Dict[str, Any]):
        self.section_start_time = time.time()
        self.current_section = SectionTrace(
            section_idx=section_idx,
            section_type=section_type,
            blueprint_chords=blueprint.get("target_chords", []),
            blueprint_rhythm_template=blueprint.get("density_curve", []),
            blueprint_emotion_label=blueprint.get("mood", "calm"),
            blueprint_instrumentation=blueprint.get("target_instruments", []),
            retrieved_motifs=[],
            ties_weights=blueprint.get("ties_weights", {}),
            attempts=[],
            final_midi_path="",
            final_accept_attempt=-1,
            latency_ms=0.0,
            kv_cache_reused=False
        )

    def log_motif(self, name: str, similarity: float, idx: int):
        if self.current_section:
            self.current_section.retrieved_motifs.append({
                "name": name,
                "similarity": similarity,
                "index": idx
            })

    def log_ties_weights(self, weights: Dict[str, float]):
        if self.current_section:
            self.current_section.ties_weights = weights.copy()

    def log_attempt(self, attempt_idx: int, temp: float, score: float, feedback: Dict[str, Any], accepted: bool, token_count: int):
        if self.current_section:
            metrics = feedback.get("metrics", {})
            attempt = AttemptTrace(
                attempt_idx=attempt_idx,
                score=score,
                feedback=feedback,
                accepted=accepted,
                temperature=temp,
                token_count=token_count,
                chord_match=metrics.get("chord_score"),
                voice_leading_score=metrics.get("voice_leading_score"),
                rhythm_match=metrics.get("rhythm_score"),
                inst_match=metrics.get("inst_score")
            )
            self.current_section.attempts.append(dataclasses.asdict(attempt))

    def end_section(self, final_midi_path: str, final_accept_attempt: int, kv_cache_reused: bool = False):
        if self.current_section:
            self.current_section.final_midi_path = final_midi_path
            self.current_section.final_accept_attempt = final_accept_attempt
            self.current_section.kv_cache_reused = kv_cache_reused
            self.current_section.latency_ms = (time.time() - self.section_start_time) * 1000.0
            self.trace.sections.append(self.current_section)
            self.current_section = None

    def save_trace(self, filepath: str):
        self.trace.save(filepath)

# Hook functions for public usage
def hook_initialize_piece(prompt: str, global_key: str, global_mode: str, global_bpm: float, global_time_signature: str):
    TraceContext.reset(prompt, global_key, global_mode, global_bpm, global_time_signature)

def hook_start_section(section_idx: int, section_type: str, blueprint: Dict[str, Any]):
    TraceContext.get_instance().start_section(section_idx, section_type, blueprint)

def hook_log_motif(name: str, similarity: float, idx: int):
    TraceContext.get_instance().log_motif(name, similarity, idx)

def hook_log_ties_weights(weights: Dict[str, float]):
    TraceContext.get_instance().log_ties_weights(weights)

def hook_log_attempt(attempt_idx: int, temp: float, score: float, feedback: Dict[str, Any], accepted: bool, token_count: int):
    TraceContext.get_instance().log_attempt(attempt_idx, temp, score, feedback, accepted, token_count)

def hook_end_section(final_midi_path: str, final_accept_attempt: int, kv_cache_reused: bool = False):
    TraceContext.get_instance().end_section(final_midi_path, final_accept_attempt, kv_cache_reused)

def hook_save_piece_trace(filepath: str):
    TraceContext.get_instance().save_trace(filepath)
