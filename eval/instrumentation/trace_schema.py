from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json

@dataclass
class AttemptTrace:
    attempt_idx: int
    score: float
    feedback: Dict[str, Any]
    accepted: bool
    temperature: float
    token_count: int
    # Metrics extracted by the hard scorer
    chord_match: Optional[float] = None
    voice_leading_score: Optional[float] = None
    rhythm_match: Optional[float] = None
    inst_match: Optional[float] = None

@dataclass
class SectionTrace:
    section_idx: int
    section_type: str                  # verse/chorus/bridge
    blueprint_chords: List[str]        # planned chords
    blueprint_rhythm_template: List[float] # density curve or similar
    blueprint_emotion_label: str
    blueprint_instrumentation: List[str]
    retrieved_motifs: List[Dict[str, Any]] # motif info + similarity scores
    ties_weights: Dict[str, float]      # adapter weights choices
    attempts: List[Dict[str, Any]]      # list of AttemptTrace as dicts
    final_midi_path: str
    final_accept_attempt: int           # index of final accepted attempt
    latency_ms: float
    kv_cache_reused: bool

@dataclass
class PieceTrace:
    prompt: str
    global_key: str
    global_mode: str
    global_bpm: float
    global_time_signature: str
    sections: List[SectionTrace] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'PieceTrace':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        sections = []
        for s in data.get('sections', []):
            # Parse dicts back to dataclasses if needed, or keep as dicts
            sections.append(SectionTrace(**s))
        
        return cls(
            prompt=data['prompt'],
            global_key=data['global_key'],
            global_mode=data['global_mode'],
            global_bpm=data['global_bpm'],
            global_time_signature=data['global_time_signature'],
            sections=sections,
            metadata=data.get('metadata', {})
        )
