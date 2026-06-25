import pretty_midi
import numpy as np
from typing import Dict, List, Tuple, Optional
from shared.music_theory_constants import logger, NOTE_MAP, CHORD_INTERVALS, INSTRUMENT_FAMILIES

class HardScorer:
    """
    The 'A&R Rep'. 
    Grades the polished MIDI against the JSON Blueprint AND the Motif Primer.
    Uses section-dependent weighted scoring.
    """
    
    def score(self, midi_obj: pretty_midi.PrettyMIDI, blueprint: dict, primer_midi: Optional[pretty_midi.PrettyMIDI] = None, section_name: str = "verse") -> Tuple[float, dict]:
        metrics = {}
        feedback_strs = []
        
        # 1. Chord Adherence
        if "target_chords" in blueprint and blueprint["target_chords"]:
            chord_score, chord_fb = self._score_chord_adherence(midi_obj, blueprint["target_chords"])
            metrics["chord_score"] = chord_score
            feedback_strs.append(chord_fb)
            
            # Hard Floor
            if chord_score < 0.50:
                return 0.0, {"error": "critical_chord_failure", "metrics": metrics, "feedback": " | ".join(feedback_strs)}
                
        # 2. Instrument Density
        if "target_instruments" in blueprint and blueprint["target_instruments"]:
            inst_score, inst_fb = self._score_instrument_density(midi_obj, blueprint["target_instruments"])
            metrics["inst_score"] = inst_score
            feedback_strs.append(inst_fb)
            
        # 3. Motif Coherence
        if primer_midi is not None:
            motif_score, motif_fb = self._score_motif_coherence(midi_obj, primer_midi)
            metrics["motif_score"] = motif_score
            feedback_strs.append(motif_fb)
            
        # --- SECTION-DEPENDENT WEIGHTED SCORING ---
        section_type = section_name.lower()
        if "chorus" in section_type:
            weights = {"chord": 0.4, "inst": 0.4, "motif": 0.2} # Chorus needs heavy orchestration
        elif "bridge" in section_type:
            weights = {"chord": 0.3, "inst": 0.2, "motif": 0.5} # Bridge needs motif development
        elif "intro" in section_type or "outro" in section_type:
            weights = {"chord": 0.3, "inst": 0.3, "motif": 0.4}
        else: # verse, pre-chorus
            weights = {"chord": 0.5, "inst": 0.2, "motif": 0.3} # Verse needs strong chord adherence
            
        final_score = 0.0
        if "chord_score" in metrics: final_score += metrics["chord_score"] * weights["chord"]
        if "inst_score" in metrics: final_score += metrics["inst_score"] * weights["inst"]
        if "motif_score" in metrics: final_score += metrics["motif_score"] * weights["motif"]
        
        if not metrics: final_score = 1.0 # Fallback if no metrics were calculated
        
        return final_score, {"metrics": metrics, "feedback": " | ".join(feedback_strs)}

    def _extract_chroma(self, midi_obj: pretty_midi.PrettyMIDI) -> np.ndarray:
        chroma = np.zeros(12)
        for inst in midi_obj.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    chroma[note.pitch % 12] += 1
        if np.sum(chroma) > 0:
            chroma = chroma / np.sum(chroma)
        return chroma

    def _score_motif_coherence(self, midi_obj: pretty_midi.PrettyMIDI, primer_midi: pretty_midi.PrettyMIDI) -> Tuple[float, str]:
        emb_gen = self._extract_chroma(midi_obj)
        emb_primer = self._extract_chroma(primer_midi)
        
        dot = np.dot(emb_gen, emb_primer)
        norm = np.linalg.norm(emb_gen) * np.linalg.norm(emb_primer)
        sim = dot / norm if norm > 0 else 0.0
        
        score = min(1.0, sim * 1.2) 
        return score, f"Motif Sim: {sim*100:.0f}%"

    def _get_root_pc(self, chord_symbol: str) -> int:
        if not chord_symbol or chord_symbol in ("N", "s"): return -1
        if len(chord_symbol) >= 2 and chord_symbol[1] in ('#', 'b'):
            root_str = chord_symbol[:2]
        else:
            root_str = chord_symbol[0]
        return NOTE_MAP.get(root_str, -1)

    def _score_chord_adherence(self, midi_obj: pretty_midi.PrettyMIDI, target_chords: List[str]) -> Tuple[float, str]:
        root_pcs = [self._get_root_pc(c) for c in target_chords if self._get_root_pc(c) != -1]
        if not root_pcs: return 1.0, "No chords to check."
        bass_pcs = []
        for inst in midi_obj.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    if note.pitch < 50: bass_pcs.append(note.pitch % 12)
        if not bass_pcs: return 0.5, "Warning: No bass notes generated."
        matches = sum(1 for pc in root_pcs if pc in bass_pcs)
        score = matches / len(root_pcs)
        return score, f"Chord Match: {score*100:.0f}%"

    def _score_instrument_density(self, midi_obj: pretty_midi.PrettyMIDI, target_instruments: List[str]) -> Tuple[float, str]:
        generated_programs = [inst.program for inst in midi_obj.instruments if not inst.is_drum]
        found_families = 0
        total_families = len(target_instruments)
        for target in target_instruments:
            valid_programs = INSTRUMENT_FAMILIES.get(target, [])
            if any(prog in valid_programs for prog in generated_programs):
                found_families += 1
        score = found_families / total_families if total_families > 0 else 1.0
        return score, f"Instruments: {found_families}/{total_families}"