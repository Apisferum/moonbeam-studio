import math
import random
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

# Import the Single Source of Truth
from shared.music_theory_constants import (
    logger, NOTE_MAP, PC_TO_NAME, SCALE_MODES, 
    HARMONIC_FUNCTIONS_MAJOR, HARMONIC_FUNCTIONS_MINOR, MACRO_SKELETONS,
    MAJOR_QUALITIES, MINOR_QUALITIES, ARRANGEMENT_PROFILES, TIES_PRESETS
)

class StructurePlanner:
    """
    The 'Algorithmic Composer' (Hybrid Edition).
    Combines Harmonic Function Skeletons (Macro) with Markov Creativity (Micro),
    while generating the rich metadata required by the HarmonyRouter and Critic.
    """
    def __init__(self):
        pass

    def _get_relative_key(self, root_pc: int, mode: str) -> tuple:
        """Calculates the Relative Major/Minor (shares the exact same key signature)."""
        if mode == "minor":
            # Relative Major is +3 semitones
            new_root = (root_pc + 3) % 12
            return PC_TO_NAME[new_root], "major"
        else:
            # Relative Minor is -3 semitones (or +9)
            new_root = (root_pc + 9) % 12
            return PC_TO_NAME[new_root], "minor"

    def _degree_to_chord(self, degree: str, key_pc: int, scale_intervals: list, is_minor: bool) -> Tuple[str, str]:
        clean = degree.replace("°", "")
        roman_to_idx = {"i": 0, "I": 0, "ii": 1, "iii": 2, "III": 2, "iv": 3, "IV": 3, "v": 4, "V": 4, "vi": 5, "VI": 5, "vii": 6, "VII": 6}
        idx = roman_to_idx.get(clean, 0)
        
        # --- SAFE BOUND CHECK ---
        interval = scale_intervals[idx] if idx < len(scale_intervals) else scale_intervals[-1]
        chord_root_pc = (key_pc + interval) % 12
        root_name = PC_TO_NAME.get(chord_root_pc, "C")
        
        qualities = MINOR_QUALITIES if is_minor else MAJOR_QUALITIES
        quality = qualities.get(degree, "maj7")
        if "°" in degree: quality = "dim"
            
        return f"{root_name}{quality}", quality

    def _generate_function_markov_chords(self, key: str, mode: str, section_name: str, bars: int) -> List[Dict[str, Any]]:
        """Generates chords using a Harmonic Function Skeleton + Markov Selection."""
        key_pc = NOTE_MAP.get(key, 0)
        scale_intervals = SCALE_MODES.get(mode, SCALE_MODES["major"])
        is_minor = (mode == "minor")
        func_groups = HARMONIC_FUNCTIONS_MINOR if is_minor else HARMONIC_FUNCTIONS_MAJOR
        
        base_skeleton = MACRO_SKELETONS.get(section_name, ["T", "S", "D", "T"])
        skeleton = (base_skeleton * math.ceil(bars / len(base_skeleton)))[:bars]
        
        if len(skeleton) >= 2:
            skeleton[-2] = "D"
            skeleton[-1] = "T"
            
        progression = []
        total_beats = 0
        
        for i, func in enumerate(skeleton):
            degree = random.choice(func_groups[func])
            symbol, quality = self._degree_to_chord(degree, key_pc, scale_intervals, is_minor)
            
            duration_beats = 4 if func != "D" else 2 
            
            if total_beats + duration_beats > bars * 4:
                duration_beats = (bars * 4) - total_beats
                
            progression.append({
                "symbol": symbol, "degree": degree, "quality": quality,
                "duration_beats": duration_beats, "start_beat": total_beats, "end_beat": total_beats + duration_beats
            })
            total_beats += duration_beats
            if total_beats >= bars * 4: break
            
        return progression

    def _generate_density_curve(self, section_name: str, bars: int, energy_curve: str = "build") -> List[float]:
        if bars <= 0: return []
        x = np.linspace(0, 1, bars)
        
        # Base curves per section
        if section_name == "Intro": base = 0.2 + 0.3 * x
        elif section_name == "Chorus": base = 0.7 + 0.3 * np.sin(x * np.pi)
        elif section_name == "Outro": base = 0.6 * (1 - x)
        else: base = np.full(bars, 0.5)
        
        # Apply Global Energy Curve Modifier
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

    def _get_arrangement(self, sections: List[str], mood: str) -> Dict[str, List[str]]:
        style = "build" if mood in ("heroic", "tension") else "sparse_to_dense"
        profile = ARRANGEMENT_PROFILES.get(style, ARRANGEMENT_PROFILES["build"])
        return {s: profile.get(s, ["Piano", "Strings"]) for s in sections}

    def _get_ties_weights(self, section_name: str, mood: str, instruments: List[str]) -> Dict[str, float]:
        base = TIES_PRESETS.get("balanced").copy()
        if section_name == "Chorus": base["slakh_lora"] += 0.15
        if mood in ("sad", "melancholy"): base["emopia_lora"] += 0.2
        if len(instruments) >= 4: base["slakh_lora"] += 0.1
        total = sum(base.values())
        return {k: round(v / total, 3) for k, v in base.items()}

    def plan(self, llm_output: dict) -> dict:
        # 1. Parse Global Settings
        raw_key = llm_output.get("global_key", "C")
        global_key = raw_key.split()[0].replace("minor", "").replace("major", "").strip()
        global_mode = llm_output.get("global_mode", "major")
        bpm = llm_output.get("global_bpm", 120)
        style = llm_output.get("style", "cinematic")
        energy_curve = llm_output.get("energy_curve", "build")
        modulations = llm_output.get("modulations", [])
        
        sections_input = llm_output.get("sections", [{"name": "Verse", "mood": "happy", "length": "medium"}])
        
        # Build a lookup dictionary for requested modulations
        mod_map = {m["target_section"]: m for m in modulations}
        
        section_names = [s.get("name", "Verse") for s in sections_input]
        primary_mood = sections_input[0].get("mood", "default")
        arrangement = self._get_arrangement(section_names, primary_mood)
        
        dense_timeline = []
        
        # Track the "Current Key" as we move through the song
        current_key = global_key
        current_mode = global_mode
        
        for section in sections_input:
            name = section.get("name", "Verse")
            mood = section.get("mood", primary_mood)
            bars = {"short": 4, "medium": 8, "long": 16}.get(section.get("length", "medium"), 8)
            
            # --- APPLY SAFE MODULATION ---
            if name in mod_map:
                mod_intent = mod_map[name]
                target_key = mod_intent.get("target_key")
                target_mode = mod_intent.get("target_mode")
                
                # If LLM didn't specify exact notes, use Relative Major/Minor (Safest transition)
                if not target_key or not target_mode:
                    current_key_pc = NOTE_MAP.get(current_key, 0)
                    current_key, current_mode = self._get_relative_key(current_key_pc, current_mode)
                else:
                    current_key = target_key
                    current_mode = target_mode
                    
                logger.info(f"🎼 Modulating to {current_key} {current_mode} for the {name}!")
            # -----------------------------

            # 1. Function-Aware Markov Chords (Using the CURRENT section's key!)
            progression = self._generate_function_markov_chords(current_key, current_mode, name, bars)
            chord_timeline = self._build_chord_timeline(progression, bpm)
            
            # 2. Rich Metadata
            instruments = arrangement.get(name, ["Piano"])
            ties = self._get_ties_weights(name, mood, instruments)
            
            # --- APPLY ENERGY CURVE TO DENSITY ---
            density_curve = self._generate_density_curve(name, bars, energy_curve)
            
            dense_timeline.append({
                "section_name": name, "bars": bars, 
                "key": current_key, "mode": current_mode, # <--- PER-SECTION KEY!
                "bpm": bpm, "style": style,
                "progression": progression,
                "target_chords": [c["symbol"] for c in progression],
                "chord_timeline": chord_timeline,
                "target_instruments": instruments,
                "ties_weights": ties,
                "density_curve": density_curve,
                "max_tokens": bars * 16,
                "motif_action": "generate_and_save_seed" if name == "Intro" else "retrieve_seed"
            })
            
        return {
            "global_settings": {"key": global_key, "mode": global_mode, "bpm": bpm, "style": style}, 
            "timeline": dense_timeline
        }