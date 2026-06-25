import pretty_midi
import numpy as np
from typing import List, Dict, Optional
import time

from .HarmonyRouter import HarmonyRouter
from .motif_memory import MotifMemoryFAISS
from ..critic.soft_refiner import SoftRefiner
from ..critic.midi_polisher import MIDIPolisher
from ..critic.hard_scorer import HardScorer
from shared.music_theory_constants import logger

class AgenticComposer:
    def __init__(self, harmonyrouter: HarmonyRouter, acceptance_threshold: float = 0.80):
        self.harmonyrouter = harmonyrouter
        self.refiner = SoftRefiner()
        self.polisher = MIDIPolisher()
        self.scorer = HardScorer()
        self.default_threshold = acceptance_threshold
        self.motif_memory = MotifMemoryFAISS() 

    def _map_json_to_moonbeam(self, blueprint: dict) -> dict:
        return {
            "metadata_ids": blueprint.get("metadata_ids", []), 
            "ties_weights": blueprint.get("ties_weights", {"commu_lora": 0.6, "emopia_lora": 0.2, "slakh_lora": 0.2}),
            "chord_timeline": blueprint.get("chord_timeline", []),
            "target_chords": blueprint.get("target_chords", []),
            "target_instruments": blueprint.get("target_instruments", []),
            "key": blueprint.get("key", "C"),
            "mode": blueprint.get("mode", "major"),
            "bpm": blueprint.get("bpm", 120)
        }

    def generate_section(self, section_name: str, blueprint: dict, max_attempts: int = 3) -> Optional[pretty_midi.PrettyMIDI]:
        logger.info(f"🎼 Composing Section: {section_name}...")
        mapped = self._map_json_to_moonbeam(blueprint)
        
        # --- ADAPTIVE THRESHOLDS ---
        thresholds = {
            "intro": 0.75, "verse": 0.80, "pre-chorus": 0.80,
            "chorus": 0.90, "bridge": 0.78, "outro": 0.75
        }
        current_threshold = thresholds.get(section_name.lower(), self.default_threshold)
        # ---------------------------

        current_ties = mapped["ties_weights"].copy()
        self.harmonyrouter.set_weights(current_ties)
        
        primer_tokens = None
        primer_midi = None
        base_theme = self.motif_memory.retrieve_main_theme()
        
        if base_theme and section_name.lower() != "intro":
            semantic_match = self.motif_memory.retrieve_semantic_primer(base_theme["midi"])
            if semantic_match:
                primer_tokens = semantic_match["tokens"]
                primer_midi = semantic_match["midi"]
                logger.info(f"🔗 Retrieved semantic primer: '{semantic_match['name']}'")
            
        avg_density = np.mean(blueprint.get("density_curve", [0.5]))
        current_temperature = 0.6 + (avg_density * 0.6) 
        
        best_midi = None
        best_score = 0.0
        best_tokens = None
        
        for attempt in range(max_attempts):
            start_time = time.time()
            
            raw_tokens = self.harmonyrouter.generate(
                metadata_ids=mapped["metadata_ids"],
                primer_tokens=primer_tokens,
                max_gen_len=blueprint.get("max_tokens", 256),
                temperature=current_temperature
            )
            
            current_chord = mapped["target_chords"][0] if mapped["target_chords"] else None
            safe_tokens = self.refiner.refine_tokens(
                raw_tokens, 
                root_note=mapped["key"], 
                mode=mapped["mode"],
                current_chord=current_chord,
                ties_weights=current_ties
            )
            
            raw_midi = self.harmonyrouter.tokenizer.decode_to_midi(safe_tokens)
            
            polished_midi = self.polisher.polish(
                raw_midi, 
                bpm=mapped["bpm"],
                chord_timeline=mapped["chord_timeline"],
                ties_weights=current_ties
            )
            
            # Pass section_name to the scorer for weighted evaluation
            score, feedback_data = self.scorer.score(
                polished_midi, 
                mapped, 
                primer_midi=primer_midi,
                section_name=section_name 
            )
            feedback_str = feedback_data.get("feedback", "No feedback")
            elapsed = time.time() - start_time
            
            logger.info(f"   ↳ Attempt {attempt+1} | Score: {score:.2f} | {feedback_str} | Time: {elapsed:.2f}s")
            
            if score > best_score:
                best_score = score
                best_midi = polished_midi
                best_tokens = safe_tokens
                
            # Check against the dynamic threshold
            if score >= current_threshold:
                logger.info(f"   ✅ Section '{section_name}' ACCEPTED! (Threshold: {current_threshold})")
                break
            else:
                logger.warning(f"   ❌ Section '{section_name}' REJECTED. (Needed {current_threshold})")
                
                if attempt < max_attempts - 1:
                    metrics = feedback_data.get("metrics", {})
                    logger.info("   🧠 Reflecting on failure and adaptively adjusting parameters...")
                    
                    if "chord_score" in metrics:
                        chord_error = 1.0 - metrics["chord_score"]
                        current_ties["commu_lora"] += chord_error * 0.4 
                        
                    if "inst_score" in metrics:
                        inst_error = 1.0 - metrics["inst_score"]
                        current_ties["slakh_lora"] += inst_error * 0.4
                        
                    total = sum(current_ties.values())
                    current_ties = {k: round(v/total, 3) for k, v in current_ties.items()}
                    self.harmonyrouter.set_weights(current_ties)
                    logger.info(f"      ↳ Adaptive TIES: {current_ties}")
                    
                    temp_drop = (1.0 - score) * 0.2
                    current_temperature = max(0.5, current_temperature - temp_drop)
                    logger.info(f"      ↳ Adaptive Temp: {current_temperature:.2f}")
                
        if best_midi and best_tokens:
            self.motif_memory.save(section_name, best_tokens, best_midi)
            
        if best_score < current_threshold:
             logger.error(f"   ⚠️ Max attempts reached for '{section_name}'. Returning best effort (Score: {best_score:.2f}).")
             
        return best_midi

    def compose_full_song(self, song_timeline: List[dict]) -> pretty_midi.PrettyMIDI:
        final_song = pretty_midi.PrettyMIDI()
        
        for section_data in song_timeline:
            section_name = section_data.get("section_name", section_data.get("name", "Section"))
            section_midi = self.generate_section(section_name, section_data)
            
            if section_midi:
                for inst in section_midi.instruments:
                    final_song.instruments.append(inst)
                    
        return final_song