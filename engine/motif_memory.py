import faiss
import numpy as np
import pretty_midi
from typing import Optional, Dict, Any
from shared.music_theory_constants import logger

class MotifMemoryFAISS:
    """
    Vector Database for Motif Storage and Retrieval using FAISS.
    Uses Cosine Similarity (Inner Product) for true semantic musical matching.
    """
    def __init__(self):
        self.dim = 28  # 12D Chroma + 12D Intervals + 4D Stats
        # IndexFlatIP = Inner Product. When vectors are normalized, IP == Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dim) 
        self.store = [] 

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalizes a vector to unit length for Cosine Similarity."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
        
    def _extract_embedding(self, midi_obj: pretty_midi.PrettyMIDI) -> np.ndarray:
        """Extracts a 28D musical fingerprint (Chroma + Intervals + Stats)."""
        chroma = np.zeros(12)
        intervals = np.zeros(12)
        total_notes = 0
        pitch_sum, vel_sum, dur_sum = 0.0, 0.0, 0.0
        
        all_notes = []
        for inst in midi_obj.instruments:
            if not inst.is_drum:
                all_notes.extend(inst.notes)
        all_notes.sort(key=lambda x: x.start)
        
        prev_pitch = -1
        for note in all_notes:
            chroma[note.pitch % 12] += 1
            pitch_sum += note.pitch
            vel_sum += note.velocity
            dur_sum += (note.end - note.start)
            
            if prev_pitch != -1:
                interval = (note.pitch - prev_pitch) % 12
                intervals[interval] += 1
            prev_pitch = note.pitch
            total_notes += 1
                    
        if total_notes > 0:
            chroma = chroma / np.sum(chroma)
            if np.sum(intervals) > 0:
                intervals = intervals / np.sum(intervals)
            stats = [
                total_notes / 100.0,                  
                (pitch_sum / total_notes) / 127.0,    
                (vel_sum / total_notes) / 127.0,      
                dur_sum / total_notes                 
            ]
        else:
            stats = [0.0, 0.0, 0.0, 0.0]
            
        return np.concatenate([chroma, intervals, stats]).astype('float32')

    def save(self, section_name: str, tokens: list, midi_obj: pretty_midi.PrettyMIDI):
        """Adds a generated section to the vector database."""
        emb = self._extract_embedding(midi_obj)
        emb = self._normalize(emb).reshape(1, -1) # <--- NORMALIZE BEFORE ADDING
        
        self.index.add(emb)
        self.store.append({
            "name": section_name,
            "type": section_name.lower(), # Stored for future section-conditioned retrieval
            "tokens": tokens,
            "midi": midi_obj
        })
        logger.info(f"💾 [FAISS] Saved '{section_name}' to vector memory. (Index size: {self.index.ntotal})")
        
    def retrieve_main_theme(self) -> Optional[Dict[str, Any]]:
        """Retrieves the Intro/Main Theme to use as a primer for subsequent sections."""
        for item in self.store:
            if "intro" in item["name"].lower() or "theme" in item["name"].lower():
                return item
        return self.store[0] if self.store else None
    
    def retrieve_semantic_primer(self, current_primer_midi: pretty_midi.PrettyMIDI) -> Optional[Dict[str, Any]]:
        """
        True Semantic Retrieval using Cosine Similarity.
        """
        if self.index.ntotal == 0 or current_primer_midi is None:
            return self.retrieve_main_theme()
            
        query_emb = self._extract_embedding(current_primer_midi)
        query_emb = self._normalize(query_emb).reshape(1, -1) # <--- NORMALIZE QUERY
        
        # Search FAISS for the 1 nearest neighbor
        similarities, indices = self.index.search(query_emb, 1)
        best_idx = indices[0][0]
        similarity = similarities[0][0]
        
        # Cosine Similarity threshold (1.0 = identical, 0.0 = unrelated)
        # If similarity > 0.6, they share a strong musical/melodic shape
        if similarity > 0.60:
            logger.info(f"💡 [FAISS] Semantic match found! (Cosine Sim: {similarity:.2f})")
            return self.store[best_idx]
            
        return self.retrieve_main_theme()