import faiss
import numpy as np
import pretty_midi
from typing import Optional, Dict, Any, List
from shared.music_theory_constants import logger

class MotifMemoryFAISS:
    """
    Vector Database for Motif Storage and Retrieval using FAISS.
    Uses Cosine Similarity (Inner Product) for true semantic musical matching.
    """
    def __init__(self):
        self.dim = 28  # 12D Chroma + 12D Intervals + 4D Stats
        cpu_index = faiss.IndexFlatIP(self.dim)
        
        # Check if CUDA is available and transfer to GPU if so
        try:
            import torch
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                # Transfer CPU index to GPU (Device 0)
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.info("⚡ [FAISS] Successfully initialized GPU index using StandardGpuResources.")
            else:
                self.index = cpu_index
        except Exception as e:
            logger.warning(f"⚠️ [FAISS] GPU index creation failed ({e}) — falling back to CPU index.")
            self.index = cpu_index
        self.store = []

    def clear(self):
        """Resets the FAISS index and clears the store to prevent cross-song context leaks and memory growth."""
        self.index.reset()
        self.store.clear()
        logger.info("🧹 [FAISS] Cleared motif memory database.")

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _extract_embedding_from_tokens(self, tokens: list) -> np.ndarray:
        """🚀 GOD-TIER OPTIMIZATION: Calculates embedding DIRECTLY from tokens."""
        if not tokens:
            return np.zeros(self.dim, dtype='float32')
            
        tokens_np = np.array(tokens, dtype=np.int32)
        valid_mask = tokens_np[:, 0] >= 0 
        tokens_np = tokens_np[valid_mask]
        
        if len(tokens_np) == 0:
            return np.zeros(self.dim, dtype='float32')

        durations = tokens_np[:, 1]
        octaves = tokens_np[:, 2]
        pitch_classes = tokens_np[:, 3]
        velocities = tokens_np[:, 5]
        
        chroma = np.bincount(pitch_classes % 12, minlength=12).astype('float32')
        if chroma.sum() > 0: chroma /= chroma.sum()
            
        intervals = np.zeros(12, dtype='float32')
        if len(pitch_classes) > 1:
            diffs = (pitch_classes[1:] - pitch_classes[:-1]) % 12
            intervals = np.bincount(diffs, minlength=12).astype('float32')
            if intervals.sum() > 0: intervals /= intervals.sum()
                
        stats = np.array([
            len(tokens_np) / 100.0,
            octaves.mean() / 10.0,
            velocities.mean() / 127.0,
            durations.mean() / 100.0
        ], dtype='float32')
        
        return np.concatenate([chroma, intervals, stats])

    def _extract_embedding(self, midi_obj: pretty_midi.PrettyMIDI) -> np.ndarray:
        """Fallback extraction using pretty_midi (kept for backward compatibility)."""
        chroma = np.zeros(12)
        intervals = np.zeros(12)
        total_notes = 0
        pitch_sum, vel_sum, dur_sum = 0.0, 0.0, 0.0

        all_notes = [note for inst in midi_obj.instruments if not inst.is_drum for note in inst.notes]
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

    def save(self, section_name: str, tokens: list, midi_obj: Optional[pretty_midi.PrettyMIDI] = None):
        """Adds a generated section to the vector database. Prefers raw tokens for speed."""
        if tokens:
            emb = self._extract_embedding_from_tokens(tokens)
        elif midi_obj is not None:
            emb = self._extract_embedding(midi_obj)
        else:
            emb = np.zeros(self.dim, dtype='float32')
            
        emb = self._normalize(emb).reshape(1, -1)

        self.index.add(emb)
        self.store.append({
            "name": section_name,
            "type": section_name.lower(),
            "tokens": tokens,
            "midi": midi_obj
        })
        logger.info(f"💾 [FAISS] Saved '{section_name}' to vector memory instantly. (Index size: {self.index.ntotal})")

    def save_section(self, midi_obj: Optional[pretty_midi.PrettyMIDI], section: dict, tokens: Optional[list] = None):
        section_name = section.get("section_name", section.get("name", "section"))
        self.save(section_name, tokens or [], midi_obj)

    def retrieve_main_theme(self) -> Optional[Dict[str, Any]]:
        for item in self.store:
            if "intro" in item["name"].lower() or "theme" in item["name"].lower():
                return item
        return self.store[0] if self.store else None

    def retrieve_semantic_primer(self, query_input) -> Optional[Dict[str, Any]]:
        try:
            from eval.instrumentation.hooks import hook_log_motif
        except ImportError:
            def hook_log_motif(*args): pass

        if self.index.ntotal == 0 or query_input is None:
            theme = self.retrieve_main_theme()
            if theme:
                hook_log_motif(theme["name"], 1.0, 0)
            return theme

        # Determine if input is tokens (list) or MIDI object
        if isinstance(query_input, list):
            query_emb = self._extract_embedding_from_tokens(query_input)
        elif hasattr(query_input, 'instruments'): # pretty_midi fallback
            query_emb = self._extract_embedding(query_input)
        else:
            theme = self.retrieve_main_theme()
            if theme:
                hook_log_motif(theme["name"], 1.0, 0)
            return theme

        query_emb = self._normalize(query_emb).reshape(1, -1)

        similarities, indices = self.index.search(query_emb, 1)
        best_idx = indices[0][0]
        similarity = similarities[0][0]

        if similarity > 0.60:
            logger.info(f"💡 [FAISS] Semantic match found! (Cosine Sim: {similarity:.2f})")
            res = self.store[best_idx]
            hook_log_motif(res["name"], float(similarity), int(best_idx))
            return res

        theme = self.retrieve_main_theme()
        if theme:
            hook_log_motif(theme["name"], float(similarity), 0)
        return theme

    def retrieve_primer(self, query_input) -> Optional[List]:
        entry = self.retrieve_semantic_primer(query_input) if query_input is not None else self.retrieve_main_theme()
        return entry["tokens"] if entry else None