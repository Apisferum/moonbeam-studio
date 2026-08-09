import sys
import os
import gc

moonbeam_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "moonbeam-studio"))
if moonbeam_root not in sys.path:
    sys.path.insert(0, moonbeam_root)

import torch
import logging
import numpy as np
from collections import deque
from typing import Optional
from safetensors.torch import load_file
from transformers import LlamaConfig

try:
    import simdjson
    _json_parser = simdjson.Parser()
    USE_SIMDJSON = True
except ImportError:
    import json
    USE_SIMDJSON = False

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    import json
    USE_ORJSON = False

from transformers.models.llama.modeling_llama import LlamaForCausalLM_Conditional_Generation
from recipes.inference.custom_music_generation.generation import MusicLlama
from llama_recipes.datasets.music_tokenizer import MusicTokenizer

import ties_core

logger = logging.getLogger(__name__)
ADAPTERS = ["commu_lora", "emopia_lora", "slakh_lora"]

def _read_json(path: str) -> dict:
    if USE_SIMDJSON:
        with open(path, "rb") as f:
            return _json_parser.parse(f.read()).as_dict()
    with open(path, "r") as f:
        return json.load(f)

def _write_json(path: str, obj: dict) -> None:
    if USE_ORJSON:
        with open(path, "wb") as f:
            f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY))
        return
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _pick_dtype(device: str) -> torch.dtype:
    if device != "cuda" or not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16

# TIMING FORCING: real-time tick convention shared with the rest of the
# pipeline (music_theory_constants.TICKS_PER_SECOND). Kept as a local
# constant here rather than importing shared.music_theory_constants, since
# this module already has its own tight import surface and this value is
# fixed by the tokenizer's own architecture (TIME_RESOLUTION=100 in
# MusicTokenizer), not something this file should independently configure.
_TICKS_PER_SECOND = 100

class HarmonyRouter:
    def __init__(self, base_model_path: str, lora_checkpoint_dir: str, model_config_path: str,
                 master_dict_path: str, density: float = 0.8, device: Optional[str] = None):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cuda_ok = self.device == "cuda" and torch.cuda.is_available()
        self.dtype = _pick_dtype(self.device)
        self.density = density

        print("🎹 [1/5] Loading Base Model Architecture...")
        self.config = LlamaConfig.from_pretrained(model_config_path)
        self.model = LlamaForCausalLM_Conditional_Generation(self.config)

        print("🧠 [2/5] Loading Base Weights...")
        ckpt = torch.load(base_model_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        base_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(base_sd, strict=False)

        self._temp_base_sd = base_sd
        del ckpt, sd
        if self._cuda_ok:
            torch.cuda.empty_cache()

        print("📂 [3/5] Extracting LoRA Task Vectors...")
        self._task_vectors, self._module_names = self._load_task_vectors(lora_checkpoint_dir)

        self._base_sd = {}
        for mn in self._module_names:
            key = f"{mn}.weight"
            if key in self._temp_base_sd:
                self._base_sd[key] = self._temp_base_sd[key].clone()

        del self._temp_base_sd
        gc.collect()
        print(f"🧹 Freed ~3GB RAM by filtering base weights to {len(self._base_sd)} LoRA modules.")

        print("🦀 [4/5] Initializing Rust TIES Core...")
        task_arrays = [[delta.float().numpy().flatten() for delta in self._task_vectors[mn]] for mn in self._module_names]
        self._merger = ties_core.TIESMerger(task_arrays, self.density)

        if self._cuda_ok:
            print(f"   ↳ Moving model to cuda, dtype={self.dtype} (auto-selected for this GPU's compute capability)")
            self.model.to(device="cuda", dtype=self.dtype)
            torch.cuda.empty_cache()

        self._param_map = {mn: self.model.get_parameter(f"{mn}.weight") for mn in self._module_names}

        print("🎛️ [5/5] Loading Tokenizer & MusicLlama Wrapper...")
        self.tokenizer = MusicTokenizer(
            timeshift_vocab_size=self.config.onset_vocab_size, dur_vocab_size=self.config.dur_vocab_size,
            octave_vocab_size=self.config.octave_vocab_size, pitch_class_vocab_size=self.config.pitch_class_vocab_size,
            instrument_vocab_size=self.config.instrument_vocab_size, velocity_vocab_size=self.config.velocity_vocab_size,
        )

        master_dict = _read_json(master_dict_path)
        self.master_dict = master_dict 
        
        for key, value in master_dict.items():
            self.tokenizer.add_new_tokens(token_name=key, token_val=value)

        # 🚀 DRIFT PREVENTION TRIPWIRE (MOVED TO CORRECT LOCATION)
        from shared.music_theory_constants import TICKS_PER_SECOND
        tokenizer_resolution = getattr(self.tokenizer, 'TIME_RESOLUTION', 100)
        if tokenizer_resolution != TICKS_PER_SECOND:
            raise ValueError(f"FATAL ARCHITECTURE DRIFT: MusicTokenizer resolution ({tokenizer_resolution}) does not match TICKS_PER_SECOND ({TICKS_PER_SECOND})!")

        self._generator = MusicLlama(self.model, self.tokenizer, self.config)
        self.octave_vocab_size = self.config.octave_vocab_size
        self.pitch_class_vocab_size = self.config.pitch_class_vocab_size
        self.instrument_vocab_size = self.config.instrument_vocab_size
        print("✅ HarmonyRouter is ONLINE and ready to conduct!")

    def _load_task_vectors(self, lora_checkpoint_dir: str):
        epoch_dirs = [d for d in os.listdir(lora_checkpoint_dir) if d.startswith("epoch_")]
        if not epoch_dirs:
            raise ValueError(f"❌ No epoch folders found in {lora_checkpoint_dir}")

        completed = [d for d in epoch_dirs if "_step_" not in d]
        target = max(completed, key=lambda x: int(x.split("_")[1])) if completed else max(epoch_dirs, key=lambda x: int(x.split("_")[1]))
        latest = os.path.join(lora_checkpoint_dir, target)
        print(f"   📂 Loading adapters from: {latest}")

        raw_deltas = {adapter: {} for adapter in ADAPTERS}
        self._module_shapes = {}

        for adapter_name in ADAPTERS:
            adapter_dir = os.path.join(latest, adapter_name, adapter_name) if os.path.exists(os.path.join(latest, adapter_name, adapter_name)) else os.path.join(latest, adapter_name)
            cfg = _read_json(os.path.join(adapter_dir, "adapter_config.json"))
            scaling = cfg["lora_alpha"] / cfg["r"]
            wf = os.path.join(adapter_dir, "adapter_model.safetensors")
            lora_sd = load_file(wf, device="cpu") if os.path.exists(wf) else torch.load(os.path.join(adapter_dir, "adapter_model.bin"), map_location="cpu", weights_only=True)

            modules = {}
            for k, v in lora_sd.items():
                if "lora_A" in k:
                    modules.setdefault(k.replace(".lora_A.weight", "").replace("base_model.model.", ""), {})["A"] = v
                elif "lora_B" in k:
                    modules.setdefault(k.replace(".lora_B.weight", "").replace("base_model.model.", ""), {})["B"] = v

            for mn, t in modules.items():
                if "A" in t and "B" in t:
                    delta = (t["B"] @ t["A"]) * scaling
                    raw_deltas[adapter_name][mn] = delta
                    if mn not in self._module_shapes:
                        self._module_shapes[mn] = tuple(delta.shape)

        module_task_vectors = {}
        module_names = []

        for mn in self._temp_base_sd:
            clean_mn = mn.replace(".weight", "")
            if clean_mn in self._module_shapes:
                module_names.append(clean_mn)
                vec_list = []
                for adapter_name in ADAPTERS:
                    if clean_mn in raw_deltas[adapter_name]:
                        vec_list.append(raw_deltas[adapter_name][clean_mn])
                    else:
                        vec_list.append(torch.zeros(self._module_shapes[clean_mn]))
                module_task_vectors[clean_mn] = vec_list

        return module_task_vectors, module_names

    def set_weights(self, weights_dict: dict):
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            weights_dict = {k: v / total_weight for k, v in weights_dict.items()}

        weights = [weights_dict.get(k, 0.0) for k in ADAPTERS]
        base_arrays = [self._base_sd[f"{mn}.weight"].float().cpu().numpy().flatten() for mn in self._module_names]
        merged_arrays = self._merger.merge(base_arrays, weights)

        with torch.no_grad():
            for mn, arr in zip(self._module_names, merged_arrays):
                self._param_map[mn].copy_(torch.from_numpy(arr).reshape(self._module_shapes[mn]))
        print(f"🎛️ Weights hot-swapped: {weights_dict}")

    def _build_forced_stream(self, note_events: list, bpm: int = 120, primer_offset_ticks: int = 0) -> deque:
        """
        TIMING FORCING: mirrors AgenticComposer._build_forced_streams — see
        that method's docstring for the full rationale, including the
        BUGFIX around primer_offset_ticks: target_tick must be expressed on
        the SAME absolute onset scale the model is actually continuing
        from, which starts at the primer's own last onset (not 0) whenever
        a primer is present. This entry point isn't on AgenticComposer's
        actual call path (it pre-builds forced_streams itself and passes
        them directly to .generate()), but this is a public method other
        callers could reasonably use, so it shouldn't silently lack this
        fix while the other path has it.
        """
        stream = deque()
        max_oct = self.tokenizer.octave_vocab_size - 3
        max_pit = self.tokenizer.pitch_class_vocab_size - 3
        max_ins = self.tokenizer.instrument_vocab_size - 3
        beat_duration_ticks = (60.0 / bpm) * _TICKS_PER_SECOND

        for note in note_events:
            oct_idx = max(0, min(note.get("octave", 0), max_oct))
            pit_idx = max(0, min(note.get("pitch_class", 0), max_pit))
            ins_idx = max(0, min(note.get("instrument_program", 0), max_ins))
            target_tick = primer_offset_ticks + int(round(note.get('start_beat', 0) * beat_duration_ticks))
            duration_beats = note.get('duration_beats', 0.25)
            target_duration_ticks = max(1, int(round(duration_beats * beat_duration_ticks)))

            stream.append({
                "octave_tok": self.tokenizer.octave_dict.get(oct_idx, 0),
                "pitch_tok": self.tokenizer.pitch_dict.get(pit_idx, 0),
                "instrument_tok": self.tokenizer.instrument_dict.get(ins_idx, 0),
                "target_tick": target_tick,
                "target_duration_ticks": target_duration_ticks,
            })
        return stream

    def generate(self, metadata_ids: list, primer_tokens: list = None, max_gen_len: int = 512,
                 temperature: float = 0.8, top_p: float = 0.9,
                 bpm: int = 120, num_measures: int = 8, time_signature: str = "4/4",
                 forced_token_streams: list = None, note_events: list = None):
        
        if not metadata_ids:
            metadata_ids = [-4] * 11
        elif len(metadata_ids) < 11:
            metadata_ids.extend([-4] * (11 - len(metadata_ids)))
        elif len(metadata_ids) > 11:
            metadata_ids = metadata_ids[:11]

        if forced_token_streams is None and note_events:
            # BUGFIX: must offset target_tick by the primer's own last onset
            # when a primer is present — see _build_forced_stream's
            # docstring. Confirmed via a real run where this was missing:
            # every forced note in a primer-continued section landed
            # outside the section's own time window.
            primer_offset_ticks = 0
            if primer_tokens:
                six_d_primer_toks = [t for t in primer_tokens if len(t) == 6]
                if six_d_primer_toks:
                    primer_offset_ticks = six_d_primer_toks[-1][0]
            forced_token_streams = [self._build_forced_stream(note_events, bpm=bpm, primer_offset_ticks=primer_offset_ticks)]

        results = self._generator.music_completion(
            [primer_tokens] if primer_tokens else [[self.tokenizer.sos_token_compound]],
            bpm_condition=[bpm], time_signature_condition=[time_signature], num_measures_condition=[num_measures],
            metadata_condition=[metadata_ids], chord_condition=None, max_gen_len=max_gen_len,
            temperature=temperature, top_p=top_p,
            condition_token_lengths=[len(primer_tokens) if primer_tokens else 1],
            chord_dict=None, if_return_chords=False, forced_token_streams=forced_token_streams,
        )

        gen_data = results[0]
        if 'tokens' in gen_data:
            return gen_data['tokens']
        if 'generation' in gen_data and isinstance(gen_data['generation'], dict):
            return gen_data['generation'].get('tokens', gen_data['generation'].get('content', None))
        return gen_data.get('content', None)

    def save_generation_metadata(self, filepath: str, metadata: dict):
        _write_json(filepath, metadata)

    def clear_ties_cache(self):
        freed = self._merger.clear_cache()
        print(f"🧹 TIES cache cleared ({freed} entries freed).")