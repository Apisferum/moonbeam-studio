import os
import torch
import logging
import numpy as np
from safetensors.torch import load_file
from transformers import LlamaConfig

# --- HIGH-PERFORMANCE JSON STACK ---
try:
    import simdjson
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
# -----------------------------------

# Import custom Moonbeam architecture and the author's generation loop
from transformers.models.llama.modeling_llama import LlamaForCausalLM_Conditional_Generation
from generation import MusicLlama
from llama_recipes.datasets.music_tokenizer import MusicTokenizer

# Import our compiled Rust extension
import ties_core 

logger = logging.getLogger(__name__)

class HarmonyRouter:
    """
    The Master Conductor. 
    Manages the base model, hot-swaps LoRA weights via Rust, and generates 6D MIDI.
    """
    def __init__(
        self,
        base_model_path: str,
        lora_checkpoint_dir: str,
        model_config_path: str,
        master_dict_path: str,
        density: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.density = density

        print("🎹 [1/5] Loading Base Model Architecture...")
        self.config = LlamaConfig.from_pretrained(model_config_path)
        self.model = LlamaForCausalLM_Conditional_Generation(self.config)

        print("🧠 [2/5] Loading Base Weights...")
        ckpt = torch.load(base_model_path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        self._base_sd = {k.replace("module.", ""): v.clone() for k, v in sd.items()}
        self.model.load_state_dict(self._base_sd, strict=False)

        print("📂 [3/5] Extracting LoRA Task Vectors...")
        self._task_vectors, self._module_names = self._load_task_vectors(lora_checkpoint_dir)
        
        # Build direct parameter map for zero-copy in-place updates
        self._param_map = {
            mn: self.model.get_parameter(f"{mn}.weight")
            for mn in self._module_names
        }

        print("🦀 [4/5] Initializing Rust TIES Core...")
        task_arrays = [
            [delta.float().numpy() for delta in self._task_vectors[mn]]
            for mn in self._module_names
        ]
        self._merger = ties_core.TIESMerger(task_arrays, self.density)

        print("🎛️ [5/5] Loading Tokenizer & MusicLlama Wrapper...")
        self.tokenizer = MusicTokenizer(
            timeshift_vocab_size=self.config.onset_vocab_size, 
            dur_vocab_size=self.config.dur_vocab_size, 
            octave_vocab_size=self.config.octave_vocab_size, 
            pitch_class_vocab_size=self.config.pitch_class_vocab_size, 
            instrument_vocab_size=self.config.instrument_vocab_size, 
            velocity_vocab_size=self.config.velocity_vocab_size
        )
        
        # --- HIGH-PERFORMANCE DICTIONARY LOADING ---
        if USE_SIMDJSON:
            parser = simdjson.Parser()
            with open(master_dict_path, 'rb') as f:
                proxy = parser.parse(f.read())
                # Convert simdjson proxy to standard Python dict
                master_dict = {k: v for k, v in proxy.items()}
        else:
            import json
            with open(master_dict_path, "r") as f:
                master_dict = json.load(f)
        # ------------------------------------------

        for key, value in master_dict.items():
            self.tokenizer.add_new_tokens(token_name=key, token_val=value)
            
        # Wrap in the author's generation loop
        self._generator = MusicLlama(self.model, self.tokenizer, self.config)
        
        # Move to GPU and cast to BF16 for fast inference
        if self.device == "cuda":
            self.model.to("cuda").to(torch.bfloat16)
            
        print("✅ HarmonyRouter is ONLINE and ready to conduct!")

    def _load_task_vectors(self, lora_checkpoint_dir: str):
        """Loads the A and B matrices from the latest epoch and calculates deltas."""
        epoch_dirs = [d for d in os.listdir(lora_checkpoint_dir) if d.startswith("epoch_") and "_step_" not in d]
        if not epoch_dirs:
            raise ValueError(f"❌ No completed epoch folders found in {lora_checkpoint_dir}")
            
        latest = os.path.join(lora_checkpoint_dir, max(epoch_dirs, key=lambda x: int(x.split("_")[1])))
        
        adapters = ["commu_lora", "emopia_lora", "slakh_lora"]
        module_task_vectors = {}

        for adapter_name in adapters:
            adapter_dir = os.path.join(latest, adapter_name)
            if os.path.exists(os.path.join(adapter_dir, adapter_name)):
                adapter_dir = os.path.join(adapter_dir, adapter_name) # Handle nested folders

            # Use standard json for the tiny adapter_config file
            import json
            with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
                cfg = json.load(f)
            scaling = cfg["lora_alpha"] / cfg["r"]

            wf = os.path.join(adapter_dir, "adapter_model.safetensors")
            lora_sd = load_file(wf, device="cpu") if os.path.exists(wf) else torch.load(os.path.join(adapter_dir, "adapter_model.bin"), map_location="cpu")

            modules = {}
            for k, v in lora_sd.items():
                if "lora_A" in k:
                    mn = k.replace(".lora_A.weight", "").replace("base_model.model.", "")
                    modules.setdefault(mn, {})["A"] = v
                elif "lora_B" in k:
                    mn = k.replace(".lora_B.weight", "").replace("base_model.model.", "")
                    modules.setdefault(mn, {})["B"] = v

            for mn, t in modules.items():
                if "A" in t and "B" in t:
                    delta = (t["B"] @ t["A"]) * scaling
                    module_task_vectors.setdefault(mn, []).append(delta)

        module_names = [mn for mn in module_task_vectors if f"{mn}.weight" in self._base_sd]
        return module_task_vectors, module_names

    def set_weights(self, weights_dict: dict):
        """
        Hot-swaps the model weights using the Rust backend.
        weights_dict: {"commu_lora": 0.6, "emopia_lora": 0.2, "slakh_lora": 0.2}
        """
        adapters = ["commu_lora", "emopia_lora", "slakh_lora"]
        assert set(weights_dict.keys()) == set(adapters), "Must provide weights for all 3 adapters."
        assert abs(sum(weights_dict.values()) - 1.0) < 1e-4, "Weights must sum to 1.0"
        
        weights = [weights_dict[k] for k in adapters]
        
        # 1. Get base weights as CPU Float32 NumPy arrays
        base_arrays = [self._base_sd[f"{mn}.weight"].float().cpu().numpy() for mn in self._module_names]
        
        # 2. Call Rust (Trim cache hit/miss + Sign Election + Disjoint Merge)
        merged_arrays = self._merger.merge(base_arrays, weights)
        
        # 3. In-place PyTorch update (Zero-copy view -> GPU/BF16 cast)
        with torch.no_grad():
            for mn, arr in zip(self._module_names, merged_arrays):
                param = self._param_map[mn]
                # torch.from_numpy creates a CPU tensor, .copy_ handles the GPU/BF16 transfer instantly
                param.copy_(torch.from_numpy(np.array(arr)))
                
        print(f"🎛️ Weights hot-swapped: {weights_dict}")

    def generate(self, metadata_ids: list, primer_tokens: list = None, max_gen_len: int = 512, temperature: float = 0.8, top_p: float = 0.9):
        """
        Generates a MIDI object. 
        If primer_tokens is provided, it uses them as the starting context (Motif).
        """
        metadata_condition_decoder = [metadata_ids]
        
        # --- MOTIF INJECTION ---
        if primer_tokens and len(primer_tokens) > 0:
            # Use the retrieved motif as the prompt!
            # (Assuming primer_tokens is a list of 6D lists, e.g., [[0,4,4,0,40,80], ...])
            prompts = [primer_tokens] 
            logger.info(f"🔗 Using Motif Primer ({len(primer_tokens)} tokens) for generation.")
        else:
            # Fallback: Start from scratch with just the SOS token
            prompts = [[[self.tokenizer.sos_token_compound for _ in range(6)]]] 
        # -----------------------
        
        results = self._generator.music_completion(
            prompts,
            bpm_condition=[120], 
            time_signature_condition=["4/4"],
            num_measures_condition=[8],
            metadata_condition=metadata_condition_decoder,
            chord_condition=None, 
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            condition_token_lengths=[len(primer_tokens) if primer_tokens else 1], # Tell the engine where the prompt ends
            chord_dict=None,
            if_return_chords=False
        )
        
        return results[0]['generation']['content']

    def save_generation_metadata(self, filepath: str, metadata: dict):
        """
        High-performance JSON writer for saving generation logs/parameters.
        Uses orjson for C-level serialization speeds.
        """
        if USE_ORJSON:
            with open(filepath, "wb") as f:
                f.write(orjson.dumps(metadata, option=orjson.OPT_SERIALIZE_NUMPY))
        else:
            import json
            with open(filepath, "w") as f:
                json.dump(metadata, f)