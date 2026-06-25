from HarmonyRouter import HarmonyRouter
import json

# 1. Boot the Engine (Takes ~10 seconds)
engine = HarmonyRouter(
    base_model_path="/path/to/moonbeam_839M.pt",
    lora_checkpoint_dir="/path/to/multi_task_lora",
    model_config_path="src/llama_recipes/configs/model_config_multi_task.json",
    master_dict_path="processed/ComMU/indexed_tokens_dict.json"
)

# 2. Load the Master Dictionary to build a prompt
with open("processed/ComMU/indexed_tokens_dict.json", "r") as f:
    master_dict = json.load(f)

# 3. Hot-Swap to "Cinematic Structure" (80% CoMMU, 10% EMOPIA, 10% SLakh)
engine.set_weights({"commu_lora": 0.8, "emopia_lora": 0.1, "slakh_lora": 0.1})

prompt_ids = [
    master_dict["audio_key_cmajor"], master_dict["pitch_range_mid"], master_dict["num_measures_8"],
    master_dict["bpm_120"], master_dict["genre_cinematic"], master_dict["track_role_main_melody"],
    master_dict["inst_string_ensemble"], master_dict["sample_rhythm_standard"], master_dict["time_signature_4/4"],
    master_dict["min_velocity_50"], master_dict["max_velocity_100"]
]

midi_objs = engine.generate(prompt_ids, max_gen_len=256)
midi_objs[0].save("test_cinematic.mid")
print("🎶 Generated test_cinematic.mid!")

# 4. Hot-Swap to "Emotional Orchestral" (10% CoMMU, 40% EMOPIA, 50% SLakh)
# Notice how we DON'T reload the model. We just slide the faders!
engine.set_weights({"commu_lora": 0.1, "emopia_lora": 0.4, "slakh_lora": 0.5})

midi_objs_2 = engine.generate(prompt_ids, max_gen_len=256)
midi_objs_2[0].save("test_emotional.mid")
print("🎶 Generated test_emotional.mid!")