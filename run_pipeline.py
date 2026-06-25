import json
import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Ensure the shared, engine, critic, and brain folders are in the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from engine.HarmonyRouter import HarmonyRouter
from engine.agentic_composer import AgenticComposer
from shared.music_theory_constants import logger

def main():
    load_dotenv() # Load Groq API key if needed
    
    parser = argparse.ArgumentParser(description="Moonbeam AI Record Label Pipeline")
    parser.add_argument("--prompt", type=str, default="Generate a 2-minute epic cinematic track. Starts very sad and sparse in A minor, then explodes into a massive, heroic orchestral chorus.", help="The user prompt")
    parser.add_argument("--output", type=str, default="masterpiece.mid", help="Output MIDI filename")
    parser.add_argument("--mock_llm", action="store_true", help="Use mock LLM instead of Groq API")
    args = parser.parse_args()

    # --- 1. DEFINE PATHS (Update these to your actual checkpoint paths!) ---
    BASE_MODEL_PATH = "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/moonbeam_checkpoint/moonbeam_839M.pt"
    LORA_DIR = "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/moonbeam_checkpoint/multi_task_lora"
    CONFIG_PATH = "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/model_config_multi_task.json"
    MASTER_DICT_PATH = "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/processed/ComMU/indexed_tokens_dict.json"

    # --- 2. BOOT THE PERFORMER (HarmonyRouter) ---
    print("\n🚀 Booting HarmonyRouter (Loading Base Model + Rust TIES Core)...")
    harmony_router = HarmonyRouter(
        base_model_path=BASE_MODEL_PATH,
        lora_checkpoint_dir=LORA_DIR,
        model_config_path=CONFIG_PATH,
        master_dict_path=MASTER_DICT_PATH,
        device="cuda" # Change to "cpu" if testing locally without GPU
    )

    # --- 3. BOOT THE CONDUCTOR (AgenticComposer) ---
    print("\n🎼 Initializing AgenticComposer (Brain + Critic + FAISS Memory)...")
    composer = AgenticComposer(
        harmonyrouter=harmony_router,
        acceptance_threshold=0.75, # Slightly lenient for first runs
    )
    
    # Override LLM mock setting based on CLI arg
    composer.llm.use_mock = args.mock_llm

    # --- 4. COMPOSE THE MASTERPIECE ---
    print(f"\n🎤 USER PROMPT: '{args.prompt}'\n")
    
    # The Brain generates the JSON, the Planner builds the timeline
    llm_intent = composer.llm.generate_intent(args.prompt)
    blueprint = composer.planner.plan(llm_intent)
    
    # --- NEW: SAVE THE BLUEPRINT TO DISK ---
    blueprint_filename = args.output.replace(".mid", "_blueprint.json")
    with open(blueprint_filename, "w") as f:
        json.dump({"llm_intent": llm_intent, "dense_blueprint": blueprint}, f, indent=2)
    print(f"📝 Song Blueprint saved to: {blueprint_filename}")
    # --------------------------------------    

    # The Conductor runs the Rejection Loop
    final_song_midi = composer.compose_full_song(blueprint["timeline"])

    # --- 5. SAVE TO DISK ---
    final_song_midi.write(args.output)
    print(f"\n🏆 SUCCESS! Final masterpiece saved to: {args.output}")

if __name__ == "__main__":
    main()