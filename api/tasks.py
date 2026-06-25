import os
import sys
import json
import traceback
from celery import Celery
from celery.exceptions import Ignore

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize Celery with Redis as the Broker and Backend
redis_url = "redis://localhost:6379/0"
celery_app = Celery("moonbeam_tasks", broker=redis_url, backend=redis_url)

# --- GLOBAL ENGINE INSTANCES (Loaded ONCE per Celery Worker process) ---
harmony_router = None
composer = None

@celery_app.task(bind=True, name="tasks.generate_song")
def generate_song_task(self, task_id: str, prompt: str, use_mock_llm: bool):
    global harmony_router, composer
    
    # Lazy load the heavy models on the first run of the worker
    if harmony_router is None:
        self.update_state(state='LOADING_MODELS', meta={'progress': 'Booting 839M Model & Rust TIES Core into VRAM...'})
        from engine.HarmonyRouter import HarmonyRouter
        from engine.agentic_composer import AgenticComposer
        
        # --- UPDATE THESE PATHS TO YOUR ACTUAL CHECKPOINTS ---
        BASE_MODEL_PATH = "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/moonbeam_checkpoint/moonbeam_839M.pt"
        LORA_DIR = "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/moonbeam_checkpoint/multi_task_lora"
        CONFIG_PATH = "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/model_config_multi_task.json"
        MASTER_DICT_PATH = "/home/aashishbishow/moon/Moonbeam-MIDI-Foundation-Model/processed/ComMU/indexed_tokens_dict.json"

        harmony_router = HarmonyRouter(
            base_model_path=BASE_MODEL_PATH, lora_checkpoint_dir=LORA_DIR,
            model_config_path=CONFIG_PATH, master_dict_path=MASTER_DICT_PATH, device="cuda"
        )
        composer = AgenticComposer(harmonyrouter=harmony_router, acceptance_threshold=0.75)
        os.makedirs("outputs", exist_ok=True)

    try:
        # 1. Brain & Planner
        self.update_state(state='PLANNING', meta={'progress': 'Brain is analyzing prompt & planning structure...'})
        composer.llm.use_mock = use_mock_llm
        
        llm_intent = composer.llm.generate_intent(prompt)
        blueprint = composer.planner.plan(llm_intent)
        
        # --- SAVE THE JSON BLUEPRINT (Your Request!) ---
        blueprint_path = f"outputs/{task_id}_blueprint.json"
        with open(blueprint_path, "w") as f:
            json.dump({"llm_intent": llm_intent, "dense_blueprint": blueprint}, f, indent=2)
            
        # 2. Agentic Loop (Generation -> Critic -> FAISS)
        self.update_state(state='COMPOSING', meta={'progress': 'Generating sections, running Critic & FAISS memory...'})
        final_song_midi = composer.compose_full_song(blueprint["timeline"])
        
        # 3. Save MIDI
        midi_path = f"outputs/{task_id}.mid"
        final_song_midi.write(midi_path)
        
        return {
            "status": "completed", 
            "midi_path": midi_path, 
            "blueprint_path": blueprint_path,
            "message": "Masterpiece rendered."
        }
        
    except Exception as e:
        traceback.print_exc()
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise Ignore() # Prevents Celery from auto-retrying and spamming the GPU