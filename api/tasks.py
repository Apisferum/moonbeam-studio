import os
import sys
import json
import traceback
import torch
from celery import Celery
from celery.exceptions import Ignore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# FIX: read broker/backend from env, with the old hardcoded value only as a
# local-dev fallback — breaks otherwise the moment Redis isn't on localhost
# (Docker, remote broker, etc).
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("moonbeam_tasks", broker=REDIS_URL, backend=REDIS_URL)

# FIX: absolute, single-source output directory — main.py imports this same
# constant so both processes agree on where files actually live, regardless
# of each process's own working directory at launch.
OUTPUT_DIR = os.environ.get(
    "MOONBEAM_OUTPUT_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
)

# Optional: hard ceiling so a stuck generation doesn't block a worker forever.
celery_app.conf.task_time_limit = int(os.environ.get("MOONBEAM_TASK_TIME_LIMIT", 1800))       # hard kill
celery_app.conf.task_soft_time_limit = int(os.environ.get("MOONBEAM_TASK_SOFT_TIME_LIMIT", 1700))  # raises SoftTimeLimitExceeded first
celery_app.conf.worker_max_tasks_per_child = 2  # Recycle worker process to release CUDA/system memory completely

# --- GLOBAL ENGINE INSTANCES (Loaded ONCE per Celery Worker PROCESS) ---
# CRITICAL DEPLOYMENT NOTE: Celery's default pool is "prefork" — one OS
# process per configured concurrency slot, each with its own copy of these
# globals. On a single-GPU box (T4, RTX, etc.), running this worker with
# concurrency > 1 means EACH forked process independently loads its own
# full 839M model + LoRA adapters into VRAM the first time it picks up a
# task — an near-guaranteed OOM under any real concurrent load, and it will
# look like a random crash rather than an obvious config mistake.
#
# Run this worker as:
#   celery -A api.tasks worker --pool=solo
# (or --concurrency=1 with the default pool) so there is exactly one model
# instance per GPU. Do not raise concurrency unless you deliberately shard
# across multiple GPUs with per-worker device pinning.
harmony_router = None
composer = None


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@celery_app.task(bind=True, name="tasks.generate_song")
def generate_song_task(self, task_id: str, prompt: str, use_mock_llm: bool):
    global harmony_router, composer

    try:
        if harmony_router is None:
            self.update_state(state='LOADING_MODELS', meta={'progress': 'Booting 839M Model & Rust TIES Core into VRAM...'})
            from engine.HarmonyRouter import HarmonyRouter
            from engine.agentic_composer import AgenticComposer

            BASE_MODEL_PATH = "/home/aashishbishow/ProjectX/Moonbeam Pretrained Weightsmoonbeam_checkpoint/moonbeam_839M.pt"
            LORA_DIR = "/home/aashishbishow/ProjectX/moonbeam_chunk_20260716_140713"
            CONFIG_PATH = "/home/aashishbishow/ProjectX/moonbeam-codebase/src/llama_recipes/configs/model_config_multi_task.json"
            MASTER_DICT_PATH = "/home/aashishbishow/ProjectX/Moonbeam Multi-Task Data/ComMU/indexed_tokens_dict.json"

            # FIX: was hardcoded device="cuda" — bypassed HarmonyRouter's own
            # auto-detect/fallback entirely. Now defers to actual hardware.
            harmony_router = HarmonyRouter(
                base_model_path=BASE_MODEL_PATH, lora_checkpoint_dir=LORA_DIR,
                model_config_path=CONFIG_PATH, master_dict_path=MASTER_DICT_PATH,
                device=_resolve_device()
            )
            composer = AgenticComposer(harmonyrouter=harmony_router, acceptance_threshold=0.75)
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.update_state(state='PLANNING', meta={'progress': 'Brain is analyzing prompt & planning structure...'})
        composer.llm.use_mock = use_mock_llm

        llm_intent = composer.llm.generate_intent(prompt)
        blueprint = composer.planner.plan(llm_intent)

        blueprint_path = os.path.join(OUTPUT_DIR, f"{task_id}_blueprint.json")
        with open(blueprint_path, "w") as f:
            json.dump({"llm_intent": llm_intent, "dense_blueprint": blueprint}, f, indent=2)

        self.update_state(state='COMPOSING', meta={'progress': 'Generating sections, running Critic & FAISS memory...'})
        final_song_midi = composer.compose_full_song(blueprint["timeline"])

        midi_path = os.path.join(OUTPUT_DIR, f"{task_id}.mid")
        final_song_midi.write(midi_path)
        
        # Force resource cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return {
            "status": "completed",
            "midi_path": midi_path,
            "blueprint_path": blueprint_path,
            "message": "Masterpiece rendered."
        }

    except Exception as e:
        # FIX: model-loading failures are now caught here too (previously
        # sat outside the try block, so a bad checkpoint path or an OOM
        # during load would raise as a raw unhandled exception instead of
        # going through the same clean FAILURE-state path as generation
        # errors). harmony_router stays None on load failure either way,
        # so the next task on this worker will retry loading from scratch.
        traceback.print_exc()
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise Ignore()