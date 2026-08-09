import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from celery.result import AsyncResult

from api.tasks import celery_app, generate_song_task, OUTPUT_DIR

app = FastAPI(title="Moonbeam Distributed AI Record Label API")

# NOTE: wildcard origins + credentials is rejected by browsers per the CORS
# spec (Starlette will silently strip credentials rather than error).
# Pick a real origin allowlist if you need credentials, or drop credentials
# if you genuinely want wildcard access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # set True only alongside a concrete origin list
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_DOWNLOAD_EXTENSIONS = {".mid", ".json"}


class GenerateRequest(BaseModel):
    prompt: str
    use_mock_llm: bool = False


@app.post("/generate")
async def generate_music(req: GenerateRequest):
    task_id = str(uuid.uuid4())
    task = generate_song_task.delay(task_id, req.prompt, req.use_mock_llm)

    return {
        "task_id": task_id,
        "celery_task_id": task.id,
        "status": "queued",
        "message": "Generation pushed to distributed worker queue."
    }


@app.get("/status/{celery_task_id}")
async def get_status(celery_task_id: str):
    task_result = AsyncResult(celery_task_id, app=celery_app)

    if task_result.state == 'PENDING':
        return {"status": "queued", "progress": "Waiting for available GPU worker..."}
    elif task_result.state in ['LOADING_MODELS', 'PLANNING', 'COMPOSING']:
        # FIX: task_result.info can be None in edge cases — guard before .get()
        info = task_result.info or {}
        return {
            "status": "processing",
            "stage": task_result.state,
            "progress": info.get('progress', 'Working...')
        }
    elif task_result.state == 'SUCCESS':
        info = task_result.info or {}
        midi_path = info.get('midi_path')
        blueprint_path = info.get('blueprint_path')
        if not midi_path or not blueprint_path:
            return {"status": "failed", "error": "Task reported success but produced no output paths."}
        return {
            "status": "completed",
            "midi_download_url": f"/download/{os.path.basename(midi_path)}",
            "blueprint_download_url": f"/download/{os.path.basename(blueprint_path)}"
        }
    elif task_result.state == 'FAILURE':
        info = task_result.info
        error_msg = info.get('error') if isinstance(info, dict) else str(info)
        return {"status": "failed", "error": error_msg}
    else:
        return {"status": task_result.state}


@app.get("/download/{file_name}")
async def download_file(file_name: str):
    # FIX: path traversal guard. Reject anything that isn't a plain filename
    # resolving inside OUTPUT_DIR, and restrict to expected extensions.
    _, ext = os.path.splitext(file_name)
    if ext.lower() not in ALLOWED_DOWNLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    candidate = os.path.normpath(os.path.join(OUTPUT_DIR, file_name))
    if os.path.dirname(candidate) != os.path.normpath(OUTPUT_DIR):
        # normpath collapsed ".." or the request tried to escape OUTPUT_DIR
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "audio/midi" if candidate.endswith(".mid") else "application/json"
    return FileResponse(candidate, media_type=media_type, filename=file_name)


@app.get("/")
async def root():
    return {"message": "Moonbeam Distributed AI Record Label API is running."}