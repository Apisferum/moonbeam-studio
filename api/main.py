import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from celery.result import AsyncResult

# Import the Celery app and task
from api.tasks import celery_app, generate_song_task

app = FastAPI(title="Moonbeam Distributed AI Record Label API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    use_mock_llm: bool = False

@app.post("/generate")
async def generate_music(req: GenerateRequest):
    task_id = str(uuid.uuid4())
    
    # Push task to Celery queue via Redis (Returns instantly!)
    task = generate_song_task.delay(task_id, req.prompt, req.use_mock_llm)
    
    return {
        "task_id": task_id, 
        "celery_task_id": task.id,
        "status": "queued", 
        "message": "Generation pushed to distributed worker queue."
    }

@app.get("/status/{celery_task_id}")
async def get_status(celery_task_id: str):
    # Query Celery/Redis for the task state
    task_result = AsyncResult(celery_task_id, app=celery_app)
    
    if task_result.state == 'PENDING':
        return {"status": "queued", "progress": "Waiting for available GPU worker..."}
    elif task_result.state in ['LOADING_MODELS', 'PLANNING', 'COMPOSING']:
        return {
            "status": "processing", 
            "stage": task_result.state,
            "progress": task_result.info.get('progress', 'Working...')
        }
    elif task_result.state == 'SUCCESS':
        return {
            "status": "completed", 
            "midi_download_url": f"/download/{os.path.basename(task_result.info['midi_path'])}",
            "blueprint_download_url": f"/download/{os.path.basename(task_result.info['blueprint_path'])}"
        }
    elif task_result.state == 'FAILURE':
        return {"status": "failed", "error": str(task_result.info)}
    else:
        return {"status": task_result.state}

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join("outputs", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    media_type = "audio/midi" if file_path.endswith(".mid") else "application/json"
    return FileResponse(file_path, media_type=media_type, filename=file_name)

@app.get("/")
async def root():
    return {"message": "Moonbeam Distributed AI Record Label API is running."}