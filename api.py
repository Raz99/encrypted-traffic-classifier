# FastAPI endpoints to queue inference jobs via Celery and poll results
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
from celery import Celery
from celery.result import AsyncResult

app = FastAPI()

# Setup Celery connection
celery_app = Celery('tasks', 
                    broker='redis://redis:6379/0', 
                    backend='redis://redis:6379/0')

@app.post("/predict/{task_type}")
async def predict(task_type: str, file: UploadFile = File(...)):
    """Upload a CSV, enqueue an inference task, return Celery task_id."""
    contents = await file.read() # Raw uploaded bytes
    df = pd.read_csv(io.BytesIO(contents)) # CSV to DataFrame
    
    data_json = df.to_json(orient='split') # Convert DataFrame to JSON for worker
    task = celery_app.send_task("tasks.run_inference", args=[data_json, task_type]) # Enqueue task
    
    return {"task_id": task.id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """Return task status; include results when finished."""
    res = AsyncResult(task_id, app=celery_app) # Get task result
    if res.ready():
        return {"status": "SUCCESS", "results": res.result}
    return {"status": "PENDING"}