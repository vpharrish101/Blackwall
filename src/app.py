import joblib
import pandas as pd
import uuid

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from celery.result import AsyncResult
from Pets.Blackwall.src.CeleryLayer import train_task

model=FastAPI(title="DIFra")


class TrainRequest(BaseModel):
    train_path:str
    val_path:str
    target:str
    model_type:str

class PredictRequest(BaseModel):
    model_path:str
    data_path:str


@model.get("/get/health")
def health_chk():
    return {"Status":"OK",
            "Message":"Backend up and running"}


@model.get("/get/model/job/{job_id}")
def status_chk(job_id:str):
    result=AsyncResult(
        id=job_id
        )
    return {
        "job_id":job_id,
        "status":result.status,
        "result":result.result
        }


@model.post("/post/train")
def train_api(req:TrainRequest):
    job_id=str(uuid.uuid4())
    job=train_task.delay( #type:ignore
        req.train_path,
        req.val_path,
        req.target,
        req.model_type,
        job_id
        )

    return {
        "job_id":job_id,
        "status":"queued"
        }


@model.post("/post/predict")
def predict_api(req:PredictRequest):
    try:
        model_obj=joblib.load(req.model_path)
        data=pd.read_csv(req.data_path)
        preds=model_obj.predict(data)

        return {
            "predictions":preds.tolist()
            }

    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))