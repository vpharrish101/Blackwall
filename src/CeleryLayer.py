import os

from celery import Celery
from models import m3_rf,m2_logreg,m1_lgb

app=Celery("tasks",
           broker="redis://localhost:6379/0",
           backend="redis://localhost:6379/0")


@app.task
def train_task(train_path:str, 
               val_path:str, 
               target:str, 
               model_type:str, 
               job_id:str):

    model_path=rf"D:\Python311\Pets\DIFra\models\{job_id}\model.pkl"
    os.makedirs(os.path.dirname(model_path),exist_ok=True)

    if model_type=="lgbm": result=m1_lgb(train_path,val_path,target,model_path)
    elif model_type=="logreg": result=m2_logreg(train_path,val_path,target,model_path)
    elif model_type=="rf": result=m3_rf(train_path,val_path,target,model_path)

    return result  #type:ignore