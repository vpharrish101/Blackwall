import xgboost as xgb
import numpy as np

def train_child(X,y):
    y=np.asarray(y).astype(int)
    model=xgb.XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        base_score=0.5,
        tree_method="hist"
    )
    model.fit(X,y)
    return model

def predict_child(model,X):
    return model.predict_proba(X)[:,1]
