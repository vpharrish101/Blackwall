import lightgbm 
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def m1_lgb(train_pth:str,
           test_pth:str,
           y_target:str,
           model_pth:str):

    lgb_clf=lightgbm.LGBMClassifier()
    train_csv=pd.read_csv(train_pth)
    test_csv=pd.read_csv(test_pth)

    X_train=train_csv.drop(y_target,axis=1)
    Y_train=train_csv[y_target]
    X_test=test_csv.drop(y_target,axis=1)
    Y_test=test_csv[y_target]

    lgb_clf.fit(X_train,Y_train)

    preds=lgb_clf.predict(X_test)
    acc=accuracy_score(Y_test,preds)    #type:ignore

    joblib.dump(lgb_clf,model_pth)
    return {
        "Accuracy":acc,
        "Model path":model_pth
    }



def m2_logreg(train_pth:str,
              test_pth:str,
              y_target:str,
              model_pth:str):

    logreg=LogisticRegression(max_iter=1000)
    train_csv=pd.read_csv(train_pth)
    test_csv=pd.read_csv(test_pth)

    X_train=train_csv.drop(y_target,axis=1)
    Y_train=train_csv[y_target]
    X_test=test_csv.drop(y_target,axis=1)
    Y_test=test_csv[y_target]

    logreg.fit(X_train,Y_train)

    preds=logreg.predict(X_test)
    acc=accuracy_score(Y_test,preds)

    joblib.dump(logreg,model_pth)

    return {
        "Accuracy":acc,
        "Model path":model_pth
    }



def m3_rf(train_pth:str,
          test_pth:str,
          y_target:str,
          model_pth:str):

    rf=RandomForestClassifier(n_estimators=100)
    train_csv=pd.read_csv(train_pth)
    test_csv=pd.read_csv(test_pth)

    X_train=train_csv.drop(y_target,axis=1)
    Y_train=train_csv[y_target]
    X_test=test_csv.drop(y_target,axis=1)
    Y_test=test_csv[y_target]

    rf.fit(X_train,Y_train)

    preds=rf.predict(X_test)
    acc=accuracy_score(Y_test,preds)

    joblib.dump(rf,model_pth)

    return {
        "Accuracy":acc,
        "Model path":model_pth
    }


def predict(model_pth:str,
            data_pth:str):

    model=joblib.load(model_pth)
    data=pd.read_csv(data_pth)
    preds=model.predict(data)
    probs=model.predict_proba(data)

    return {
        "predictions":preds.tolist(),
        "probabilities":probs.tolist()
    }