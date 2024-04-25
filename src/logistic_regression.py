# Logistic Regression Algorithm:
import json
import os
import pickle
import pandas as pd
import numpy as np
import sys
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# parameters:
'''penalty{‘l1’, ‘l2’, ‘elasticnet’}, default=’l2’
dual bool, default=False
tol float, default=1e-4
C float, default=1.0
fit_intercept bool, default=True
intercept_scaling float, default=1.0
class_weight dict or ‘balanced’, default=None
random_state int, RandomState instance, default=None
solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
max_iter int, default=100
multi_class{‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
verbose int, default=0
warm_start bool, default=False
n_jobs int, default=None
l1_ratio float, default=None'''

def load():
    try:
        with open('conf.json','r')as file:
            con=json.load(file)
        return 1, con
    except:
        d1 = {'Status':'0', 'Error': 'No such file or Directory', 'Error_code': 'A345'}
        return 0,d1


def logistic_cls_model(con):
    try:
        df = pd.read_csv(con["logistic_reg_classification"]["input_file"])
        x = df[con["logistic_reg_classification"]["IV"]]
        y = df[con["logistic_reg_classification"]["DV"]]
        new_df = pd.concat([x, y], axis=1)
        min_required_rows = 100
        min_required_columns = 5
        if len(new_df) >= min_required_rows and len(new_df.columns) >= min_required_columns:
            x_support = x.apply(lambda row: all(isinstance(value, (int, float)) for value in row), axis=1).all()
            y_support = y.apply(lambda row: all(isinstance(value, (int, float)) for value in row), axis=1).all()
            if x_support == True and y_support == True:
                check_null1 = x.isna().any().any()
                check_null2 = y.isna().any().any()
                if check_null1 == False and check_null2 == False:
                    X = np.asarray(x)
                    Y = np.asarray(y)
                    h=con["logistic_reg_classification"]["class_weight"]
                    if h == "balanced" or h == 'balanced_subsample':
                        class_weights = h
                    elif h == 'None':
                        class_weights = None
                    else:
                        class_weights = eval(h)
                    d=con["logistic_reg_classification"]["dual"]
                    if d == 'True':
                        d1=True
                    elif d =='False':
                        d1=False
                    fit=con["logistic_reg_classification"]["fit_intercept"]
                    if fit =='True':
                        f1=True
                    elif fit =='False':
                        f1=False
                    warm=con["logistic_reg_classification"]["warm_start"]
                    if warm =='True':
                        w1=True
                    elif warm =='False':
                        w1=False
                    p1=con["logistic_reg_classification"]["penalty"]
                    if p1=='l1' or p1=='l2' or p1=='elasticnet':
                        p3=p1
                    elif p1 == 'None':
                        p3=None
                    else:
                        d1={'Status': '0',"Error":'penalty parameter should be None or (l1,l2,elasticnet)','Error_code': 'A619'}
                        return d1
                    r1=con["logistic_reg_classification"]["random_state"]
                    if r1=='None':
                        r2=None
                    elif r1.isdigit():
                        r2=int(r1)
                    else:
                        d1={'Status': '0',"Error":'random_state parameter should be None or integer','Error_code': 'A620'}
                        return d1
                    n=con["logistic_reg_classification"]["n_jobs"]
                    if n == 'None':
                        n1=None
                    elif n.isdigit():
                        n1=int(n)
                    else:
                        d1={'Status': '0',"Error":'n_jobs parameter should be None or integer','Error_code': 'A621'}
                        return d1
                    l1=con["logistic_reg_classification"]["l1_ratio"]
                    if l1 =='None':
                        l2=None
                    elif l1.replace('.', '', 1).isdigit():
                        l2=float(l2)
                    else:
                        d1={'Status': '0',"Error":'l1_ratio parameter should be None or float','Error_code': 'A622'}
                        return d1
                    logistic_classification = LogisticRegression(
                        penalty=p3,
                        dual=d1,
                        tol=float(con["logistic_reg_classification"]["tol"]),
                        C=float(con["logistic_reg_classification"]["C"]),
                        fit_intercept=f1,
                        intercept_scaling=float(con["logistic_reg_classification"]["intercept_scaling"]),
                        class_weight=class_weights,
                        random_state=r2,
                        solver=str(con["logistic_reg_classification"]["solver"]),
                        max_iter=int(con["logistic_reg_classification"]["max_iter"]),
                        multi_class=str(con["logistic_reg_classification"]["multi_class"]),
                        verbose=int(con["logistic_reg_classification"]["verbose"]),
                        warm_start=w1,
                        n_jobs=n1,
                        l1_ratio=l2
                    )
                    pipe = make_pipeline(StandardScaler(), logistic_classification)
                    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
                    for train_index, test_index in skf.split(X, Y):
                        x_train_fold, x_test_fold = X[train_index], X[test_index]
                        y_train_fold, y_test_fold = Y[train_index], Y[test_index]
                    
                    # model:
                    pipeline1 = pipe.fit(x_train_fold, y_train_fold)
                    
                    # #testing with X_train__fold:
                    y_pred = pipeline1.predict(x_train_fold)
                    confusion_matrix1 = confusion_matrix(y_train_fold, y_pred)
                    classification_report1 = classification_report(y_train_fold, y_pred,output_dict=True)
                    accuracy1 = accuracy_score(y_train_fold, y_pred) * 100
                    precision1 = precision_score(y_train_fold, y_pred, average='weighted') * 100
                    recall1 = recall_score(y_train_fold, y_pred, average='weighted') * 100
                    F1_score1 = f1_score(y_train_fold, y_pred, average='weighted') 
                    
                    # testing with X_test__fold:
                    y_pred = pipeline1.predict(x_test_fold)
                    confusion_matrix2 = confusion_matrix(y_test_fold, y_pred)
                    classification_report2 = classification_report(y_test_fold, y_pred,output_dict=True)
                    accuracy2 = accuracy_score(y_test_fold, y_pred) * 100
                    F1_score2 = f1_score(y_test_fold, y_pred, average='weighted') 
                    precision2 = precision_score(y_test_fold, y_pred, average='weighted') * 100
                    recall2 = recall_score(y_test_fold, y_pred, average='weighted') * 100
                    d = {
                        "Train_info":{
                            "Train_f1_score":F1_score1,
                            "Train_confusion_matrix":confusion_matrix1.tolist(),
                            "Train_classification_report":[classification_report1]
                            },
                        
                        "Test_info":{
                            "Test_f1_score":F1_score2,
                            "Test_confusion_matrix":confusion_matrix2.tolist(),
                            "Test_classification_report":[classification_report2]
                            }
                    }
                    if d["Test_info"]["Test_f1_score"] >= 0.5:
                        if con["logistic_reg_classification"]["model_generation"] == "Yes":
                            path = con["logistic_reg_classification"]["output_path"]
                            name = path + con["logistic_reg_classification"]["model_name"] + '.sav'
                            if os.path.exists(path):
                                pickle.dump(pipeline1, open(name, 'wb'))
                                pipeline1 = None
                                b = "Model Generated Successfully"
                                d1 = {"Status":'1', "Message":b, "Metrics":d, "download_path":name,"model_status":'1'}
                                return d1
                            else:
                                os.mkdir(path)
                                pickle.dump(pipeline1, open(name, 'wb'))
                                pipeline1 = None
                                b = "Model Generated Successfully"
                                d1 = {"Status":'1', "Message":b, "Metrics":d, "download_path":name,"model_status":'1'}
                                return d1
                        else:
                            b = "Please Ensure Model Generation option is selected"
                            d1 = {"Status":'1', "Message":b, "Metrics":d,"model_status":'0'}
                            return d1
                    else:
                        b = "Less Efficient F1_Score"
                        d1 = {"Status":'1', "Message":b, "Metrics":d,"model_status":'0'}
                        return d1
                else:
                    d1 = {'Status': '0', 'Error': 'Null values found in data', 'Error_code': 'A349'}
                    return  d1
            else:
                d1 = {'Status': '0', 'Error': 'Unsupported Data', 'Error_code': 'A348'}
                return  d1
        else:
            d1 = {'Status': '0', 'Error': 'Insufficient Data', 'Error_code': 'A347'}
            return d1
            
    except Exception as e:
        logger = logging.getLogger()
        logger.critical(e)
        d1 = {'Status': '0', 'Error': str(e), 'Error_code': 'A346'}
        return d1

if __name__ == "__main__":
    t,con=load()
    if t == 1:
        print(logistic_cls_model(con))
    else:
        print(con)
