# -*- coding: utf-8 -*-

from __future__ import division
import datetime

import pandas as pd
import numpy as np
import xgboost as xgb
import os
# import lightgbm as lgb
from sklearn import svm
import pickle
import calendar
import datetime
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.externals import joblib
import pickle


os.chdir('/Users/jitaekim/Desktop/cse6250/final project/Final Code Files/Backend/CodeBackend')

def f(row):
    if row['avg'] > 0.8:
        val = 'very-critical'
    elif row['avg'] > 0.6 and row['avg'] <= 0.8 :
        val = 'critical'
    elif row['avg'] > 0.4 and row['avg'] <= 0.6 :
        val = 'moderate-critical'
    elif row['avg'] > 0.2 and row['avg'] <= 0.4 :
        val = 'pretty-safe'
    else:
        val = 'totally-safe'
    return val

def normalize(data): 
    for i in range(3, data.shape[1]):
        data.iloc[:, i] = np.log(data.iloc[:, i] + 0.00001)
    return data

# data_parent = pd.read_csv('Data/Input/scvo2_features_data_filtered.csv')
# data_parent = pd.read_csv('Data/Input/septic_patients_data.csv')
data_parent = pd.read_csv('Data/Test/sample_test_data.csv')

#data_parent = pd.read_csv('results.csv')
datax_temp = data_parent[['tissue_extraction', 'temp_fin', 'ph', 'hb', 'lactate']]
scaler = joblib.load("ModelFiles/scaler.save")
datax = scaler.transform(datax_temp)

xgb = pickle.load(open("ModelFiles/xgboost.dat", "rb"))
svm = pickle.load(open("ModelFiles/svm.dat", "rb"))
lr = pickle.load(open("ModelFiles/lr.dat", "rb"))
randomforest = pickle.load(open("ModelFiles/randomforest.dat", "rb"))

xgb_pred = pd.DataFrame(xgb.predict_proba(datax)[:, 1])
rf_pred = pd.DataFrame(randomforest.predict_proba(datax)[:, 1])

temp = pd.concat([xgb_pred, rf_pred], axis = 1)
temp['avg'] = temp.mean(axis = 1) 
combined_df = pd.concat([pd.DataFrame(data_parent[['subject_id', 'datetime']]), pd.DataFrame(datax_temp),  temp['avg']], axis = 1)

combined_df['patient_category'] = combined_df.apply(f, axis = 1) 

# critical_patients = combined_df.loc[combined_df['patient_category'].isin(['very-critical', 'critical', 'moderate-critical'])]
combined_df.to_csv('critical_patients_records.csv')

xgb_prob = pd.DataFrame(xgb.predict_proba(datax))
svm_prob = pd.DataFrame(svm.predict_proba(datax))
lr_prob = pd.DataFrame(lr.predict_proba(datax))
rf_prob = pd.DataFrame(randomforest.predict_proba(datax))

final_df = pd.concat([xgb_prob, rf_prob, lr_prob, svm_prob], axis=1)
final_df.to_csv("prob_predictions_model-level.csv")


print "#############"
print "testing done"
print "#############"

