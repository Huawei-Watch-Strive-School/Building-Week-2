import pandas as pd
import numpy as np
import random
import pickle
from time import time, sleep
import joblib
from sklearn.preprocessing import MinMaxScaler, Normalizer, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import set_config, impute, compose
from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix, roc_auc_score

df = pd.read_csv('./data/sample.csv')
X = df.drop('target', axis=1)
y = df.target
preprocessor = joblib.load('./model/preprocessor.x')
model = joblib.load('./model/model.x')
X_test = preprocessor.transform(X)
pred = model.predict(X_test)

# print(pred)
# print(f"Accuracy { accuracy_score(y, pred)*100}")

predictions = []

for i in range(df.shape[0]):
    x_test = df.values[i]
    sleep(1)
    pred = model.predict(x_test)
    if pred == 'Car':
        predictions.append('Car')
    elif pred == 'Still':
        predictions.append('Still')
    elif pred == 'Walking':
        predictions.append('Walking')

    if len(predictions) == 3:
        still_cont = 0
        walk_cont = 0
        car_cont = 0

        for p in predictions:
            if p == 'Still':
                still_cont += 1
            elif p == 'Car':
                walk_cont += 1
            elif p == 'Walking':
                car_cont += 1

        if still_cont >= 2:
            print('You are currently in still position')
        elif walk_cont >= 2:
            print('You are currently walking')
        elif car_cont >= 2:
            print('You are currently in a car')
        predictions.remove(predictions[0])
