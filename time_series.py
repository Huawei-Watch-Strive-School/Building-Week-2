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
# pred = model.predict(X_)
# print(f"Accuracy { accuracy_score(y, pred)*100}")

predictions = []

for row in range(df.shape[0]):
    sleep(1)
    pred = random.randrange(0, 3)
    pred = model.predict(X_test)
    if pred == 0:
        predictions.append(0)
    elif pred == 1:
        predictions.append(1)
    elif pred == 2:
        predictions.append(2)
    elif pred == 3:
        predictions.append(3)

    if predictions == 5:

        still_cont = 0
        walk_cont = 0
        car_cont = 0
        bus_cont = 0

        for p in predictions:
            if p == 0:
                still_cont += 1
            elif p == 1:
                walk_cont += 1
            elif p == 2:
                car_cont += 1
            elif p == 3:
                bus_cont += 1

        if still_cont >= 2:
            print('You are currently in still position')
        elif walk_cont >= 2:
            print('You are currently walking')
        elif car_cont >= 2:
            print('You are currently in a car')
        elif bus_cont >= 2:
            print('You are currently in a bu')

        predictions.remove(predictions[0])
