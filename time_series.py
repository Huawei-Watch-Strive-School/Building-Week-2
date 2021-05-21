import pandas as pd
import numpy as np
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
clf = Pipeline([('pre', preprocessor), ('classification', model)])

# pred = clf.predict(X)

# print(pred)
# print(f"Accuracy { accuracy_score(y, pred)*100}")
print(df.shape[0])
predictions = []

for i in range(0, 15):
    x_test = df.drop('target', axis=1).values[i].reshape(1, -1)
    x_test = pd.DataFrame(x_test, columns=df.drop('target', axis=1).columns)
    sleep(1)
    pred = clf.predict(x_test)
    if pred == 'Car':
        predictions.append('Car')
    elif pred == 'Still':
        predictions.append('Still')
    elif pred == 'Walking':
        predictions.append('Walking')

    if len(predictions) == 5:
        still_cont = 0
        walk_cont = 0
        car_cont = 0

        for p in predictions:
            if p == 'Still':
                still_cont += 1
            elif p == 'Car':
                car_cont += 1
            elif p == 'Walking':
                walk_cont += 1
            print(predictions)
            if still_cont >= 3:
                print('You are currently in still position')
                continue
            elif walk_cont >= 2:
                print('You are currently walking')
                continue
            elif car_cont >= 2:
                print('You are currently in a car')

                continue
            predictions.remove(predictions[0])
