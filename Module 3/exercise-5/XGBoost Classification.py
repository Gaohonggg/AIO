import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("Problem4.csv")

X = df.drop(columns=["Target"])
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=7)

xg_class = xgb.XGBClassifier(seed=7)
xg_class.fit(X_train,y_train)

preds = xg_class.predict(X_test)

accur_train = accuracy_score(y_train,xg_class.predict(X_train))
accur = accuracy_score(y_test,preds)

print("Accuracy train: {}".format(accur_train))
print("Accuracy: {}".format(accur))