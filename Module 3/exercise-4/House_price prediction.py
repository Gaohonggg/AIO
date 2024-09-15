import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("Housing.csv")
categorical_cols = df.select_dtypes(include=["object"]).columns.to_list()
print( categorical_cols )

encoder = OrdinalEncoder()
df[ categorical_cols ] = encoder.fit_transform(df[categorical_cols])

scaler = StandardScaler()
dataset_arr = scaler.fit_transform(df)
print( dataset_arr )
print("--------------------------------------------------------------------")

X = dataset_arr[:,1:]
y = dataset_arr[:,0]

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,random_state=1)

rdfr_regresstor = RandomForestRegressor(random_state=1)
rdfr_regresstor.fit(X_train,y_train)

adab_regresstor = AdaBoostRegressor(random_state=1)
adab_regresstor.fit(X_train,y_train)

grb_regresstor = GradientBoostingRegressor(random_state=1)
grb_regresstor.fit(X_train,y_train)

y_predict_rdfr = rdfr_regresstor.predict(X_test)
y_predict_adab = adab_regresstor.predict(X_test)
y_predict_grb = grb_regresstor.predict(X_test)

print("Result of random forest: ")
mae = mean_absolute_error(y_test,y_predict_rdfr)
mse = mean_squared_error(y_test,y_predict_rdfr)
print("MAE = {}    MSE = {}".format(mae,mse))

print("Result of adaboost: ")
mae = mean_absolute_error(y_test,y_predict_adab)
mse = mean_squared_error(y_test,y_predict_adab)
print("MAE = {}    MSE = {}".format(mae,mse))

print("Result of gradient boost: ")
mae = mean_absolute_error(y_test,y_predict_grb)
mse = mean_squared_error(y_test,y_predict_grb)
print("MAE = {}    MSE = {}".format(mae,mse))