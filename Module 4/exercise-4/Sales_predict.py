import pandas as pd
import numpy as np

df = pd.read_csv("SalesPrediction.csv")
df = pd.get_dummies(df)

df = df.fillna( df.mean() )
X = df[["TV","Radio","Social Media","Influencer_Macro","Influencer_Mega",
        "Influencer_Micro","Influencer_Nano"]]
y = df[["Sales"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.33,
                                                    random_state=0)

from sklearn.preprocessing import StandardScaler
scaler =  StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.fit_transform(X_test)
print( scaler.mean_[0] )

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_processed)
X_test_poly = poly_features.fit_transform(X_test_processed)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score

poly_models = LinearRegression()
poly_models.fit(X_train_poly,y_train)
predict = poly_models.predict(X_test_poly)
print( r2_score(y_test,predict) )
