import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("Problem3.csv")

categorical_cols = df.select_dtypes(include=['object','bool']).columns.to_list()
print( categorical_cols )

for col_name in categorical_cols:
    n_categories = df[col_name].nunique()
    print(n_categories)

ordinal_encoder = OrdinalEncoder()
encoded_categorical_cols = ordinal_encoder.fit_transform(df[categorical_cols])

encoded_categorical_df = pd.DataFrame(
    encoded_categorical_cols,columns=categorical_cols
)

numerical_df = df.drop(categorical_cols,axis=1)
df = pd.concat([numerical_df,encoded_categorical_df],axis=1)

X = df.drop(columns=["area"])
y = df["area"]

X_train, X_test, y_train, y_test = train_test_split(X,y
                                                    ,test_size=0.3
                                                    ,random_state=7)

xg_reg = xgb.XGBRegressor(seed=7,
                          learning_rate=0.01,
                          n_estimators=102,
                          max_depth=3)
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

mae = mean_absolute_error(y_test,preds)
mse = mean_squared_error(y_test,preds)

print("MAE: {}".format(mae))
print("MSE: {}".format(mse))