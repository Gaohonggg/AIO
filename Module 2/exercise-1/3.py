import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("advertising.csv")
data = df.to_numpy()

print( df["Sales"].max(), df["Sales"].idxmax() )
print( df["TV"].mean() )
print( df["Sales"].where( df["Sales"]>=20 ).count() )
print( df["Radio"].where( df["Sales"]>=15).mean() )

mean_of_news = df["Newspaper"].mean()
print( df["Sales"].where( df["Newspaper"]>=mean_of_news).sum() )

A = df["Sales"].mean()
score = df["Sales"].apply(
    lambda x:"Good" if x>A else("Bad" if x<A else "Average")
).to_numpy()

print( score[7:10] )