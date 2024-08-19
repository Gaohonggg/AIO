import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("IMDB-Movie-Data.csv")

print( df.info() )
print( df.describe() )
genre = df["Genre"]
print(genre)
some_cols = df[["Title","Genre","Actors","Director","Rating"]]
print( df.iloc[10:15][["Title","Rating"]] )
print( df[((df["Year"]>=2010) & (df["Year"]<=2015))
       & (df["Rating"]<6.0)
       & (df["Revenue (Millions)"] > df["Revenue (Millions)"].quantile(0.95))] )
print( df.groupby("Director")[["Rating"]].mean().head() )
print( df.groupby("Director")[["Rating"]].mean().sort_values(["Rating"],ascending=False) )
print( df.isnull().sum() )
m = df["Revenue (Millions)"].mean()
df["Revenue (Millions)"].fillna(m,inplace=True)

def rating(rating):
    if rating >= 7.5:
        return "Good"
    elif rating >= 6.0:
        return "Average"
    else:
        return "Bad"

df["Rating_category"] = df["Rating"].apply(rating)
print( df.head() )