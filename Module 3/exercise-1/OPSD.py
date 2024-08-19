import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("opsd_germany_daily.csv",index_col=0,parse_dates=True)

df["Year"] = df.index.year
df["Month"] = df.index.month
df["Weekday Name"] = df.index.day_name()

sns.set(rc={"figure.figsize":(11,4)})
df["Consumption"].plot(linewidth=0.5)
plt.show()

cols_plot = ["Consumption","Solar","Wind"]
axes = df[cols_plot].plot(marker='.',alpha=0.5,linestyle="None",figsize=(11,9),subplots=True)
for ax in axes:
    ax.set_ylabel("Daily Totals (GWh)")
plt.show()

fig, axes = plt.subplots(3,1,figsize=(11,10),sharex=True)
for name, ax in zip(["Consumption","Solar","Wind"],axes):
    sns.boxplot(data=df,x="Month",y=name,ax=ax)
    ax.set_ylabel("Gwh")
    ax.set_title(name)
    if ax != axes[-1]:
        ax.set_xlabel("")
plt.show()

data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']
opsd_weekly_mean = df[data_columns].resample('W').mean()
start, end = '2017-01', '2017-06'
fig, ax = plt.subplots()
ax.plot(df.loc[start:end, 'Solar'],
        marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(opsd_weekly_mean.loc[start:end, 'Solar'],
        marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.set_ylabel('Solar Production (GWh)')
ax.legend()
plt.show()

opsd_7d = df[data_columns].rolling(7,center=True).mean()

import matplotlib.dates as mdates
opsd_365d = df[data_columns].rolling(window=365, center=True, min_periods=360).mean()
fig, ax = plt.subplots()
ax.plot(df['Consumption'], marker='.', markersize=2, color='0.6', linestyle='None', label='Daily')
ax.plot(opsd_7d['Consumption'], linewidth=2, label='7-d Rolling Mean')
ax.plot(opsd_365d['Consumption'], color='0.2', linewidth=3, label='Trend (365-d Rolling Mean)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Consumption (GWh)')
ax.set_title('Trends in Electricity Consumption')
plt.show()

fig, ax = plt.subplots()
for nm in ['Wind', 'Solar', 'Wind+Solar']:
    ax.plot(opsd_365d[nm], label=nm)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.set_ylim(0, 400)
ax.legend()
ax.set_ylabel('Production (GWh)')
ax.set_title('Trends in Electricity Production (365-d Rolling Means)')
plt.show()
