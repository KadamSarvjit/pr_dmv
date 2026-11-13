import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
df = pd.read_csv(url)

df = df[["Date", "AAPL.Close", "AAPL.Volume"]]
df.columns = ["Date", "Close", "Volume"]
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

print(df.head())
print(df.info())
print(df.describe())

df["Close"].plot(title="AAPL Closing Price", figsize=(10, 4))
plt.show()

df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df[["Close", "MA20", "MA50"]].plot(title="Moving Averages", figsize=(10, 4))
plt.show()

df.groupby(df.index.month)["Close"].mean().plot(kind="bar", title="Average Close by Month")
plt.show()

print(df[["Close", "Volume"]].corr())
df.plot(kind="scatter", x="Volume", y="Close", title="Close vs Volume")
plt.show()

train = df["Close"][:-30].reset_index(drop=True)
test = df["Close"][-30:].reset_index(drop=True)

model = ARIMA(train, order=(5, 1, 0)).fit()
forecast = model.forecast(steps=30)

plt.plot(test, label="Actual")
plt.plot(forecast, label="Forecast")
plt.title("ARIMA Forecast")
plt.legend()
plt.show()
