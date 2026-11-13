import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

path = kagglehub.dataset_download("rohanrao/air-quality-data-in-india")
df = pd.read_csv(path + "/city_day.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df["Date"] = pd.to_datetime(df["Date"])
df = df[["City", "Date", "PM2.5", "PM10", "CO", "NO2", "SO2", "O3", "AQI", "AQI_Bucket"]]
df = df.dropna(subset=["AQI"])

delhi = df[df["City"] == "Delhi"].sort_values("Date").set_index("Date")
delhi["Year"] = delhi.index.year
delhi["Month"] = delhi.index.month

plt.figure(figsize=(12, 5))
plt.plot(delhi.index, delhi["AQI"], color="red", label="AQI")
plt.title("Delhi AQI Trend (2015-2020)")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, col, color in zip(axes, ["PM2.5", "PM10", "CO"], ["blue", "green", "orange"]):
    ax.plot(delhi.index, delhi[col], color=color, label=col)
    ax.set_ylabel(col)
    ax.legend()
    ax.grid(True)
axes[2].set_xlabel("Date")
fig.suptitle("Delhi Pollutant Levels Over Time")
plt.show()

monthly = delhi[delhi["Year"] == 2019].groupby("Month")["AQI"].mean()
monthly.plot(kind="bar", color="teal", figsize=(10, 5))
plt.title("Average Monthly AQI in Delhi (2019)")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.show()

order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
colors = ["green", "lightgreen", "yellow", "orange", "red", "darkred"]
counts = pd.crosstab(delhi["Year"], delhi["AQI_Bucket"]).reindex(columns=order, fill_value=0)
counts.plot(kind="bar", stacked=True, color=colors, figsize=(10, 6))
plt.title("Days in Each AQI Category per Year (Delhi)")
plt.xlabel("Year")
plt.ylabel("Number of Days")
plt.legend(title="AQI Category")
plt.show()

data = [delhi[delhi["AQI_Bucket"] == b]["AQI"] for b in order]
plt.figure(figsize=(10, 6))
plt.boxplot(data)
plt.xticks(range(1, len(order) + 1), order)
plt.title("AQI Distribution by AQI Category (Delhi)")
plt.xlabel("AQI Category")
plt.ylabel("AQI")
plt.grid(True)
plt.show()

top_cities = df["City"].value_counts().head(5).index
city_data = [df[df["City"] == c]["AQI"] for c in top_cities]
plt.figure(figsize=(10, 6))
plt.violinplot(city_data, showmedians=True)
plt.xticks(range(1, len(top_cities) + 1), top_cities)
plt.title("AQI Distribution Across Cities")
plt.xlabel("City")
plt.ylabel("AQI")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(delhi["PM2.5"], delhi["AQI"], color="purple", alpha=0.5)
plt.title("AQI vs PM2.5 (Delhi)")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.grid(True)
plt.show()

bubble = delhi.dropna(subset=["PM2.5", "PM10", "CO"])
plt.figure(figsize=(9, 6))
plt.scatter(bubble["PM2.5"], bubble["AQI"], s=bubble["CO"] * 20, c=bubble["PM10"],
            cmap="viridis", alpha=0.6)
plt.colorbar(label="PM10")
plt.title("AQI vs PM2.5 (bubble size = CO, color = PM10)")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.show()
