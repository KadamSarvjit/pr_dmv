import pandas as pd

url = "https://raw.githubusercontent.com/JetBrains/lets-plot-docs/master/data/Ames_house_prices_train.csv"
df = pd.read_csv(url)

print(df.shape)
print(df.head())
print(df.info())

df.columns = df.columns.str.lower().str.replace(" ", "_")
df = df.rename(columns={"saleprice": "sale_price", "grlivarea": "living_area", "yrsold": "year_sold"})

print(df.isnull().sum().sort_values(ascending=False).head(10))
df = df.dropna(axis=1, thresh=len(df) * 0.6)
df = df.fillna(df.median(numeric_only=True))
df = df.fillna("Unknown")

avg_price = df.groupby("neighborhood")["sale_price"].mean().reset_index(name="avg_price")
df = df.merge(avg_price, on="neighborhood")

subset = df[(df["year_sold"] >= 2008) & (df["bldgtype"] == "1Fam")]
print(subset.shape)

df["neighborhood_code"] = df["neighborhood"].astype("category").cat.codes
df = pd.get_dummies(df, columns=["bldgtype", "housestyle"], dtype=int)

print(df.groupby("neighborhood")["sale_price"].mean().sort_values(ascending=False))

print(df.shape)
q1 = df["sale_price"].quantile(0.25)
q3 = df["sale_price"].quantile(0.75)
iqr = q3 - q1
df = df[df["sale_price"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
print(df.shape)

df.to_csv("cleaned_real_estate.csv", index=False)
