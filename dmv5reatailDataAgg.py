import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/pplonski/datasets-for-start/refs/heads/master/superstore-sales/superstore_dataset2011-2015.csv"

df = pd.read_csv(url, encoding="latin1")

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df = df[["Region", "Category", "Quantity", "Sales"]].dropna()

region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
print(region_sales)

region_sales.plot(kind="bar", color="skyblue", figsize=(10, 6))
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

region_sales.head(5).plot(kind="pie", autopct="%1.1f%%", figsize=(7, 7))
plt.title("Sales Share of Top 5 Regions")
plt.ylabel("")
plt.show()

print("Top Region:", region_sales.idxmax())
print(region_sales.head(3))

region_category = df.groupby(["Region", "Category"])["Sales"].sum().reset_index()
print(region_category)

pivot = region_category.pivot(index="Region", columns="Category", values="Sales")

pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
plt.title("Sales by Region and Category (Stacked)")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Category")
plt.tight_layout()
plt.show()

pivot.plot(kind="bar", figsize=(12, 6))
plt.title("Sales by Region and Category (Grouped)")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Category")
plt.tight_layout()
plt.show()
