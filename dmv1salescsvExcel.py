import pandas as pd
import matplotlib.pyplot as plt

url = "https://huggingface.co/spaces/zenitsu09/insightpilot/resolve/main/data/sales_data_sample.csv"

data = pd.read_csv(url, encoding="latin1")
data.iloc[:1000].to_csv("sales1.csv", index=False)
data.iloc[1000:2000].to_excel("sales2.xlsx", index=False)
data.iloc[2000:].to_json("sales3.json", orient="records")

df_csv = pd.read_csv("sales1.csv")
df_excel = pd.read_excel("sales2.xlsx")
df_json = pd.read_json("sales3.json")

df = pd.concat([df_csv, df_excel, df_json], ignore_index=True)
print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.drop_duplicates()
df = df.fillna("Unknown")

df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"])
df["YEAR"] = df["ORDERDATE"].dt.year
df["MONTH"] = df["ORDERDATE"].dt.month
df["CONTACT_NAME"] = df["CONTACTFIRSTNAME"] + " " + df["CONTACTLASTNAME"]
df["REVENUE"] = df["QUANTITYORDERED"] * df["PRICEEACH"]

customer_total = df.groupby("CUSTOMERNAME")["SALES"].sum().reset_index(name="CUSTOMER_TOTAL")
df = df.merge(customer_total, on="CUSTOMERNAME")

print(df["SALES"].describe())
print("Total Sales:", df["SALES"].sum())
print("Average Order Value:", df.groupby("ORDERNUMBER")["SALES"].sum().mean())
print(df["PRODUCTLINE"].value_counts())
print(df.groupby("YEAR")["SALES"].sum())

df.groupby("PRODUCTLINE")["SALES"].sum().plot(kind="bar")
plt.title("Total Sales by Product Line")
plt.show()

df["PRODUCTLINE"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Product Category Distribution")
plt.ylabel("")
plt.show()

df.boxplot(column="SALES", by="DEALSIZE")
plt.title("Sales by Deal Size")
plt.suptitle("")
plt.show()
