import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df = pd.read_csv(url)

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df.isnull().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print(df.duplicated().sum())
df = df.drop_duplicates()

text_cols = df.select_dtypes(include="object").columns.drop("customerID")
for c in text_cols:
    df[c] = df[c].str.strip().str.title()
df = df.replace({"No Internet Service": "No", "No Phone Service": "No"})

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")

for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
    q1 = df[c].quantile(0.25)
    q3 = df[c].quantile(0.75)
    iqr = q3 - q1
    df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                           labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"])

df = df.drop("customerID", axis=1)
df = pd.get_dummies(df, drop_first=True, dtype=int)

df.to_csv("cleaned_telecom_churn.csv", index=False)

X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "NumServices"]
X_train = X_train.copy()
X_test = X_test.copy()
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print(X_train.shape, X_test.shape)
print(y_train.value_counts(normalize=True))
