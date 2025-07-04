import pandas as pd

# Load cleaned data
df = pd.read_csv("data/AAPL.csv", header=2)
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")

# Feature Engineering
df_feat = df[["Close"]].copy()

# Lag features (Close 1 day ago, 2 days ago...)
for i in range(1, 4):
    df_feat[f"Close_lag_{i}"] = df_feat["Close"].shift(i)

# Moving Averages
df_feat["MA7"] = df_feat["Close"].rolling(window=7).mean()
df_feat["MA14"] = df_feat["Close"].rolling(window=14).mean()
df_feat["MA30"] = df_feat["Close"].rolling(window=30).mean()

# Drop rows with NaN (from lag & rolling)
df_feat.dropna(inplace=True)

# Save processed features
df_feat.to_csv("data/features.csv")
print("✅ Features saved to data/features.csv")
