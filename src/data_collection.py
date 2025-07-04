# src/data_collection.py

import yfinance as yf
import os

def download_stock_data(ticker="AAPL", start="2018-01-01", end="2023-12-31", save_path="data"):
    print(f"📥 Downloading {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        print("⚠️ No data found.")
        return

    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{ticker}.csv")
    df.to_csv(file_path)

    print(f"✅ Saved to {file_path}")
    return df

if __name__ == "__main__":
    download_stock_data("AAPL")
