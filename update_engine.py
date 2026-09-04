import yfinance as yf
import json
import os
from datetime import datetime

os.makedirs("data", exist_ok=True)

TICKERS = [
    "SWDY.CA", "ABUK.CA", "COMI.CA", "HRHO.CA", "TMGH.CA",
    "EAST.CA", "ETEL.CA", "FWRY.CA", "JUFO.CA", "ORWE.CA",
    "EKHO.CA", "ORAS.CA", "SKPC.CA", "AMOC.CA", "ESRS.CA"
]

results = []
for ticker in TICKERS:
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
        if data.empty:
            print(f"{ticker}: no data")
            continue
        close_col = data['Close']
        if len(close_col.shape) > 1:
            price = float(close_col.iloc[-1,0])
            prev = float(close_col.iloc[-2,0]) if len(close_col) > 1 else price
        else:
            price = float(close_col.iloc[-1])
            prev = float(close_col.iloc[-2]) if len(close_col) > 1 else price
            
        change = ((price - prev) / prev * 100) if prev != 0 else 0
        m14_score = 90 if change >= -1 else 65 if change >= -2 else 30
        status = "BUY" if m14_score >= 80 else "WAIT" if m14_score >= 50 else "BLOCK"
        
        results.append({
            "ticker": ticker,
            "name_ar": ticker.replace(".CA",""),
            "price": round(price, 2),
            "change_pct": round(change, 2),
            "m14_score": m14_score,
            "status": status,
            "last_update": datetime.now().isoformat(),
            "source": "EGX - Candle Close"
        })
        print(f"{ticker}: {price}")
    except Exception as e:
        print(f"{ticker} error: {e}")
        continue

output = {
    "last_update": datetime.now().isoformat(),
    "source": "M14 Motor - TradingView Close + 5min Live",
    "count": len(results),
    "prices": results
}

with open("data/live_prices.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results)}")
