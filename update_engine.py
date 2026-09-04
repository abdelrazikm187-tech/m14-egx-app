
import yfinance as yf
import json
import pandas as pd
from datetime import datetime

# لستة اسهم EGX30 - ضيف براحتك
TICKERS = [
    "SWDY.CA","ABUK.CA","COMI.CA","HRHO.CA","TMGH.CA","EAST.CA",
    "ETEL.CA","FWRY.CA","JUFO.CA","ORWE.CA","EKHO.CA","ORAS.CA",
    "SKPC.CA","AMOC.CA","ESRS.CA"
]

prices = []
for t in TICKERS:
    try:
        ticker = yf.Ticker(t)
        hist = ticker.history(period="5d")
        if hist.empty:
            continue
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist)>1 else last
        price = float(last['Close'])
        change_pct = ((price - float(prev['Close'])) / float(prev['Close']) * 100) if prev['Close']!=0 else 0
        
        # M14 Score مبسط - انت هتستبدله بحسابك الحقيقي
        score = 90 if change_pct > -1 else 65 if change_pct > -2 else 30
        status = "BUY" if score >= 80 else "WAIT" if score >= 50 else "BLOCK"
        
        prices.append({
            "ticker": t,
            "name_ar": t.split('.')[0],
            "price": round(price,2),
            "change_pct": round(change_pct,2),
            "m14_score": score,
            "status": status,
            "last_update": datetime.now().isoformat(),
            "high": round(float(hist['High'].max()),2),
            "low": round(float(hist['Low'].min()),2)
        })
    except Exception as e:
        print(f"Error {t}: {e}")

output = {
    "last_update": datetime.now().isoformat(),
    "source": "M14 Motor V2.9 - Real Prices",
    "count": len(prices),
    "prices": prices
}

with open("data/live_prices.json","w",encoding="utf-8") as f:
    json.dump(output,f,ensure_ascii=False,indent=2)

print(f"Saved {len(prices)} prices")
