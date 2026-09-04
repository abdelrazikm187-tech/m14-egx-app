import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="M14 V2.9 - EGX Live Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { direction: rtl; text-align: right; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .metric-card { background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; margin-bottom: 10px; }
    .status-buy { color: #00FF7F; font-weight: bold; font-size: 20px; }
    .status-wait { color: #FFD700; font-weight: bold; font-size: 20px; }
    .status-block { color: #FF4500; font-weight: bold; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

class M14Engine:
    def __init__(self, adx_th=22.025549):
        self.adx_th = adx_th

    def _rma(self, series, length):
        return series.ewm(alpha=1.0/length, adjust=False).mean()

    def calculate_indicators(self, df):
        df = df.copy()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = self._rma(gain, 14)
        avg_loss = self._rma(loss, 14)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
        df['rsi_vel'] = df['rsi'].diff()

        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = self._rma(df['tr'], 14)
        df['atrp'] = (df['atr'] / df['close']) * 100

        up = df['high'].diff()
        down = -df['low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        tr_rma = self._rma(df['tr'], 14)
        plus_rma = self._rma(pd.Series(plus_dm, index=df.index), 14)
        minus_rma = self._rma(pd.Series(minus_dm, index=df.index), 14)
        plus_di = np.where(tr_rma == 0, 0, 100 * plus_rma / tr_rma)
        minus_di = np.where(tr_rma == 0, 0, 100 * minus_rma / tr_rma)
        sum_di = plus_di + minus_di
        dx = np.where(sum_di == 0, 0, 100 * np.abs(plus_di - minus_di) / sum_di)
        df['adx'] = self._rma(pd.Series(dx, index=df.index), 14)

        df['ema20_dist'] = (df['close'] - df['ema20']) / df['ema20'] * 100
        df['ema50_dist'] = (df['close'] - df['ema50']) / df['ema50'] * 100
        hl_diff = (df['high'] - df['low']).replace(0, 1e-10)
        df['close_pos'] = (df['close'] - df['low']) / hl_diff
        return df

    def evaluate(self, df, symbol, name_ar):
        if len(df) < 200:
            return None

        df = self.calculate_indicators(df)
        n = len(df)
        
        last_sh, last_sl = None, None
        wave_low, wave_high = None, None
        last_sh_bar, last_sl_bar = None, None
        last_sh_price = None
        last_bos_bar = None
        smc_bullish = False
        in_down_state = False
        handoff_armed = True
        handoff_pending = False
        veto_active = False
        active_res = None
        prev_wave_pct = None
        wave_phase, fib_zone = "N/A", "N/A"
        
        for i in range(2, n):
            is_ph = (df['high'].iloc[i-2] > df['high'].iloc[i-4]) and (df['high'].iloc[i-2] > df['high'].iloc[i-3]) and (df['high'].iloc[i-2] > df['high'].iloc[i-1]) and (df['high'].iloc[i-2] > df['high'].iloc[i])
            is_pl = (df['low'].iloc[i-2] < df['low'].iloc[i-4]) and (df['low'].iloc[i-2] < df['low'].iloc[i-3]) and (df['low'].iloc[i-2] < df['low'].iloc[i-1]) and (df['low'].iloc[i-2] < df['low'].iloc[i])

            if is_ph:
                ph = df['high'].iloc[i-2]
                last_sh, last_sh_price, last_sh_bar, last_sl_bar = ph, ph, i-2, None
                if wave_low is not None: wave_high = ph
            if is_pl:
                pl = df['low'].iloc[i-2]
                last_sl, wave_low, last_sl_bar = pl, pl, i-2

            close_i = df['close'].iloc[i]
            bos_bull = (last_sh is not None) and (close_i > last_sh)
            bos_bear = (last_sl is not None) and (close_i < last_sl)

            if bos_bull:
                smc_bullish, last_bos_bar = True, i
                if last_sh is not None: wave_high = last_sh
            elif bos_bear:
                smc_bullish = False

            structure_age = (i - last_bos_bar) if last_bos_bar is not None else 999
            smc_map = smc_bullish and (structure_age <= 50)

            if (wave_high is not None) and (wave_low is not None) and (wave_high != wave_low):
                wave_pct = (close_i - wave_low) / (wave_high - wave_low) * 100
            else: wave_pct = None

            if wave_pct is not None:
                if wave_pct < 0: fib_zone, wave_phase = "<0%", "Below Wave"
                elif wave_pct < 23.6: fib_zone, wave_phase = "0–23.6%", "EARLY"
                elif wave_pct < 50: fib_zone, wave_phase = "23.6–50%", "DEVELOPING"
                elif wave_pct < 61.8: fib_zone, wave_phase = "50–61.8%", "EXPANSION"
                elif wave_pct < 78.6: fib_zone, wave_phase = "61.8–78.6%", "EXPANSION"
                elif wave_pct <= 100:
                    fib_zone = "78.6–100%"
                    wave_phase = "PULLBACK" if (prev_wave_pct and prev_wave_pct > 100 and wave_pct >= 61.8) else "LATE"
                else: fib_zone, wave_phase = ">100% Breakout", "BREAKOUT"

                if prev_wave_pct and prev_wave_pct >= 78.6 and 38.2 <= wave_pct < 78.6 and smc_map:
                    wave_phase = "PULLBACK"
                prev_wave_pct = wave_pct

            sh_age = (i - last_sh_bar) if last_sh_bar is not None else 999
            ll_count = 0
            if (last_sh_bar is not None) and (sh_age > 0):
                for k in range(min(sh_age, 20)):
                    idx = i - k
                    if idx > 0 and df['low'].iloc[idx] < df['low'].iloc[idx - 1]:
                        ll_count += 1

            price_vs_sh = ((close_i - last_sh_price) / last_sh_price * 100) if last_sh_price else 0.0
            sl_confirmed = (last_sl_bar is not None) and (last_sh_bar is not None) and (last_sl_bar > last_sh_bar)

            down_active_rt = (price_vs_sh <= -4.0 and df['ema20_dist'].iloc[i] < 3.0 and df['rsi_vel'].iloc[i] < 0 and df['close_pos'].iloc[i] < 0.55 and sh_age <= 20 and ll_count >= 2 and not sl_confirmed)
            highest_10 = df['high'].iloc[max(0, i-10):i].max() if i >= 10 else df['high'].iloc[i]
            down_end_raw = (df['high'].iloc[i] > highest_10 and close_i > df['ema20'].iloc[i] and df['rsi_vel'].iloc[i] > 0 and df['close_pos'].iloc[i] > 0.60)

            if down_active_rt: in_down_state = True
            elif in_down_state and down_end_raw: in_down_state = False

            market_clear = not in_down_state

            atrp_val, rsi_val = df['atrp'].iloc[i], df['rsi'].iloc[i]
            e20_dist, e50_dist, adx_val = df['ema20_dist'].iloc[i], df['ema50_dist'].iloc[i], df['adx'].iloc[i]

            atr_bin = 2 if atrp_val > 3.8 else (1 if atrp_val > 2.7 else 0)
            rsi_bin = 2 if rsi_val > 57.5 else (1 if rsi_val > 47.5 else 0)
            e20_bin = 2 if e20_dist > 2.0 else (1 if e20_dist > -0.8 else 0)
            e50_bin = 4 if e50_dist > 10.5 else (3 if e50_dist > 4.5 else (2 if e50_dist > -0.5 else (1 if e50_dist > -3.8 else 0)))

            up_strong = (close_i > df['ema200'].iloc[i]) and (df['ema20'].iloc[i] > df['ema50'].iloc[i]) and (adx_val > self.adx_th)
            up_weak = (close_i > df['ema200'].iloc[i]) and not up_strong
            regime = 0 if up_strong else (1 if up_weak else 2)

            c1 = (atr_bin==2 and rsi_bin==1 and e20_bin==1 and e50_bin==2)
            c2 = (atr_bin==0 and rsi_bin==1 and e20_bin==1 and e50_bin==2)
            c3 = (atr_bin==2 and rsi_bin==1 and e20_bin==2 and e50_bin==2 and regime==0)
            c4 = (atr_bin==1 and rsi_bin==1 and e20_bin==2 and e50_bin==2 and regime==0)
            c5 = (atr_bin==2 and rsi_bin==0 and e20_bin==1 and e50_bin==1)
            c6 = (atr_bin==2 and rsi_bin==0 and e20_bin==0 and e50_bin==0 and regime==0)
            c7 = (atr_bin==2 and rsi_bin==1 and e20_bin==0 and e50_bin==2 and regime==0)
            c8 = (atr_bin==2 and rsi_bin==2 and e20_bin==1 and e50_bin==3 and regime==0)
            c9 = (atr_bin==1 and rsi_bin==1 and e20_bin==2 and e50_bin==3 and regime==0)
            c10 = (atr_bin==2 and rsi_bin==1 and e20_bin==0 and e50_bin==3 and regime==0)
            c11 = (atr_bin==2 and rsi_bin==2 and e20_bin==2 and e50_bin==4 and regime==1)
            c12 = (atr_bin==1 and rsi_bin==2 and e20_bin==2 and e50_bin==4 and regime==1)
            c13 = (atr_bin==2 and rsi_bin==0 and e20_bin==0 and e50_bin==2 and regime==0)

            is_top13 = c1 or c2 or c3 or c4 or c5 or c6 or c7 or c8 or c9 or c10 or c11 or c12 or c13
            is_cont = (atr_bin >= 1) and (rsi_bin == 2) and (e20_bin >= 1) and (e50_bin >= 3) and (regime in [0, 1])
            entry_signal_raw = is_top13 or is_cont

            res_level = df['high'].iloc[max(0, i-50):i-1].max() if i > 50 else df['high'].iloc[0]
            touches = sum(1 for j in range(max(1, i-100), i) if abs(df['high'].iloc[j] - res_level) / res_level * 100 <= 0.5)
            dist_pct = (res_level - close_i) / close_i * 100
            has_res_raw = (0 <= dist_pct <= 1.0) and (touches >= 3) and (res_level > close_i) and (rsi_val < 60)

            if veto_active:
                if close_i > active_res or close_i < active_res * 0.98:
                    veto_active, active_res = False, None
                elif has_res_raw and res_level < active_res: active_res = res_level
            else:
                if entry_signal_raw and has_res_raw: veto_active, active_res = True, res_level

            v52_entry_signal = entry_signal_raw and (not veto_active) and (i > 200)
            if handoff_pending and down_end_raw: handoff_armed, handoff_pending = True, False
            allow_long = market_clear and handoff_armed

        curr = df.iloc[-1]
        decision = "BUY" if (v52_entry_signal and allow_long) else ("BLOCK" if in_down_state or veto_active else "WAIT")
        
        score = 0.0
        if v52_entry_signal: score += 40.0
        if allow_long: score += 20.0
        if market_clear: score += 15.0
        if smc_map: score += 10.0
        if wave_pct and 38.2 <= wave_pct <= 78.6: score += 10.0
        if curr['adx'] > self.adx_th: score += 5.0

        return {
            "df": df, "symbol": symbol, "name_ar": name_ar,
            "price": round(float(curr['close']), 2),
            "change_pct": round(float((curr['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100), 2),
            "decision": decision, "score": round(score, 1),
            "down_gate": "DOWN ACTIVE" if in_down_state else "CLEAR",
            "smc_map": "BULLISH" if smc_map else "BEARISH",
            "veto": "ACTIVE" if veto_active else "CLEAR",
            "active_res": round(float(active_res), 2) if active_res else None,
            "ema20": round(float(curr['ema20']), 2), "ema50": round(float(curr['ema50']), 2), "ema200": round(float(curr['ema200']), 2),
            "rsi": round(float(curr['rsi']), 1), "adx": round(float(curr['adx']), 1),
            "swing_high": round(float(last_sh), 2) if last_sh else None,
            "swing_low": round(float(last_sl), 2) if last_sl else None,
            "wave_pct": round(float(wave_pct), 1) if wave_pct else None,
            "fib_zone": fib_zone, "wave_phase": wave_phase
        }

EGX_STOCKS = [
    {"symbol": "COMI.CA", "name": "البنك التجاري الدولي"},
    {"symbol": "EAST.CA", "name": "الشرقية - إيسترن كومباني"},
    {"symbol": "HRHO.CA", "name": "مجموعة إي إف جي القابضة"},
    {"symbol": "TMGH.CA", "name": "طلعت مصطفى القابضة"},
    {"symbol": "SWDY.CA", "name": "السويدى إليكتريك"},
    {"symbol": "ABUK.CA", "name": "أبو قير للأسمدة"},
    {"symbol": "MFPC.CA", "name": "مصر لإنتاج السماد - موبكو"},
    {"symbol": "FWRY.CA", "name": "فورى لتكنولوجيا البنوك"},
    {"symbol": "ETEL.CA", "name": "المصرية للاتصالات"},
    {"symbol": "AMOC.CA", "name": "الإسكندرية للزيوت المعدنية"}
]

@st.cache_data(ttl=300)
def load_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y", interval="1d")
        if df.empty: return None
        df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[['open','high','low','close','volume']]
    except: return None

engine = M14Engine()

st.sidebar.title("🎮 M14 Control Panel")
page = st.sidebar.radio("الانتقال إلى:", ["📊 Scanner (المسح المباشر)", "🔬 Microscope (صفحة السهم Detail)", "ℹ عن M14 Engine"])

if page == "📊 Scanner (المسح المباشر)":
    st.title("📊 M14 V2.9 - EGX Market Scanner")
    st.caption("مسح مباشر للأسهم وترتيبها حسب قوة إشارة M14 Score - الأسعار من TradingView Close")

    filter_type = st.radio("الفلتر السريع:", ["ALL", "BUY 🟢", "WAIT 🟡", "BLOCK 🔴"], horizontal=True)

    results = []
    progress_bar = st.progress(0)
    
    for idx, item in enumerate(EGX_STOCKS):
        df = load_stock_data(item["symbol"])
        if df is not None:
            res = engine.evaluate(df, item["symbol"], item["name"])
            if res: results.append(res)
        progress_bar.progress((idx + 1) / len(EGX_STOCKS))
    progress_bar.empty()

    scanner_df = pd.DataFrame(results)
    if not scanner_df.empty:
        scanner_df = scanner_df.sort_values(by="score", ascending=False)

        if "BUY" in filter_type: scanner_df = scanner_df[scanner_df["decision"] == "BUY"]
        elif "WAIT" in filter_type: scanner_df = scanner_df[scanner_df["decision"] == "WAIT"]
        elif "BLOCK" in filter_type: scanner_df = scanner_df[scanner_df["decision"] == "BLOCK"]

        for _, row in scanner_df.iterrows():
            badge = "🟢 BUY" if row["decision"] == "BUY" else ("🟡 WAIT" if row["decision"] == "WAIT" else "🔴 BLOCK")
            
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 2])
                col1.subheader(f"{row['symbol']} - {row['name_ar']}")
                col2.metric("السعر الحالي", f"{row['price']} ج.م", f"{row['change_pct']}%")
                col3.metric("M14 Score", f"{row['score']} / 100")
                col4.markdown(f"### {badge}")
                col5.write(f"**Down Gate:** {row['down_gate']}")
                col5.write(f"**Fib Zone:** {row['fib_zone']}")
                st.divider()

elif page == "🔬 Microscope (صفحة السهم Detail)":
    st.title("🔬 Stock Microscope Detail View")
    
    selected_sym = st.selectbox("اختر السهم للتحليل العميق:", [item["symbol"] for item in EGX_STOCKS])
    name_ar = next(item["name"] for item in EGX_STOCKS if item["symbol"] == selected_sym)
    
    df = load_stock_data(selected_sym)
    if df is not None:
        res = engine.evaluate(df, selected_sym, name_ar)
        if res:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("السعر", f"{res['price']} ج.م", f"{res['change_pct']}%")
            c2.metric("القرار النهائي", res['decision'])
            c3.metric("Down Gate", res['down_gate'])
            c4.metric("SMC Structure", res['smc_map'])

            st.divider()

            st.subheader("📈 الرسم البياني التفاعلي + مستويات M14")
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(x=res['df'].index, open=res['df']['open'], high=res['df']['high'], low=res['df']['low'], close=res['df']['close'], name="OHLC"))
            
            fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['ema20'], line=dict(color='orange', width=1.5), name="EMA20"))
            fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['ema50'], line=dict(color='blue', width=1.5), name="EMA50"))
            fig.add_trace(go.Scatter(x=res['df'].index, y=res['df']['ema200'], line=dict(color='purple', width=1.5), name="EMA200"))

            if res['swing_high']:
                fig.add_hline(y=res['swing_high'], line_dash="dash", line_color="red", annotation_text="Swing High 100%")
            if res['swing_low']:
                fig.add_hline(y=res['swing_low'], line_dash="dash", line_color="green", annotation_text="Swing Low 0%")

            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.write("### 📐 المؤشرات الهيكلية")
                st.json({
                    "RSI (14)": res['rsi'],
                    "ADX (14)": res['adx'],
                    "EMA20": res['ema20'],
                    "EMA50": res['ema50'],
                    "Veto Resistance Status": res['veto'],
                    "Active Resistance": res['active_res']
                })
            with col_b:
                st.write("### 🌀 فيبوناتشي والموجات")
                st.json({
                    "Wave Percent": f"{res['wave_pct']}%",
                    "Fib Zone": res['fib_zone'],
                    "Wave Phase": res['wave_phase'],
                    "Last Swing High": res['swing_high'],
                    "Last Swing Low": res['swing_low']
                })

elif page == "ℹ عن M14 Engine":
    st.title("ℹ M14 V2.9 Frozen Core Engine")
    st.markdown("""
    * **Frozen Core:** محرك قرار موحد لا يتأثر بتعديلات الواجهة.
    * **Online Data:** الأسعار من TradingView Close {{close}} - Production Truth
    * **SMC Map:** تحديد صانع السوق وكسر الهيكل (BOS).
    * **Down Gate:** حارس حماية يمنع الدخول في الأسهم الهابطة حاداً.
    * **Fibonacci Engine:** تشخيص لحظي لمرحلة الموجة ومناطق التصحيح.
    * **Data Contract:** Historical CSV = Research only, Production = Online data
    """)
