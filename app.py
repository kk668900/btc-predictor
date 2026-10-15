import streamlit as st
import pandas as pd
import numpy as np
import requests
import feedparser
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

st.set_page_config(page_title="BTC 智能预测站", page_icon="₿", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; max-width: 1100px;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("₿ BTC 智能预测站")
st.caption("免费公开行情 · 1小时方向估算 · 新闻情绪 · 模拟交易（不连接交易所）")

SYMBOL = "BTCUSDT"
BINANCE_URL = "https://api.binance.com/api/v3/klines"

@st.cache_data(ttl=60)
def fetch_klines(limit=1000):
    r = requests.get(BINANCE_URL, params={"symbol": SYMBOL, "interval": "5m", "limit": limit}, timeout=15)
    r.raise_for_status()
    raw = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["time","open","high","low","close","volume"]].dropna().sort_values("time").reset_index(drop=True)

def make_features(df):
    x = df.copy()
    x["ret1"] = x["close"].pct_change()
    x["ret3"] = x["close"].pct_change(3)
    x["ret12"] = x["close"].pct_change(12)
    x["volatility"] = x["ret1"].rolling(12).std()
    x["ema12"] = x["close"].ewm(span=12, adjust=False).mean()
    x["ema26"] = x["close"].ewm(span=26, adjust=False).mean()
    x["ema_gap"] = (x["ema12"] / x["ema26"]) - 1
    delta = x["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_gap"] = x["macd"] - x["macd_signal"]
    x["volume_change"] = x["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    # Future 1h direction: six 5-minute candles ahead
    x["target"] = (x["close"].shift(-6) > x["close"]).astype(int)
    features = ["ret1","ret3","ret12","volatility","ema_gap","rsi","macd_gap","volume_change"]
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["target"]).copy()
    return x, features

@st.cache_data(ttl=300)
def fetch_news():
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    items = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            for e in parsed.entries[:20]:
                title = str(e.get("title","")).strip()
                if title:
                    items.append({"title": title, "link": e.get("link",""), "published": e.get("published","")})
        except Exception:
            pass
    # Simple transparent keyword score, not a language model
    positive = ["surge","rally","bull","gain","rise","approval","adoption","突破","上涨","利好","增长","买入"]
    negative = ["drop","fall","crash","bear","loss","hack","lawsuit","ban","下跌","暴跌","利空","监管","抛售"]
    for item in items:
        t = item["title"].lower()
        p = sum(w in t for w in positive)
        n = sum(w in t for w in negative)
        item["score"] = (p-n) / max(1, p+n)
    return items

def money(v):
    return f"${v:,.2f}"

try:
    df = fetch_klines()
    if len(df) < 200:
        st.error("行情数据不足，请稍后重试。")
        st.stop()
except Exception as e:
    st.error("暂时无法获取 Binance 免费公开行情。请检查网络后点击左侧“重新运行”，或稍后重试。")
    st.caption(f"技术信息：{type(e).__name__}: {e}")
    st.stop()

feat_df, feature_cols = make_features(df)
# Keep the last six rows out of training because their future labels are not observable.
train_df = feat_df.iloc[:-6].copy()
latest_row = feat_df.iloc[-1]
if len(train_df) < 100:
    st.error("有效历史样本不足，暂时不能训练模型。")
    st.stop()

# Time-ordered holdout; no random shuffle.
split = int(len(train_df) * 0.8)
X_train = train_df[feature_cols].iloc[:split]
y_train = train_df["target"].iloc[:split]
X_test = train_df[feature_cols].iloc[split:]
y_test = train_df["target"].iloc[split:]
model = RandomForestClassifier(n_estimators=180, max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
test_acc = accuracy_score(y_test, model.predict(X_test)) if len(X_test) else float("nan")
proba = model.predict_proba(latest_row[feature_cols].to_frame().T)[0]
classes = list(model.classes_)
up_prob = float(proba[classes.index(1)]) if 1 in classes else 0.5
down_prob = 1 - up_prob

news = fetch_news()
news_score = float(np.mean([n["score"] for n in news])) if news else 0.0
# Blend model with a small technical/news adjustment; probabilities remain heuristic.
rsi = float(latest_row["rsi"])
ema_gap = float(latest_row["ema_gap"])
macd_gap = float(latest_row["macd_gap"])
technical_bias = np.clip(ema_gap * 18 + np.tanh(macd_gap / max(float(latest_row["close"])*0.0005, 1e-9))*0.08, -0.12, 0.12)
news_bias = np.clip(news_score * 0.05, -0.05, 0.05) if news else 0.0
combined_up = float(np.clip(up_prob + technical_bias + news_bias, 0.05, 0.95))
current_price = float(df["close"].iloc[-1])
# Empirical 1h absolute move estimate from recent six-bar returns, displayed as a rough range.
hourly_returns = df["close"].pct_change(6).dropna().tail(200)
move = float(hourly_returns.std()) if len(hourly_returns) else 0.01
move = max(move, 0.001)
low_est = current_price * (1 - 1.5*move)
high_est = current_price * (1 + 1.5*move)

with st.sidebar:
    st.header("控制面板")
    if st.button("🔄 刷新行情与新闻", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("行情：Binance 公共现货 K 线（BTCUSDT，5分钟）")
    st.caption("预测周期：未来约1小时")
    st.caption("不连接交易所账户，不会执行真实订单。")
    st.divider()
    st.subheader("模拟账户")
    if "cash" not in st.session_state:
        st.session_state.cash = 10000.0
    if "btc" not in st.session_state:
        st.session_state.btc = 0.0
    if "trades" not in st.session_state:
        st.session_state.trades = []
    if "entry_price" not in st.session_state:
        st.session_state.entry_price = None
    starting_cash = st.number_input("初始虚拟资金（美元）", min_value=100.0, max_value=10000000.0, value=10000.0, step=500.0)
    fee_pct = st.number_input("模拟手续费（单边 %）", min_value=0.0, max_value=2.0, value=0.1, step=0.05) / 100
    if st.button("重置模拟账户", use_container_width=True):
        st.session_state.cash = float(starting_cash)
        st.session_state.btc = 0.0
        st.session_state.trades = []
        st.session_state.entry_price = None
        st.rerun()

m1, m2, m3 = st.columns(3)
m1.metric("BTC 当前参考价", money(current_price))
m2.metric("模型估算上涨概率", f"{combined_up*100:.1f}%")
m3.metric("时间顺序留出集准确率", f"{test_acc*100:.1f}%" if not np.isnan(test_acc) else "暂无")

direction = "偏多" if combined_up >= 0.56 else ("偏空" if combined_up <= 0.44 else "中性 / 观望")
st.subheader(f"未来 1 小时：{direction}")
st.write(f"模型方向概率：上涨 **{up_prob*100:.1f}%**，下跌 **{down_prob*100:.1f}%**。加入技术与新闻的启发式调整后，上涨估算为 **{combined_up*100:.1f}%**。")
st.caption(f"粗略波动参考区间（不是价格目标或保证）：{money(low_est)} 至 {money(high_est)}。新闻情绪评分：{news_score:+.2f}；抓取到 {len(news)} 条标题。")

tab1, tab2, tab3, tab4 = st.tabs(["行情图表", "指标与新闻", "模拟交易", "模型说明"])
with tab1:
    chart_df = df.tail(240)
    fig = go.Figure(data=[go.Candlestick(x=chart_df["time"], open=chart_df["open"], high=chart_df["high"], low=chart_df["low"], close=chart_df["close"], name="BTC/USDT")])
    fig.update_layout(height=420, margin=dict(l=5,r=5,t=25,b=5), xaxis_rangeslider_visible=False, title="BTC/USDT · 5分钟K线")
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    a,b,c,d = st.columns(4)
    a.metric("RSI(14)", f"{rsi:.1f}")
    b.metric("EMA12 / EMA26差", f"{ema_gap*100:+.3f}%")
    c.metric("MACD差值", f"{macd_gap:.2f}")
    d.metric("新闻情绪", f"{news_score:+.2f}" if news else "不可用")
    st.markdown("**近期新闻标题（关键词情绪评分，不代表事实判断）**")
    if news:
        for n in news[:12]:
            if n["link"]:
                st.markdown(f"- [{n['title']}]({n['link']}) · 情绪 {n['score']:+.1f}")
            else:
                st.write(f"- {n['title']} · 情绪 {n['score']:+.1f}")
    else:
        st.info("当前未抓取到新闻。模型不会伪造新闻数据，新闻情绪调整按中性处理。")
with tab3:
    st.write(f"虚拟现金：**{money(st.session_state.cash)}**  |  BTC持仓：**{st.session_state.btc:.6f} BTC**")
    equity = st.session_state.cash + st.session_state.btc * current_price
    st.write(f"账户总资产：**{money(equity)}**  |  相对初始账户净值盈亏：**{money(equity - 10000)}**（以 $10,000 为默认基准）")
    max_buy = st.session_state.cash / (1 + fee_pct)
    buy_amount = st.number_input("本次模拟买入金额（美元）", min_value=10.0, max_value=1000000.0, value=100.0, step=10.0, key="buy_amount")
    sell_pct = st.slider("卖出持仓比例（%）", min_value=1, max_value=100, value=100)
    c1,c2 = st.columns(2)
    with c1:
        if st.button("模拟买入", use_container_width=True):
            spend = min(float(buy_amount), st.session_state.cash / (1 + fee_pct))
            if spend >= 10:
                qty = spend / current_price
                fee = spend * fee_pct
                st.session_state.cash -= spend + fee
                st.session_state.btc += qty
                st.session_state.trades.append({"时间": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "操作":"模拟买入", "价格":current_price, "数量BTC":qty, "手续费":fee})
                st.rerun()
            else:
                st.warning("虚拟现金不足。")
    with c2:
        if st.button("模拟卖出", use_container_width=True):
            qty = st.session_state.btc * sell_pct / 100
            if qty > 0:
                proceeds = qty * current_price
                fee = proceeds * fee_pct
                st.session_state.cash += proceeds - fee
                st.session_state.btc -= qty
                st.session_state.trades.append({"时间": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "操作":"模拟卖出", "价格":current_price, "数量BTC":qty, "手续费":fee})
                st.rerun()
            else:
                st.warning("当前没有 BTC 持仓。")
    if st.session_state.trades:
        st.dataframe(pd.DataFrame(st.session_state.trades).iloc[::-1], use_container_width=True, hide_index=True)
        csv = pd.DataFrame(st.session_state.trades).to_csv(index=False).encode("utf-8-sig")
        st.download_button("导出交易记录 CSV", data=csv, file_name="btc_paper_trades.csv", mime="text/csv")
    else:
        st.info("暂无模拟交易记录。所有操作只修改本网页的虚拟账户。")
with tab4:
    st.markdown("""
- **数据**：Binance 公共行情接口，无需 API 密钥。
- **特征**：短期收益率、波动率、EMA差、RSI、MACD差、成交量变化。
- **模型**：随机森林分类器，按时间顺序切分训练集与测试集，不随机打乱。
- **目标**：当前收盘价与约1小时后的收盘价比较，预测方向。
- **新闻**：读取公开 RSS 标题，使用简单中英关键词评分；不是大型语言模型，也可能漏判讽刺或语境。
- **准确率**：只是一次时间顺序留出集的方向准确率，不等于未来胜率，也未充分模拟交易成本。
- **重要限制**：概率经过模型估计但未做严格校准；技术/新闻融合为透明的启发式调整。使用前应进一步做滚动回测、基准比较与手续费/滑点测试。
- **隐私与安全**：不要求交易所密钥，不执行真实买卖。模拟账户存在浏览器会话中，重启后可能清空。
""")

st.divider()
st.caption("仅用于学习与模拟研究，不构成投资建议。加密资产波动极高，模型可能连续判断错误。")
