import streamlit as st
import pandas as pd
import numpy as np
import requests
import feedparser
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="BTC 实时预测与杠杆模拟站", page_icon="₿", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1rem; max-width: 1200px;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("₿ BTC 实时预测与杠杆模拟站")
st.caption("实时参考价 · 1小时方向估算 · 新闻情绪 · 多空杠杆模拟（仅虚拟资金，不连接交易所）")

# Rerun the page every five seconds. Live ticker is fetched each rerun; candles/model are cached separately.
refresh_count = st_autorefresh(interval=5_000, limit=None, key="btc_live_auto_refresh")
st.caption(f"自动刷新：每 5 秒 · 本次会话刷新次数：{refresh_count}")

TICKER_URL = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
FEATURES = ["ret1", "ret3", "ret12", "volatility", "ema_gap", "rsi", "macd_gap", "volume_change"]


@st.cache_data(ttl=4)
def fetch_live_ticker():
    response = requests.get(
        TICKER_URL,
        headers={"Accept": "application/json", "User-Agent": "BTC-Paper-Trader/1.0"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    price = float(payload["price"])
    if not np.isfinite(price) or price <= 0:
        raise ValueError("实时价格无效")
    return price, payload.get("time", "")


@st.cache_data(ttl=60)
def fetch_klines():
    response = requests.get(
        CANDLES_URL,
        params={"granularity": 300},
        headers={"Accept": "application/json", "User-Agent": "BTC-Paper-Trader/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list) or not raw:
        raise ValueError("Coinbase 没有返回有效 K 线数据")
    df = pd.DataFrame(raw, columns=["timestamp", "low", "high", "open", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = (
        df[["time", "open", "high", "low", "close", "volume"]]
        .replace([np.inf, -np.inf], np.nan).dropna()
        .sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    )
    if len(df) < 200:
        raise ValueError(f"只有 {len(df)} 根有效K线，模型至少需要 200 根。")
    return df


def make_features(df):
    x = df.copy()
    x["ret1"] = x["close"].pct_change()
    x["ret3"] = x["close"].pct_change(3)
    x["ret12"] = x["close"].pct_change(12)
    x["volatility"] = x["ret1"].rolling(12).std()
    x["ema12"] = x["close"].ewm(span=12, adjust=False).mean()
    x["ema26"] = x["close"].ewm(span=26, adjust=False).mean()
    x["ema_gap"] = x["ema12"] / x["ema26"] - 1
    delta = x["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    x["macd"] = x["ema12"] - x["ema26"]
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_gap"] = x["macd"] - x["macd_signal"]
    x["volume_change"] = x["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    x["target"] = (x["close"].shift(-6) > x["close"]).astype(float)
    x.loc[x.index[-6:], "target"] = np.nan
    x = x.replace([np.inf, -np.inf], np.nan)
    usable = x.dropna(subset=FEATURES + ["target"]).copy()
    latest = x.dropna(subset=FEATURES).iloc[-1]
    return usable, latest


@st.cache_data(ttl=300)
def fetch_news():
    feeds = ["https://www.coindesk.com/arc/outboundfeeds/rss/", "https://cointelegraph.com/rss"]
    positive = ["surge", "rally", "bull", "gain", "rise", "approval", "adoption", "突破", "上涨", "利好", "增长", "买入"]
    negative = ["drop", "fall", "crash", "bear", "loss", "hack", "lawsuit", "ban", "下跌", "暴跌", "利空", "监管", "抛售"]
    items = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:20]:
                title = str(entry.get("title", "")).strip()
                if title:
                    items.append({"title": title, "link": entry.get("link", ""), "published": entry.get("published", "")})
        except Exception:
            pass
    for item in items:
        title = item["title"].lower()
        p = sum(word in title for word in positive)
        n = sum(word in title for word in negative)
        item["score"] = (p - n) / max(1, p + n)
    return items


def money(value):
    return f"${value:,.2f}"


def liq_price(entry, side, leverage, maintenance_rate=0.005):
    # Simplified educational estimate, not an exchange's actual liquidation formula.
    distance = max((1.0 / leverage) - maintenance_rate, 0.001)
    return entry * (1 - distance) if side == "做多" else entry * (1 + distance)


# Initialize paper account state.
if "cash" not in st.session_state:
    st.session_state.cash = 10000.0
if "position" not in st.session_state:
    st.session_state.position = None
if "trades" not in st.session_state:
    st.session_state.trades = []
if "last_live_price" not in st.session_state:
    st.session_state.last_live_price = None

with st.sidebar:
    st.header("控制面板")
    st.caption("实时价：Coinbase BTC-USD ticker；K线：5分钟公开数据")
    st.caption("本页面仅模拟交易，不会向交易所下单。")
    st.divider()
    st.subheader("虚拟账户")
    starting_cash = st.number_input("重置账户资金（美元）", min_value=100.0, max_value=10000000.0, value=10000.0, step=500.0)
    fee_pct = st.number_input("模拟手续费（单边 %）", min_value=0.0, max_value=1.0, value=0.05, step=0.01) / 100
    leverage = st.slider("模拟杠杆倍数", min_value=1, max_value=100, value=100, step=1)
    margin_input = st.number_input("本次开仓保证金（美元）", min_value=10.0, max_value=1000000.0, value=100.0, step=10.0)
    if st.button("重置模拟账户", use_container_width=True):
        st.session_state.cash = float(starting_cash)
        st.session_state.position = None
        st.session_state.trades = []
        st.rerun()

try:
    live_price, ticker_time = fetch_live_ticker()
    st.session_state.last_live_price = live_price
except Exception as exc:
    live_price = st.session_state.last_live_price
    if live_price is None:
        st.error("无法取得 Coinbase 实时参考价。请稍后刷新或检查网络。")
        st.caption(f"技术信息：{type(exc).__name__}: {exc}")
        st.stop()
    st.warning("实时价格请求失败，暂时显示上一次成功价格。")
    st.caption(f"技术信息：{type(exc).__name__}: {exc}")

try:
    df = fetch_klines()
except Exception as exc:
    st.error("无法获取用于模型计算的五分钟K线。实时价格可能仍可显示，但当前无法生成新预测。")
    st.caption(f"技术信息：{type(exc).__name__}: {exc}")
    st.stop()

feat_df, latest_row = make_features(df)
if len(feat_df) < 100:
    st.error(f"有效历史训练样本不足（{len(feat_df)} 条），暂时无法训练模型。")
    st.stop()

split = int(len(feat_df) * 0.8)
train_part, test_part = feat_df.iloc[:split].copy(), feat_df.iloc[split:].copy()
if train_part["target"].nunique() < 2:
    st.error("训练样本只有一个方向类别，暂时无法训练。")
    st.stop()

model = RandomForestClassifier(
    n_estimators=180, max_depth=6, min_samples_leaf=5,
    class_weight="balanced", random_state=42,
)
model.fit(train_part[FEATURES], train_part["target"].astype(int))
test_acc = accuracy_score(test_part["target"].astype(int), model.predict(test_part[FEATURES])) if len(test_part) else float("nan")
probabilities = model.predict_proba(latest_row[FEATURES].to_frame().T)[0]
classes = list(model.classes_)
up_prob = float(probabilities[classes.index(1)]) if 1 in classes else 0.5
down_prob = 1 - up_prob

news = fetch_news()
news_score = float(np.mean([item["score"] for item in news])) if news else 0.0
rsi = float(latest_row["rsi"])
ema_gap = float(latest_row["ema_gap"])
macd_gap = float(latest_row["macd_gap"])
technical_bias = np.clip(
    ema_gap * 18 + np.tanh(macd_gap / max(float(latest_row["close"]) * 0.0005, 1e-9)) * 0.08,
    -0.12, 0.12,
)
news_bias = np.clip(news_score * 0.05, -0.05, 0.05) if news else 0.0
combined_up = float(np.clip(up_prob + technical_bias + news_bias, 0.05, 0.95))
candle_price = float(df["close"].iloc[-1])
hourly_returns = df["close"].pct_change(6).dropna().tail(200)
move = max(float(hourly_returns.std()) if len(hourly_returns) else 0.01, 0.001)
low_est, high_est = live_price * (1 - 1.5 * move), live_price * (1 + 1.5 * move)

# Check simplified paper liquidation against the latest live price.
position = st.session_state.position
if position:
    side_sign = 1 if position["side"] == "做多" else -1
    unrealized = (live_price - position["entry"]) * position["qty"] * side_sign
    liquidation_triggered = (
        (position["side"] == "做多" and live_price <= position["liq_price"])
        or (position["side"] == "做空" and live_price >= position["liq_price"])
        or unrealized <= -position["margin"] * 0.995
    )
    if liquidation_triggered:
        st.session_state.trades.append({
            "时间": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "操作": f"{position['side']}模拟强平",
            "开仓价": position["entry"], "平仓价": live_price,
            "杠杆": position["leverage"], "保证金": position["margin"],
            "数量BTC": position["qty"], "盈亏": -position["margin"],
            "备注": "简化强平模拟；不等同交易所规则",
        })
        st.session_state.position = None
        st.session_state.cash += 0.0  # Locked margin is assumed lost on simulated liquidation.
        st.warning("模拟仓位触发简化强平条件，保证金已损失。真实交易所规则可能不同。")
        position = None

m1, m2, m3 = st.columns(3)
m1.metric("BTC 实时参考价", money(live_price))
m2.metric("模型估算上涨概率", f"{combined_up * 100:.1f}%")
m3.metric("历史留出集方向准确率", f"{test_acc * 100:.1f}%" if not np.isnan(test_acc) else "暂无")
st.caption(f"实时价格来源时间：{ticker_time or '接口未提供时间'} · K线最新收盘价：{money(candle_price)}（5分钟数据可能落后于实时价）")
direction = "偏多" if combined_up >= 0.56 else ("偏空" if combined_up <= 0.44 else "中性 / 观望")
st.subheader(f"未来约 1 小时：{direction}")
st.write(f"模型方向概率：上涨 **{up_prob*100:.1f}%**，下跌 **{down_prob*100:.1f}%**；技术和新闻启发式调整后，上涨估算 **{combined_up*100:.1f}%**。")
st.caption(f"粗略波动参考区间（非价格目标或保证）：{money(low_est)} 至 {money(high_est)}。新闻情绪 {news_score:+.2f}，标题 {len(news)} 条。")

st.warning("100 倍杠杆极高风险：价格反向波动约 0.5%–1% 就可能损失大部分保证金。本页面仅作模拟，强平公式为教学近似值，不代表任何交易所的实际规则。")

tab1, tab2, tab3, tab4 = st.tabs(["行情图表", "指标与新闻", "100倍多空模拟", "模型说明"])
with tab1:
    chart_df = df.tail(240)
    fig = go.Figure(data=[go.Candlestick(
        x=chart_df["time"], open=chart_df["open"], high=chart_df["high"],
        low=chart_df["low"], close=chart_df["close"], name="BTC/USD",
    )])
    fig.update_layout(height=420, margin=dict(l=5, r=5, t=25, b=5), xaxis_rangeslider_visible=False, title="BTC/USD · 5分钟K线")
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    a, b, c, d = st.columns(4)
    a.metric("RSI(14)", f"{rsi:.1f}")
    b.metric("EMA12 / EMA26差", f"{ema_gap*100:+.3f}%")
    c.metric("MACD差值", f"{macd_gap:.2f}")
    d.metric("新闻情绪", f"{news_score:+.2f}" if news else "不可用")
    st.markdown("**近期新闻标题（关键词评分，不代表事实判断）**")
    if news:
        for item in news[:12]:
            if item["link"]:
                st.markdown(f"- [{item['title']}]({item['link']}) · 情绪 {item['score']:+.1f}")
            else:
                st.write(f"- {item['title']} · 情绪 {item['score']:+.1f}")
    else:
        st.info("当前未抓取到新闻；新闻情绪调整按中性处理。")

with tab3:
    position = st.session_state.position
    unrealized = 0.0
    if position:
        sign = 1 if position["side"] == "做多" else -1
        unrealized = (live_price - position["entry"]) * position["qty"] * sign
        estimated_close_fee = position["qty"] * live_price * fee_pct
        margin_equity = max(0.0, position["margin"] + unrealized - estimated_close_fee)
        st.write(f"当前持仓：**{position['side']} {position['leverage']}x**")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("开仓价", money(position["entry"]))
        p2.metric("名义仓位", money(position["qty"] * position["entry"]))
        p3.metric("未实现盈亏（未扣平仓费）", money(unrealized), delta_color="normal")
        p4.metric("估算强平参考价", money(position["liq_price"]))
        st.write(f"保证金：{money(position['margin'])} · BTC数量：{position['qty']:.6f} · 估算平仓手续费：{money(estimated_close_fee)}")
        if st.button("平仓（模拟）", use_container_width=True):
            gross_pnl = unrealized
            exit_fee = position["qty"] * live_price * fee_pct
            returned_margin = max(0.0, position["margin"] + gross_pnl)
            returned_margin = max(0.0, returned_margin - exit_fee)
            st.session_state.cash += returned_margin
            st.session_state.trades.append({
                "时间": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "操作": f"{position['side']}模拟平仓",
                "开仓价": position["entry"], "平仓价": live_price,
                "杠杆": position["leverage"], "保证金": position["margin"],
                "数量BTC": position["qty"], "盈亏": gross_pnl,
                "平仓手续费": exit_fee, "备注": "模拟成交",
            })
            st.session_state.position = None
            st.rerun()
    else:
        st.info("当前没有持仓。选择做多或做空开仓。")
        st.write(f"可用虚拟资金：**{money(st.session_state.cash)}**")
        st.write(f"本次保证金 {money(min(margin_input, st.session_state.cash))} × {leverage} 倍 = 名义仓位约 **{money(min(margin_input, st.session_state.cash)*leverage)}**（不含费用）。")
        left, right = st.columns(2)
        with left:
            if st.button("📈 模拟做多开仓", use_container_width=True, type="primary"):
                margin = min(float(margin_input), float(st.session_state.cash))
                notional = margin * leverage
                qty = notional / live_price
                opening_fee = notional * fee_pct
                if margin >= 10 and margin + opening_fee <= st.session_state.cash:
                    st.session_state.cash -= margin + opening_fee
                    st.session_state.position = {
                        "side": "做多", "entry": live_price, "qty": qty,
                        "margin": margin, "leverage": leverage,
                        "liq_price": liq_price(live_price, "做多", leverage),
                        "opening_fee": opening_fee,
                    }
                    st.session_state.trades.append({
                        "时间": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "操作": "做多模拟开仓", "开仓价": live_price, "平仓价": np.nan,
                        "杠杆": leverage, "保证金": margin, "数量BTC": qty,
                        "盈亏": np.nan, "开仓手续费": opening_fee, "备注": "模拟成交",
                    })
                    st.rerun()
                else:
                    st.error("虚拟资金不足以支付保证金和开仓手续费。")
        with right:
            if st.button("📉 模拟做空开仓", use_container_width=True):
                margin = min(float(margin_input), float(st.session_state.cash))
                notional = margin * leverage
                qty = notional / live_price
                opening_fee = notional * fee_pct
                if margin >= 10 and margin + opening_fee <= st.session_state.cash:
                    st.session_state.cash -= margin + opening_fee
                    st.session_state.position = {
                        "side": "做空", "entry": live_price, "qty": qty,
                        "margin": margin, "leverage": leverage,
                        "liq_price": liq_price(live_price, "做空", leverage),
                        "opening_fee": opening_fee,
                    }
                    st.session_state.trades.append({
                        "时间": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "操作": "做空模拟开仓", "开仓价": live_price, "平仓价": np.nan,
                        "杠杆": leverage, "保证金": margin, "数量BTC": qty,
                        "盈亏": np.nan, "开仓手续费": opening_fee, "备注": "模拟成交",
                    })
                    st.rerun()
                else:
                    st.error("虚拟资金不足以支付保证金和开仓手续费。")

    pos = st.session_state.position
    locked_margin = pos["margin"] if pos else 0.0
    if pos:
        sign = 1 if pos["side"] == "做多" else -1
        unrealized_now = (live_price - pos["entry"]) * pos["qty"] * sign
    else:
        unrealized_now = 0.0
    st.divider()
    st.metric("可用虚拟资金", money(st.session_state.cash))
    st.metric("模拟账户估算权益（含持仓浮盈亏）", money(st.session_state.cash + locked_margin + unrealized_now))
    if st.session_state.trades:
        trades_df = pd.DataFrame(st.session_state.trades)
        st.dataframe(trades_df.iloc[::-1], use_container_width=True, hide_index=True)
        csv_data = trades_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("导出模拟交易 CSV", data=csv_data, file_name="btc_leverage_paper_trades.csv", mime="text/csv")
    else:
        st.info("暂无模拟交易记录。")

with tab4:
    st.markdown("""
- **实时价格**：Coinbase BTC-USD ticker，页面每 5 秒刷新一次；请求失败时会显示上次成功价格。
- **预测数据**：Coinbase 五分钟K线，缓存约 60 秒；因此预测模型不是每 5 秒获得一根新K线。
- **模型**：随机森林方向分类，目标为约一小时后方向；新闻使用简单关键词情绪。
- **杠杆模拟**：支持做多/做空，1–100倍，记录保证金、名义仓位、浮动盈亏、手续费和模拟平仓。
- **强平估算**：使用简化教学公式，不是任何交易所的实际强平规则，也未模拟资金费率、滑点、盘口深度、分档维持保证金等。
- **持久性**：模拟账户保存在网页会话中，应用重启或会话丢失后可能清空。
- **重要**：100倍杠杆非常容易损失全部保证金；仅用于学习和模拟，不构成投资建议，不会执行真实交易。
""")

st.divider()
st.caption("仅用于学习与模拟研究，不构成投资建议。100倍杠杆可能在极小的反向波动下导致保证金几乎全部损失。")
