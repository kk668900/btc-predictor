import streamlit as st
import pandas as pd
import numpy as np
import requests
import feedparser
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
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

COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
FEATURES = [
    "ret1", "ret3", "ret12", "volatility",
    "ema_gap", "rsi", "macd_gap", "volume_change"
]


@st.cache_data(ttl=60)
def fetch_klines():
    """Fetch public Coinbase BTC-USD five-minute candles; no API key required."""
    response = requests.get(
        COINBASE_URL,
        params={"granularity": 300},
        headers={
            "Accept": "application/json",
            "User-Agent": "BTC-Predictor/1.0"
        },
        timeout=20,
    )
    response.raise_for_status()
    raw = response.json()

    if not isinstance(raw, list) or not raw:
        raise ValueError("备用行情接口没有返回有效 K 线数据。")

    # Coinbase candle format: [time, low, high, open, close, volume]
    df = pd.DataFrame(
        raw,
        columns=["timestamp", "low", "high", "open", "close", "volume"],
    )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = (
        df[["time", "open", "high", "low", "close", "volume"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )

    if len(df) < 200:
        raise ValueError(
            f"行情接口只返回 {len(df)} 根有效 K 线，少于模型需要的 200 根。"
            "请稍后刷新，或检查当前网络是否能访问 Coinbase API。"
        )
    return df


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

    x["macd"] = x["ema12"] - x["ema26"]
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_gap"] = x["macd"] - x["macd_signal"]
    x["volume_change"] = x["volume"].pct_change().replace(
        [np.inf, -np.inf], np.nan
    )

    # Future direction: six five-minute candles ahead (approximately one hour).
    x["target"] = (x["close"].shift(-6) > x["close"]).astype(float)
    x.loc[x.index[-6:], "target"] = np.nan
    x = x.replace([np.inf, -np.inf], np.nan)
    usable = x.dropna(subset=FEATURES + ["target"]).copy()
    latest = x.dropna(subset=FEATURES).iloc[-1]
    return usable, latest


@st.cache_data(ttl=300)
def fetch_news():
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    items = []
    positive = [
        "surge", "rally", "bull", "gain", "rise", "approval", "adoption",
        "突破", "上涨", "利好", "增长", "买入",
    ]
    negative = [
        "drop", "fall", "crash", "bear", "loss", "hack", "lawsuit", "ban",
        "下跌", "暴跌", "利空", "监管", "抛售",
    ]

    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:20]:
                title = str(entry.get("title", "")).strip()
                if title:
                    items.append({
                        "title": title,
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                    })
        except Exception:
            continue

    for item in items:
        title = item["title"].lower()
        p = sum(word in title for word in positive)
        n = sum(word in title for word in negative)
        item["score"] = (p - n) / max(1, p + n)
    return items


def money(value):
    return f"${value:,.2f}"


with st.sidebar:
    st.header("控制面板")
    if st.button("🔄 刷新行情与新闻", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("行情：Coinbase 公共现货 K 线（BTC-USD，5分钟）")
    st.caption("预测周期：未来约 1 小时")
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

    starting_cash = st.number_input(
        "初始虚拟资金（美元）",
        min_value=100.0, max_value=10000000.0,
        value=10000.0, step=500.0,
    )
    fee_pct = st.number_input(
        "模拟手续费（单边 %）",
        min_value=0.0, max_value=2.0,
        value=0.1, step=0.05,
    ) / 100

    if st.button("重置模拟账户", use_container_width=True):
        st.session_state.cash = float(starting_cash)
        st.session_state.btc = 0.0
        st.session_state.trades = []
        st.session_state.entry_price = None
        st.rerun()


try:
    df = fetch_klines()
except Exception as exc:
    st.error(
        "暂时无法获取 Coinbase 免费公开行情。请确认网络可以访问 Coinbase API，"
        "再点击左侧“刷新行情与新闻”或稍后重试。"
    )
    st.caption(f"技术信息：{type(exc).__name__}: {exc}")
    st.stop()

feat_df, latest_row = make_features(df)
if len(feat_df) < 100:
    st.error(f"有效历史训练样本不足（当前 {len(feat_df)} 条），暂时无法训练模型。")
    st.stop()

# Chronological holdout; do not shuffle time-series observations.
split = int(len(feat_df) * 0.8)
train_part = feat_df.iloc[:split].copy()
test_part = feat_df.iloc[split:].copy()

if train_part["target"].nunique() < 2:
    st.error("训练样本只包含一个方向类别，暂时无法训练分类模型。请稍后刷新。")
    st.stop()

model = RandomForestClassifier(
    n_estimators=180,
    max_depth=6,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
)
model.fit(train_part[FEATURES], train_part["target"].astype(int))

if len(test_part) and test_part["target"].nunique() > 0:
    test_predictions = model.predict(test_part[FEATURES])
    test_acc = accuracy_score(test_part["target"].astype(int), test_predictions)
else:
    test_acc = float("nan")

probabilities = model.predict_proba(latest_row[FEATURES].to_frame().T)[0]
classes = list(model.classes_)
up_prob = float(probabilities[classes.index(1)]) if 1 in classes else 0.5
down_prob = 1.0 - up_prob

news = fetch_news()
news_score = float(np.mean([item["score"] for item in news])) if news else 0.0

rsi = float(latest_row["rsi"])
ema_gap = float(latest_row["ema_gap"])
macd_gap = float(latest_row["macd_gap"])
technical_bias = np.clip(
    ema_gap * 18
    + np.tanh(macd_gap / max(float(latest_row["close"]) * 0.0005, 1e-9)) * 0.08,
    -0.12, 0.12,
)
news_bias = np.clip(news_score * 0.05, -0.05, 0.05) if news else 0.0
combined_up = float(np.clip(up_prob + technical_bias + news_bias, 0.05, 0.95))

current_price = float(df["close"].iloc[-1])
hourly_returns = df["close"].pct_change(6).dropna().tail(200)
move = float(hourly_returns.std()) if len(hourly_returns) else 0.01
move = max(move, 0.001)
low_est = current_price * (1 - 1.5 * move)
high_est = current_price * (1 + 1.5 * move)

m1, m2, m3 = st.columns(3)
m1.metric("BTC 当前参考价", money(current_price))
m2.metric("模型估算上涨概率", f"{combined_up * 100:.1f}%")
m3.metric(
    "时间顺序留出集准确率",
    f"{test_acc * 100:.1f}%" if not np.isnan(test_acc) else "暂无",
)

direction = (
    "偏多" if combined_up >= 0.56
    else ("偏空" if combined_up <= 0.44 else "中性 / 观望")
)
st.subheader(f"未来 1 小时：{direction}")
st.write(
    f"模型方向概率：上涨 **{up_prob * 100:.1f}%**，下跌 **{down_prob * 100:.1f}%**。"
    f"加入技术与新闻的启发式调整后，上涨估算为 **{combined_up * 100:.1f}%**。"
)
st.caption(
    f"粗略波动参考区间（不是价格目标或保证）：{money(low_est)} 至 "
    f"{money(high_est)}。新闻情绪评分：{news_score:+.2f}；"
    f"抓取到 {len(news)} 条标题。"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["行情图表", "指标与新闻", "模拟交易", "模型说明"]
)

with tab1:
    chart_df = df.tail(240)
    fig = go.Figure(data=[
        go.Candlestick(
            x=chart_df["time"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="BTC/USD",
        )
    ])
    fig.update_layout(
        height=420, margin=dict(l=5, r=5, t=25, b=5),
        xaxis_rangeslider_visible=False, title="BTC/USD · 5分钟K线",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    a, b, c, d = st.columns(4)
    a.metric("RSI(14)", f"{rsi:.1f}")
    b.metric("EMA12 / EMA26差", f"{ema_gap * 100:+.3f}%")
    c.metric("MACD差值", f"{macd_gap:.2f}")
    d.metric("新闻情绪", f"{news_score:+.2f}" if news else "不可用")
    st.markdown("**近期新闻标题（关键词情绪评分，不代表事实判断）**")
    if news:
        for item in news[:12]:
            if item["link"]:
                st.markdown(
                    f"- [{item['title']}]({item['link']}) · 情绪 {item['score']:+.1f}"
                )
            else:
                st.write(f"- {item['title']} · 情绪 {item['score']:+.1f}")
    else:
        st.info("当前未抓取到新闻。新闻情绪调整按中性处理，不会伪造新闻数据。")

with tab3:
    st.write(
        f"虚拟现金：**{money(st.session_state.cash)}**  |  "
        f"BTC持仓：**{st.session_state.btc:.6f} BTC**"
    )
    equity = st.session_state.cash + st.session_state.btc * current_price
    st.write(
        f"账户总资产：**{money(equity)}**  |  "
        f"相对初始账户净值盈亏：**{money(equity - 10000)}**"
        "（以 $10,000 为默认基准）"
    )
    buy_amount = st.number_input(
        "本次模拟买入金额（美元）",
        min_value=10.0, max_value=1000000.0,
        value=100.0, step=10.0, key="buy_amount",
    )
    sell_pct = st.slider("卖出持仓比例（%）", min_value=1, max_value=100, value=100)
    c1, c2 = st.columns(2)

    with c1:
        if st.button("模拟买入", use_container_width=True):
            spend = min(float(buy_amount), st.session_state.cash / (1 + fee_pct))
            if spend >= 10:
                qty = spend / current_price
                fee = spend * fee_pct
                st.session_state.cash -= spend + fee
                st.session_state.btc += qty
                st.session_state.trades.append({
                    "时间": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "操作": "模拟买入", "价格": current_price,
                    "数量BTC": qty, "手续费": fee,
                })
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
                st.session_state.trades.append({
                    "时间": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "操作": "模拟卖出", "价格": current_price,
                    "数量BTC": qty, "手续费": fee,
                })
                st.rerun()
            else:
                st.warning("当前没有 BTC 持仓。")

    if st.session_state.trades:
        st.dataframe(
            pd.DataFrame(st.session_state.trades).iloc[::-1],
            use_container_width=True, hide_index=True,
        )
        csv_data = pd.DataFrame(st.session_state.trades).to_csv(
            index=False
        ).encode("utf-8-sig")
        st.download_button(
            "导出交易记录 CSV", data=csv_data,
            file_name="btc_paper_trades.csv", mime="text/csv",
        )
    else:
        st.info("暂无模拟交易记录。所有操作只修改本网页的虚拟账户。")

with tab4:
    st.markdown("""
- **数据**：Coinbase 公共 BTC-USD 五分钟 K 线接口，无需 API 密钥。
- **特征**：短期收益率、波动率、EMA差、RSI、MACD差、成交量变化。
- **模型**：随机森林分类器，按时间顺序切分训练集与测试集，不随机打乱。
- **目标**：当前收盘价与约1小时后的收盘价比较，预测方向。
- **新闻**：读取公开 RSS 标题，使用简单中英关键词评分；不是大型语言模型，也可能漏判语境。
- **准确率**：只是一次时间顺序留出集的方向准确率，不等于未来胜率，也未充分模拟交易成本。
- **重要限制**：概率未做严格校准；技术/新闻融合为启发式调整。应进一步做滚动回测、基准比较与手续费/滑点测试。
- **隐私与安全**：不要求交易所密钥，不执行真实买卖。模拟账户保存在网页会话中，重启后可能清空。
""")

st.divider()
st.caption(
    "仅用于学习与模拟研究，不构成投资建议。加密资产波动极高，模型可能连续判断错误。"
)
