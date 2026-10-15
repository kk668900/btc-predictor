# BTC 实时预测与杠杆模拟站

- 页面每 5 秒自动刷新，实时价格来自 Coinbase BTC-USD ticker。
- 预测仍使用五分钟K线，约每 60 秒刷新缓存。
- 支持虚拟资金模拟做多/做空，1–100倍杠杆，显示浮盈亏、手续费和简化强平参考价。
- 不连接交易所账户，不执行真实订单。

## GitHub / Streamlit 部署
上传 app.py 和 requirements.txt 到仓库根目录。Streamlit Community Cloud 的 Main file path 设为 app.py。
注意：模拟强平规则仅为教学近似，非交易所真实公式。100倍杠杆风险极高。
