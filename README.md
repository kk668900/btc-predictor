# BTC 智能预测站（Coinbase 行情版）

## 部署到 Streamlit Community Cloud

1. 将 `app.py`、`requirements.txt` 上传到 GitHub 仓库根目录。
2. 在 https://share.streamlit.io/ 创建应用。
3. Repository 选择你的仓库，Branch 选择 `main`，Main file path 填 `app.py`。
4. 部署后用手机浏览器打开 Streamlit 提供的网址。

## 行情说明
- 行情源改为 Coinbase Exchange 的公开 BTC-USD 五分钟 K 线接口，无需 API key。
- 公共 API 可能因网络、地区、服务状态或限流而不可用；若报错，请查看页面显示的技术信息。
- 本程序不是经过验证的投资模型。预测概率与留出集准确率不保证未来表现。
- 仅用于学习和模拟交易，不会执行真实订单。
