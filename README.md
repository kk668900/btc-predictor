# BTC 智能预测站（第一版）

## 功能
- 手机友好的 Streamlit 网页
- Binance 公共 BTCUSDT 5 分钟行情（无需 API key）
- RSI、EMA、MACD、波动率
- 随机森林预测未来约 1 小时方向
- CoinDesk / Cointelegraph RSS 标题的简单关键词情绪分数
- 虚拟现金模拟买卖、手续费、交易记录 CSV 导出
- 不连接交易所账户、不执行真实订单

## Windows 安装
1. 安装 Python 3.11 或 3.12（安装时勾选 Add Python to PATH）。
2. 解压整个文件夹。
3. 双击 `启动网站.bat`，等待依赖安装完成。
4. 浏览器访问终端显示的本机地址，通常是 `http://localhost:8501`。

如果 `py` 命令不可用，请从 python.org 安装 Python，并重新打开命令提示符。

## 手机访问（同一 Wi-Fi）
1. 在电脑命令提示符中运行：
   `py -m streamlit run app.py --server.address 0.0.0.0`
2. 查电脑局域网 IPv4 地址：运行 `ipconfig`，找到当前 Wi-Fi 适配器的 IPv4 地址。
3. 手机连接同一个 Wi-Fi，在浏览器打开 `http://电脑IPv4地址:8501`。
4. 如果 Windows 防火墙询问，允许 Python 在“专用网络”通信。不要在公共 Wi-Fi 上开放端口。

## 免费部署
本地运行是最稳妥的第一步。若要手机离开家中 Wi-Fi 仍能访问，可之后尝试 Streamlit Community Cloud 等免费托管，但免费额度、休眠策略和依赖限制可能变化；不要在公开部署中放入密钥或个人数据。

## 注意
- 公共 API 和 RSS 可能限流、变化或不可访问。
- 预测是实验性质，准确率不保证未来表现。
- 模型只用时间顺序留出集做了基础检查，尚不是严谨的滚动回测。
- 新闻评分是关键词启发式，不是语义理解。
- 模拟账户保存在会话状态中，重启可能清空。
- 绝不粘贴交易所 API 密钥；当前版本不需要任何交易所密钥。
