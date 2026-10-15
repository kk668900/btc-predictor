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
