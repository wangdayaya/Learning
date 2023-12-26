# 前言
最近学习量化交易，所以试着写一个回测脚本。本文使用的是苹果公司的历史股票数据，定义了一个简单的回测策略，先使用历史股票数据进行回测，然后再使用模型预测的股票数据进行回测。

# 使用历史数据进行回测
这段代码使用了Backtrader库进行股票回测和策略模拟。具体如下：

1.  使用`yfinance`（Yahoo Finance）库从Yahoo Finance下载苹果公司（AAPL）的股票价格数据，日期范围从2020年1月1日到2023年8月1日。yfinance 是一个 Python 开源工具，可以借用雅虎的接口轻松访问金融数据。
1.  自定义策略类`MyStrategy`继承自 Backtrader 的`Strategy`类。在该类的`__init__`方法中初始化了数据，并创建了一个简单移动平均指标（Simple Moving Average，SMA）。简单移动平均线（SMA）是技术分析指标，代表资产在给定时期内的平均价格，这里设置了使用最近 20 天的平均价格。
1.  在每个交易日的下一个时间步中，`next`方法被调用。它首先检查是否有仓位，也就是是否持有股票，然后根据 SMA 信号执行买入或卖出操作。如果当前数据的收盘价大于移动平均线的值，创建一个买入`buy`的交易指令，购买 10 个单位的股票。如果当前数据的收盘价小于移动平均线的值，创建一个卖出`sell`的交易指令，卖出 10  个单位的股票。
1.  我们设置初始资金为 `100000` 块，并且设置了交易佣金的比例为 `0.001` ，也就是说佣金的价格等于交易总价乘这个比例。然后在回测引擎中设置好我们获取的苹果公司股票历史数据和自定义的回测策略，然后运行回测引擎执行策略模拟。最后可以得到回测结束后的投资的总价值，并且可以计算出净收益。可以看出来三年多的时间才净利润 1015 块，真的是衰~

```
stock_data = yf.download("AAPL", start="2020-01-01", end="2023-08-01")
class MyStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=20)

    def next(self):
        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                self.buy(size=10)
        else:
            if self.dataclose[0] < self.sma[0]:
                self.sell(size=10)
data = bt.feeds.PandasData(dataname=stock_data)
cerebro = bt.Cerebro()
cash = 100000
cerebro.broker.set_cash(cash)
cerebro.broker.setcommission(commission=0.001)
cerebro.adddata(data)
cerebro.addstrategy(MyStrategy)
cerebro.run()
portvalue = cerebro.broker.getvalue()
pnl = portvalue - cash
print(f'回测结束后的投资总价值: {round(portvalue, 10)}')
print(f'净收益: {round(pnl, 10)}')
```


结果打印：
```
回测结束后的投资总价值: 101015.8684207153
净收益: 1015.8684207153
```

# 使用预测数据进行回测
1.  和上面一样使用`yfinance`库下载从"2020-01-01"到"2023-08-01"时间范围内的苹果公司（AAPL）股票价格数据。
 
3.  我们要对下载的股票的收盘价数据进行归一化处理，也就是这里使用  `MinMaxScaler` 函数进行了所有价格的缩放，便于我们计算时候快速收敛。
4.  使用收盘价序列制作了训练数据集，输入就是以每` 10` 个时间步为单位的窗口序列，将紧邻的下一个时间步收盘价当作标签。



```

stock_data = yf.download("AAPL", start="2020-01-01", end="2023-08-01")
closing_prices = stock_data["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler()
closing_prices_scaled = scaler.fit_transform(closing_prices)
X_train = []
y_train = []
sequence_length = 10
for i in range(sequence_length, len(closing_prices_scaled)):
    X_train.append(closing_prices_scaled[i - sequence_length:i])
    y_train.append(closing_prices_scaled[i])
X_train = np.array(X_train)
y_train = np.array(y_train)
```


1.  构建一个 `LSTM` 模型，只是简单包括一个 LSTM 层和一个全连接层。模型使用 `MSE` 损失函数进行编译，使用 `Adam` 优化器进行训练，并进行 10 个周期的训练。

1.  使用训练好的 LSTM 模型对序列数据进行预测，然后将预测结果逆归一化得到预测的股票收盘价。

1.  因为预测的收盘价缺少最初的 10 个值，所以我们直接将原数据的前 10 个收盘价拼接到最开始，形成了这段时间完整的预测收盘价数据。

1. 最后取 `['Open', 'High', 'Low', 'Close', 'Volume']` 这几个列作为后续回测的数据。


```
model = Sequential()
model.add(LSTM(1, activation='relu', input_shape=(sequence_length, 1), kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32)

predicted_prices_scaled = model.predict(X_train)
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)
predicted_prices = np.concatenate([np.array(closing_prices[:10]), predicted_prices], axis=0)
stock_data['Close'] = [p[0]for p in predicted_prices]
stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

```
1.  自定义策略类`MyStrategy`，逻辑和上面一样，只是这里的回测数据使用的是用模型预测出来的收盘价，其他如初始资金、交易佣金等都和上面一样。计算结束之后打印投资总价值和净收益。

 
```
class MyStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=20)

    def next(self):
        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                self.buy(size=10)
        else:
            if self.dataclose[0] < self.sma[0]:
                self.sell(size=10)


data = bt.feeds.PandasData(dataname=stock_data)
cerebro = bt.Cerebro()
cash = 100000
cerebro.broker.set_cash(cash)
cerebro.broker.setcommission(commission=0.001)
cerebro.adddata(data)
cerebro.addstrategy(MyStrategy)
cerebro.run()
portvalue = cerebro.broker.getvalue()
pnl = portvalue - cash
print(f'回测结束后的投资总价值: {round(portvalue, 10)}')
print(f'净收益: {round(pnl, 10)}')
```
我在测试过程中使用了 1 、8 、64 三种不同数量的神经元个数分别进行了两次的训练和测试，结果打印如下，可以看出来还不如上面我直接用历史数据进行回测的净利润多呢，三年的时间才净收益几百块，我如果是量化交易员估计要被顾客捶死了！而且对于使用相同的神经元个数，训练结束后得到的净利润差别很大，很让人迷惑。

```
训练 1 个神经元
回测结束后的投资总价值: 100699.1177435303
净收益: 699.1177435303
回测结束后的投资总价值: 100403.6095542145
净收益: 403.6095542145

训练 8 个神经元
回测结束后的投资总价值: 100781.6765512848
净收益: 781.6765512848
回测结束后的投资总价值: 100852.121304245
净收益: 852.121304245

训练 64 个神经元
回测结束后的投资总价值: 100742.7554497528
净收益: 742.7554497528
回测结束后的投资总价值: 100682.2641077423
净收益: 682.2641077423
```

# 后记
本人是刚开始学习量化交易方面的知识，这仅是一个练习的小案例，其中还是有很多问题的：
- 回测策略定义太过于简单，这里主要是为了体验一下   `backtrader` 库的使用方式
- 预测模型定义太过于简单，结果波动较大，目前还不了解原因
- 数据的特征过于简单，正常情况的特征数量肯定是很多的
- 对于金融市场的规则或者知识了解的还比较欠缺，正常情况还需要考虑手续等成本费用，交易时间，交易限制等规则

随着学习的深入肯定会细化这些内容，等我把回测策略写好，分分钟挣他个利润几十倍！