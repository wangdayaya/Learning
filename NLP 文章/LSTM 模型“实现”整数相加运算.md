# 文章简介


本文主要介绍了使用 `LSTM` 模型完成简单的两个整数相加的运算。

# 数据准备

为了满足模型训练的需要，应该准备 `50000` 条样本，每个样本包含 `query 字符串`和 `ans 字符串`，如下所示：

    query：52+758
    ans: 810

我们这里限定了加法运算的两个整数都是 `1-999` 的任意一个整数，所以 query 的长度最长为 `7` （两个最大的三位数和一个加号组成的字符串长度），ans 的长度最长为 `4` ，如果长度不足，则在后面用`空格补齐`。关键代码如下：

```
while len(questions) < TRAINING_SIZE:
    f = lambda: int("".join(np.random.choice(list("1234567890")) for _ in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    ...
    q = "{}+{}".format(a, b)
    query = q + " " * (MAXLEN - len(q))
    ans = str(a + b)
    ans += " " * (DIGITS + 1 - len(ans))
    questions.append(query)
    expected.append(ans)
```

样本创建好之后，需要对样本进行`向量化处理`，也就是将每个字符都转换成对应的  `one-hot 表示`，因为每个样本的 query 长度为 7 ，字符集合长度为 12 ，所以每个 query 改成 `[7,12]` 的 one-hot 向量；每个样本的 ans 长度为 4 ，字符集合长度为 12 ，所以每个 ans 改成 `[4,12]` 的 one-hot 向量。关键代码如下：

```
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)
```


# 模型搭建

模型结构很简单，主要使用了 LSTM 层、RepeatVector 层、 Dense 层，都是基础知识，不做过多解释，编译模型时候设置损失函数为`categorical_crossentropy`，优化器为 `adam` 优化器，评估指标为准确率`accuracy` 关键代码如下：

```
model = keras.Sequential()
model.add(layers.LSTM(128, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS + 1))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.Dense(len(chars), activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

# 模型训练

选取 `90%` 的样本为训练集， `10%` 的样本为测试集，下面是模型训练的日志打印：

    Iter 1
    1407/1407 [==============================] - 11s 6ms/step - loss: 1.7796 - accuracy: 0.3499 - val_loss: 1.5788 - val_accuracy: 0.4065
    Iter 2
    1407/1407 [==============================] - 9s 6ms/step - loss: 1.3928 - accuracy: 0.4762 - val_loss: 1.2489 - val_accuracy: 0.5346
    ...
    Iter 28
    1407/1407 [==============================] - 9s 6ms/step - loss: 0.0205 - accuracy: 0.9944 - val_loss: 0.0257 - val_accuracy: 0.9917
    Iter 29
    1407/1407 [==============================] - 9s 6ms/step - loss: 0.0256 - accuracy: 0.9926 - val_loss: 0.0747 - val_accuracy: 0.9827

# 效果展示

下面展示了 10 条样本结果，预测正确的有 `☑` 表示，预测错误的有 `☒` 表示，可以看出来结果基本正确，最终的验证集准确率能达到 0.9827 。

    Q 537+65  A 602  ☑ 602 
    Q 0+998   A 998  ☑ 998 
    Q 50+691  A 741  ☑ 741 
    Q 104+773 A 877  ☑ 877 
    Q 21+84   A 105  ☑ 105 
    Q 318+882 A 1200 ☑ 1200
    Q 850+90  A 940  ☑ 940 
    Q 96+11   A 107  ☒ 907 
    Q 1+144   A 145  ☑ 145 
    Q 809+4   A 813  ☑ 813 

# 参考

https://github.com/wangdayaya/DP_2023/blob/main/NLP%20%E6%96%87%E7%AB%A0/Sequence%20to%20sequence%20learning%20for%20performing%20number%20addition.py