import numpy as np


# relu 激活函数的功能就是将线性转化后的结果中的小于等于 0 的部分都改为 0 ，实现了非线性的转换
# z 表示经过线性转换后的结果，大小为 (batch_size, hidden_size)
# 返回的结果的大小仍然是  (batch_size, hidden_size)
def relu(z):
    flag = (z <= 0)
    z[flag] = 0
    return z


# derivation_relu 是在求 relu 激活函数的导数
def derivation_relu(z):
    flag = (z <= 0)
    z[flag] = 0
    z[~flag] = 1
    return z


# softmax 函数就是将一个向量进行转换，将每个值转换到 0-1 之间的一个小数，且经过转换后的向量中的所有小数加起来和为 1
# z 表示经过线性转换后的结果，大小为 (batch_size, output_size)
# 返回的结果的大小仍然是  (batch_size, output_size)
def softmax(z):
    max_value_every_batch = np.max(z, axis=-1, keepdims=True)
    tmp = z - max_value_every_batch
    return np.exp(tmp) / np.sum(np.exp(tmp), axis=-1, keepdims=True)


# 在 softmax 计算结果的基础上，又进行了交叉熵损失的计算
# logits 表示经过线性变化输出的预测结果，还未经过 softmax 转换，大小为 (batch_size, output_size)
# y 为该批次的标签，大小为  (batch_size,)
# 返回的 a 是经过 softmax 转换的结果，loss 是计算出来的交叉熵结果
def softmax_cross_entropy(logits, y):
    n = logits.shape[0]
    a = softmax(logits)
    l = range(n)
    scores = a[l, y]
    loss = -np.sum(np.log(scores)) / n
    return a, loss


#  这里计算输出层 softmax_cross_entropy 的误差
# logits 是经过线性变化的概率结果，大小为 (batch_size, output_size)
# y 是该批样本的真实标签
def derivation_softmax_cross_entropy(logits, y):
    n = logits.shape[0]
    a = softmax(logits)
    l = range(n)
    a[l, y] = a[l, y] - 1
    return a
