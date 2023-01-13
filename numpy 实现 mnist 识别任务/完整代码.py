from util import *
import mnistLoader

import numpy as np
import gzip
import pickle
import random

MNIST_FILE_PATH = 'mnist.pkl.gz'
BATCH_SIZE = 500


# 加载 mnist 数据，返回的结果包含了训练集、验证集、测试集三部分数据
# 训练集有中输入数据大小为 (50000, 784) ，表示有 50000 张图片，每张图片有 784 个像素点，对应有标签大小为 (50000,) ，表示每张图片的所属类别
# 验证集有中输入数据大小为 (10000, 784) ，表示有 10000 张图片，每张图片有 784 个像素点，对应有标签大小为 (10000,) ，表示每张图片的所属类别
# 测试集有中输入数据大小为 (10000, 784) ，表示有 10000 张图片，每张图片有 784 个像素点，对应有标签大小为 (10000,) ，表示每张图片的所属类别
def load_data():
    with gzip.open(MNIST_FILE_PATH) as f:
        return pickle.load(f, encoding='bytes')


# 先将 data 中的样本和标签分别进行 shuffle ，然后将样本和标签分别切分成每个大小为 batch_size 的批数据
# BATCH_SIZE 设置为 500 ，最后训练集会有 100 批数据，验证集会有 20 批数据，测试集会有 20 批数据
def get_batches(data):
    n = len(data[0])
    shuffle_data_idx = random.sample(range(n), n)
    X = data[0][shuffle_data_idx]
    Y = data[1][shuffle_data_idx]
    X_batches = [X[i: i + BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]
    Y_batches = [Y[i: i + BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]
    return X_batches, Y_batches


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
    scores = a[range(n), y]
    loss = -np.sum(np.log(scores)) / n
    return a, loss


#  这里计算输出层 softmax_cross_entropy 的误差
# logits 是经过线性变化的概率结果，大小为 (batch_size, output_size)
# y 是该批样本的真实标签
def derivation_softmax_cross_entropy(logits, y):
    n = logits.shape[0]
    a = softmax(logits)
    a[range(n), y] -= 1
    return a


class Network:
    # node_num_list 存放了网络结构中每层神经元个数
    # num_layers 表示网络结构层数
    # weights 存放了相邻层之间的网络权重参数
    # biases 存放了相邻层之间的偏置参数
    def __init__(self, node_num_list):
        self.node_num_list = node_num_list
        self.num_layers = len(node_num_list)
        self.weights = [np.random.randn(i, j) for i, j in
                        zip(self.node_num_list[:-1], self.node_num_list[1:])]
        self.biases = [np.random.randn(1, j) for j in self.node_num_list[1:]]

    # 前向传播过程有两种：
    # 推理预测时的前向传播，此时不需要用到softmax, 只用logits就可以做出预测
    # 训练时的前向传播，此时需要记录中间变量z和a, 并且需要进行softmax计算

    # 这里用于推理预测的前向传播
    # 前向传播过程中从第一层到倒数第二层中要进行线性变化和激活函数计算
    # 前向传播过程中在最后一层中只需要进行线性转化，不用进行激活函数计算
    def forward(self, x):
        a = x
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(a, weight) + bias
            a = relu(z)

        logits = np.dot(a, self.weights[-1]) + self.biases[-1]
        return logits

    # 反向传播中先进行前向传播，我们要记录下每一层的线性转换结果 z 和激活函数转换结果 a
    # 在反向传播过程中记录该批训练样本数据的损失值 loss ，以及每一层的损失对权重参数和偏置参数的梯度 dws 、dbs
    def backward(self, x, y):
        dws = [np.zeros((i, j)) for i, j in zip(self.node_num_list[:-1], self.node_num_list[1:])]
        dbs = [np.zeros((1, j)) for j in self.node_num_list[1:]]

        # 前向传播
        zs = []
        _as = []
        a = x
        _as.append(a)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(a, weight) + bias
            zs.append(z)
            a = relu(z)
            _as.append(a)
        logits = np.dot(a, self.weights[-1]) + self.biases[-1]
        zs.append(logits)
        a, loss = softmax_cross_entropy(logits, y)
        _as.append(a)

        # 反向传播
        # 计算输出层的误差
        dl = derivation_softmax_cross_entropy(logits, y)
        n = len(x)
        # 记录输出层误差对于网络结构中最后一组权重参数和偏置参数的梯度
        dws[-1] = np.dot(_as[-2].T, dl) / n
        dbs[-1] = np.sum(dl, axis=0, keepdims=True) / n
        # 记录网络结构中所有后面一层传来的误差对于相邻前面一组权重参数和偏置参数的梯度
        for i in range(2, self.num_layers):
            dl = np.dot(dl, self.weights[-i + 1].T) * derivation_relu(zs[-i])
            dws[-i] = np.dot(_as[-i - 1].T, dl) / n
            dbs[-i] = np.sum(dl, axis=0, keepdims=True) / n
        return loss, dws, dbs

    # 使用训练数据来更新模型的权重参数和偏置参数，并且用验证集对模型的性能进行验证
    def train(self, train_data, val_data, lr, epoch):
        val_accs = []
        for e in range(epoch):
            # 训练样本来更新权重和偏置参数
            x_batches, y_batches = mnistLoader.get_batches(train_data)
            for i, (x, y) in enumerate(zip(x_batches, y_batches)):
                loss, dws, dbs = self.backward(x, y)
                self.weights = [weight - lr * dw for weight, dw in zip(self.weights, dws)]
                self.biases = [bias - lr * db for bias, db in zip(self.biases, dbs)]

            # 使用验证集进行模型的性能验证
            x_batches, y_batches = mnistLoader.get_batches(val_data)
            corrects = 0
            n = len(val_data[0])
            for i, (x, y) in enumerate(zip(x_batches, y_batches)):
                logits = self.forward(x)
                correct = np.sum(np.argmax(logits, axis=-1) == y)
                corrects += correct
            acc = corrects / n
            val_accs.append(acc)
            print("Epoch {}, acc {}/{}={}".format(e, corrects, n, acc))


if __name__ == "__main__":
    train_data, val_data, _ = mnistLoader.load_data()
    model = Network([784, 64, 10])
    model.train(train_data, val_data, 1, 50)
