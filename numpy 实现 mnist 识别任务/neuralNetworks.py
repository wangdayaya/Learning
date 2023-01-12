from util import *


class Netword:
    # node_num_list 存放了网络结构中每层神经元个数
    # num_layers 表示网络结构层数
    # weights 存放了相邻层之间的网络权重参数
    # biases 存放了相邻层之间的偏置参数
    def __init__(self, node_num_list):
        self.node_num_list = node_num_list
        self.num_layers = len(node_num_list)
        self.weights = [np.random.randn(i, j) for i, j in
                        zip(self.node_num_list[:-1], self.node_num_list[1:])]
        self.biases = [np.random.randn((1, j) for j in self.node_num_list[1:])]

    # 前向传播过程有两种：
    # 推理预测时的前向传播，此时不需要用到softmax, 只用logits就可以做出预测
    # 训练时的前向传播，此时需要记录中间变量z和a, 并且需要进行softmax计算

    # 这里用于推理预测的前向传播
    # 前向传播过程中从第一层到倒数第二层中要进行线性变化和激活函数计算
    # 前向传播过程中在最后一层中只需要进行线性转化，不用进行激活函数计算
    def forward(self, x):
        a = x
        for weight, bias in zip(self.weights[:-1], self.biases[-1]):
            z = np.dot(a, weight) + bias
            a = relu(z)

        logits = np.dot(a, self.weights[-1]) + self.biases[-1]
        return logits
