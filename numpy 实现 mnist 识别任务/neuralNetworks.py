from util import *
import mnistLoader


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
        dl = derivation_softmax_cross_entropy(logits, y)
        n = len(x)
        dws[-1] = np.dot(_as[-2].T, dl) / n
        dbs[-1] = np.sum(dl, axis=0, keepdims=True) / n
        for i in range(2, self.num_layers):
            dl = np.dot(dl, self.weights[-i + 1].T) * derivation_relu(zs[-i])
            dws[-i] = np.dot(_as[-i - 1].T, dl) / n
            dbs[-i] = np.sum(dl, axis=0, keepdims=True) / n
        return loss, dws, dbs

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
    train_data, val_data, test_data = mnistLoader.load_data()
    model = Network([784, 64, 10])
    model.train(train_data, val_data, 1, 50)
