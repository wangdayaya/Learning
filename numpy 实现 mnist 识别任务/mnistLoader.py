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
