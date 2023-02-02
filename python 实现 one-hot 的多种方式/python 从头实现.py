import numpy as np

data = '中国人'
N = len(data)
# 索引化
integer_encoded = [i for i in range(N)]
# one-hot 向量化
onehot_encoded = np.zeros([N, N])
for i in range(N):
    onehot_encoded[i][i] = 1
for c, i, v in zip(data, integer_encoded, onehot_encoded):
    print("token：%s，索引：%d，one-hot 向量：[%s]" % (c, i, " ".join([str(t) for t in v])))
