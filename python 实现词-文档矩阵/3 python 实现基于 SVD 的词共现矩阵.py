"""
优点：前两种方法的结果矩阵维度还是太大了，存在维度灾难问题，而且很稀疏，我们可以通过 SVD 来进行矩阵降维
缺点：SVD 计算的复杂度较高，计算开销大，甚至无法计算，而且结果难以解释
"""

# https://blog.csdn.net/xo3ylAF9kGs/article/details/105941914
# https://www.cnblogs.com/sandwichnlp/p/11596848.html
# https://zhuanlan.zhihu.com/p/480389473
# https://www.guyuehome.com/37245

import numpy as np
import pandas as pd
from numpy import linalg as la, mat
from sortedcontainers import SortedSet

docs = ["我是中国人，你是哪国人", "大家是好人，我也是好人", "我在家里蹲，你门也在蹲吗"]
# 所有出现过的字集合
chars = SortedSet()
for s in docs:
    chars |= set(s)
# 为每个字设置索引
c2i = {c: i for i, c in enumerate(chars)}
# 统计 word-doc 矩阵
M = len(chars)
N = len(docs)
word_doc = np.zeros([M, N])
for j, d in enumerate(docs):
    for c in d:
        word_doc[c2i[c]][j] += 1
print(word_doc)
# 奇异值计算
U, Sigma, VT = la.svd(word_doc)
# 重构原始矩阵，只取 k 维作为每个词向量，k 越大，最后的 U_k * Sigma_k * VT_k 越接近原始矩阵
k = 3
Sigma_k = np.zeros([k, k])
for i in range(k):
    Sigma_k[i][i] = Sigma[i]
Sigma_k = mat(Sigma_k)
# 左奇异矩阵的每一行都只有 k 列代表每个词的 k 维特征向量，右奇异矩阵的每一列也只有 k 维表示每个文档的 k 维特征向量，Sigma_k 表示左奇异向量的一行与右奇异向量的一列的对应关系，对角线有 k 个特征值的大小，降序排列，值越大代表该维度重要性越高
U_k = pd.DataFrame(U[:, :k], columns=['dim_%d' % i for i in range(k)], index=[c for c in c2i])
print(U_k)
Sigma_k = pd.DataFrame(Sigma_k)
print(Sigma_k)
VT_k = pd.DataFrame(VT[:k, :], columns=['doc_%d' % i for i in range(N)], index=['dim_%d' % i for i in range(k)])
print(VT_k)
