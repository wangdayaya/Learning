"""
优点：逻辑简单，容易实现
缺点：结果矩阵稀疏，而且最后会形成一个非常大的矩阵，大小为 M * N ，M 是词库大小，N 为文档数量，它的规模是和文档数量成正比关系，大规模数据下，全文档统计是一件非常耗时耗力的事情
"""
import numpy as np
import pandas as pd
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
# 结果用表格形式打印
result = pd.DataFrame(word_doc, columns=['doc%d' % i for i in range(N)], index=[c for c in c2i], dtype=int)
print(result)
