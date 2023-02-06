"""
优点：计算每个单词在特定大小的窗口中出现的次数，相较于上一种方式可以大大简化运算量和结果规模，大小为 V * V ，V 是词库大小
缺点：和上一个方法，结果还是稀疏矩阵
"""

import numpy as np
import pandas as pd
from sortedcontainers import SortedSet

docs = ["我是中国人，你是哪国人", "大家是好人，我也是好人", "我在家里蹲，你门也在蹲吗"]
chars = SortedSet()
for s in docs:
    chars |= set(s)
# 为每个字设置索引
c2i = {c: i for i, c in enumerate(chars)}
# 统计基于长度为 1 的滑窗的共现矩阵，也就是每个目标字的左右相邻的 1 个字符远的共现情况
M = len(chars)
window_size = 1
word_doc = np.zeros([M, M])
for d in docs:
    for i,c in enumerate(d):
        window = list(range(max(0, i - window_size), min(i + window_size + 1, len(d))))
        window.remove(i)
        for j in window:
            word_doc[c2i[d[i]], [c2i[d[j]]]] += 1
            word_doc[c2i[d[j]], [c2i[d[i]]]] += 1
word_doc/=2
# 结果用表格形式打印
result = pd.DataFrame(word_doc, columns=[c for c in c2i], index=[c for c in c2i], dtype=int)
print(result)
