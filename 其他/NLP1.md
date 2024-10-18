# NLP1




自然语言处理（英语：Natural Language Process，简称NLP）是计算机科学、信息工程以及人工智能的子领域，专注于人机语言交互，探讨如何处理和运用自然语言。


# 其他总结文章

https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/

# 自然语言处理有四大类常见的任务。
- 第一类任务：序列标注，譬如命名实体识别、语义标注、词性标注、分词等；
- 第二类任务：分类任务，譬如文本分类、情感分析等；
- 第三类任务：句对关系判断，譬如自然语言推理、问答QA、文本语义相似性等；
- 第四类任务：生成式任务，譬如机器翻译、文本摘要、写诗造句、图像描述生成等。

# 常见的 NLP 模型

w2v、RNN、Bi-LSTM、Seq2Seq、ATTENTION（SELF、CROSS）、TRANSFORMER、BERT、知识图谱、https://zhuanlan.zhihu.com/p/58931044

# 常见的损失函数

 它衡量模型的预测与标签的偏离程度，损失越小越好

### 1. **`tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`**
   
#### 含义:
- **稀疏分类交叉熵损失**，用于多分类问题，标签是整数形式（而不是 one-hot 编码）。
- `from_logits=True` 表示模型的输出是未经过 softmax 激活的 logits 值，需要在计算损失时应用 softmax。

#### 使用场景:
- **适用于多分类任务**，标签是整数形式（比如 `[0, 1, 2]`），而不是 one-hot 编码（比如 `[[1, 0, 0], [0, 1, 0]]`）。

#### 示例:
```python
import tensorflow as tf

# 标签：整数形式
y_true = [0, 1, 2]  # 表示3个样本的真实类别

# 模型输出：logits（未经过 softmax）
y_pred = [[2.0, 1.0, 0.1], [0.5, 2.0, 1.0], [0.1, 0.5, 3.0]]

# 使用稀疏分类交叉熵损失
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(y_true, y_pred)
print(f"损失: {loss.numpy()}")  # 打印损失
```

#### 输出:
该函数会对每个样本计算损失，然后求平均。

---

### 2. **`tf.keras.losses.CategoricalCrossentropy(from_logits=False)`**

![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/7b5c86837ba646ee80e12476546eb6a3~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=RJVD4rQ26jERtVKtnGedctxye3k%3D)
#### 含义:
- **分类交叉熵损失**，用于多分类问题，标签是 one-hot 编码形式。
- `from_logits=False` 表示模型输出的是经过 softmax 激活的概率分布。

#### 使用场景:
- **适用于多分类任务**，标签是 one-hot 编码（比如 `[[1, 0, 0], [0, 1, 0]]`）。

#### 示例:
```python
import tensorflow as tf

# 标签：one-hot 编码
y_true = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 表示3个样本的真实类别

# 模型输出：经过 softmax 的概率分布
y_pred = [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]

# 使用分类交叉熵损失
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss = loss_fn(y_true, y_pred)
print(f"损失: {loss.numpy()}")  # 打印损失
```

#### 输出:
同样，该函数会计算每个样本的损失并求平均。

---

### 3. **`tf.keras.metrics.SparseCategoricalAccuracy`**

#### 含义:
- **稀疏分类准确率**，用于多分类问题，标签是整数形式。该度量会比较预测类别与真实类别是否相同，并计算准确率。

#### 使用场景:
- **适用于多分类任务**，标签是整数形式（而非 one-hot 编码）。

#### 示例:
```python
import tensorflow as tf

# 标签：整数形式
y_true = [0, 1, 2]

# 模型输出：经过 softmax 激活后的概率
y_pred = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]

# 使用稀疏分类准确率
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
accuracy.update_state(y_true, y_pred)
print(f"稀疏分类准确率: {accuracy.result().numpy()}")
```

#### 输出:
该度量会返回预测类别与真实标签匹配的比例。

---

### 4. **`tf.keras.metrics.CategoricalAccuracy`**

#### 含义:
- **分类准确率**，用于多分类问题，标签是 one-hot 编码形式。该度量会比较预测类别与真实类别是否相同，并计算准确率。

#### 使用场景:
- **适用于多分类任务**，标签是 one-hot 编码形式。

#### 示例:
```python
import tensorflow as tf

# 标签：one-hot 编码
y_true = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# 模型输出：经过 softmax 激活后的概率
y_pred = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]

# 使用分类准确率
accuracy = tf.keras.metrics.CategoricalAccuracy()
accuracy.update_state(y_true, y_pred)
print(f"分类准确率: {accuracy.result().numpy()}")
```

#### 输出:
与 `SparseCategoricalAccuracy` 类似，返回分类任务中预测正确的比例。

---

### 5. **`tf.keras.losses.BinaryCrossentropy(from_logits=False)`**

![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/9c31bfe398e8469f8c2e387fbdcb1fb3~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=coTUd95cEe1gwZaonT2j9yPR%2B%2BQ%3D)
#### 含义:
- **二分类交叉熵损失**，用于二分类问题。`from_logits=False` 表示输入的是经过 sigmoid 激活的概率，而不是原始 logits。

#### 使用场景:
- **适用于二分类任务**，输出的标签是二值（0 或 1），预测值是 sigmoid 激活后的概率值。BinaryCrossentropy：二分类交叉熵，只有预测类和真实类是相等时，loss 才为 0，否则 loss 就是为一个正数。而且概率相差越大 loss 就越大。这个神奇的度量概率距离的方式称为交叉熵。

#### 示例:
```python
import tensorflow as tf

# 标签：二分类
y_true = [0, 1, 0]

# 模型输出：经过 sigmoid 的概率
y_pred = [0.1, 0.9, 0.2]

# 使用二分类交叉熵损失
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
loss = loss_fn(y_true, y_pred)
print(f"二分类交叉熵损失: {loss.numpy()}")
```

#### 输出:
该函数会计算每个样本的损失并求平均，适合二分类任务。

---

### 总结:

| **函数**                           | **含义**                                               | **使用场景**                              | **标签类型**        |
|------------------------------------|--------------------------------------------------------|-------------------------------------------|---------------------|
| `SparseCategoricalCrossentropy`    | 稀疏分类交叉熵，用于多分类问题，标签为整数形式          | 多分类，整数标签                           | 整数                |
| `CategoricalCrossentropy`          | 分类交叉熵，用于多分类问题，标签为 one-hot 编码形式     | 多分类，one-hot 标签                        | one-hot 编码        |
| `SparseCategoricalAccuracy`        | 稀疏分类准确率，标签为整数形式                         | 多分类，整数标签                           | 整数                |
| `CategoricalAccuracy`              | 分类准确率，标签为 one-hot 编码形式                    | 多分类，one-hot 标签                        | one-hot 编码        |
| `BinaryCrossentropy`               | 二分类交叉熵，标签为 0 或 1                            | 二分类，概率标签                           | 二值                |

huber：当δ~ 0时，Huber损失会趋向于MAE；当δ~ ∞（很大的数字），Huber损失会趋向于MSE。


    
![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/37a266fc754d49c796020b0aca9a1a96~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=E960YslL9J7F7dOnO8aqa6kSBtM%3D)
    
MAE(mean absolute error)，即平均绝对值误差

 
 
![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/4f2fc93e4c654c358366fbf939182187~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=VSfvsCYeCVydWaokNYcfNweThUo%3D)

MSE(mean squared error)，即均方误差

 
![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/7189a05f5aae4f5db29cdd119ace2e62~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=1HpTH0r03GMKcDO0CMhlu17phVg%3D)


https://blog.csdn.net/weixin_49346755/article/details/124523232
https://blog.csdn.net/legalhighhigh/article/details/81409551
https://www.cnblogs.com/sun-a/p/13388055.html

# 优化器

优化器将计算出的梯度应用于模型参数以最小化损失函数，损失越低，模型的预测就越好。
Adam、adamw

随机梯度下降 sgd

# 激活函数
激活函数，是在人工神经网络的神经元上运行的函数，负责将神经元的输入映射到输出端。激活函数对于人工神经网络模型去学习、理解非常复杂和非线性的函数来说具有十分重要的作用。它们将非线性特性引入到我们的网络中。在神经元中，输入通过加权，求和后，还被作用了一个函数，这个函数就是激活函数。引入激活函数是为了增加神经网络模型的非线性。若没有激活函数的每层都相当于矩阵相乘。没有激活函数的神经网络叠加了若干层之后，还是一个线性变换，与单层感知机无异。

激活函数给神经元引入了非线性的因素，使得神经网络可以逼近任何非线性函数，可以得到学习和表示几乎任何东西的神经网络模型。常见的有 relu、sigmoid（二分类）、tahn，以及求导。

## sigmoid（二分类）
![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/a8196066248b4d2bb318750d1e162d2b~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=%2FeE4cf2634QpcBcsXmgoH0RWAg8%3D)

缺点：当 z 值**非常大**或者**非常小**时，通过上图我们可以看到，sigmoid函数的导数 g′(z)将接近 0 。这会导致权重 W 的梯度将接近 0 ，使得梯度更新十分缓慢，即**梯度消失**

## tahn


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/a53b10a657e24511b185db41e65659ed~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=MQ6pm0yWOgBcjjI6UDJJJR6lR1E%3D)

tanh函数的缺点同sigmoid函数的第一个缺点一样，当 z **很大或很小**时，g′(z)接近于 0 ，会导致梯度很小，权重更新非常缓慢，即**梯度消失问题**。因此再介绍一个机器学习里特别受欢迎的激活函数 Relu函数。

## relu 


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/12f9e2546a7843efbcba312e4d7cb7c0~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=dXPUzVlb38rzLy8G6MCrNexUIIc%3D)
只要z是正值的情况下，导数恒等于 1，当z是负值的时候，导数恒等于 0。z 等于0的时候没有导数，但是我们不需要担心这个点，假设其导数是0或者1即可。

激活函数选择的经验：**如果输出是0和1的二分类问题，则输出层选择 sigmoid 函数，其他层选择 Relu 函数**

总结一下Relu 激活函数的优点：

Relu 函数在 z>0 的部分的导数值都大于0，并且不趋近于0，因而梯度下降速度较快。

Relu 函数在 z<0 的部分的导数值都等于0，此时神经元（就是模型中的圆圆）就不会得到训练，产出所谓的稀疏性，降低训练出来的模型过分拟合的概率。但即使这样，也有足够多的神经元使得 z>0。

## Leaky Relu


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/0420c1372d9e48f2a9c938fb9b194db8~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=PIRxVkyLH%2Fx%2FC1cISsoe%2FMmbOs0%3D)

https://brickexperts.github.io/2019/09/03/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0/ 


# GroupNorm、LayerNorm、BatchNorm

`GroupNorm`、`LayerNorm` 和 `BatchNorm` 是深度学习中常见的归一化方法，它们的主要作用是在神经网络中对数据进行归一化处理，稳定训练过程，提升模型的泛化能力。虽然它们的目的类似，但它们归一化的方式不同，适合的场景也有所区别。下面详细解释这三个概念及其用法，并通过例子说明它们的工作原理。

### 1. **Batch Normalization (BatchNorm) - 批归一化**

 
![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/76e514c0398648159de48281672bfd6a~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=mp2OTrytt%2BYDwLlMJM4yVGJFaTg%3D)

#### 示例:
```python
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense

# 全连接层后使用 Batch Normalization
model = tf.keras.Sequential([
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

# 卷积层中使用 BatchNorm
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    Dense(10, activation='softmax')
])
```

---

### 2. **Layer Normalization (LayerNorm) - 层归一化**
 
![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/f1a9a27cb04a4a3893eac3f652994594~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=xiTLCNeoscJnZvMptavd6LOVRjE%3D)

#### 示例:
```python
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense

# 全连接层后使用 Layer Normalization
model = tf.keras.Sequential([
    Dense(64, activation='relu'),
    LayerNormalization(),
    Dense(10, activation='softmax')
])
```

---

### 3. **Group Normalization (GroupNorm) - 组归一化**

 
![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/2257b3bcd0ae4d86bbe854a1d7fa2f2b~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=kRTdXdsDNhktKGgYnRrIZzgqJAs%3D)

#### 示例:
```python
import tensorflow as tf
from tensorflow.keras.layers import GroupNormalization, Conv2D

# 卷积层后使用 Group Normalization
model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu'),
    GroupNormalization(groups=4),  # 分为4个组
    tf.keras.layers.MaxPooling2D((2, 2)),
    Dense(10, activation='softmax')
])
```

---

### 总结对比:

| **归一化方法** | **归一化维度**         | **适用场景**                        | **特点**                        |
|----------------|------------------------|-------------------------------------|---------------------------------|
| BatchNorm      | 对整个 mini-batch 的特征维度 | 大批量训练、CNN、全连接层            | 依赖 mini-batch，适合大批量数据 |
| LayerNorm      | 对单个样本的特征维度     | NLP、RNN、小批量或单样本训练         | 不依赖 batch size，效果稳定     |
| GroupNorm      | 分组进行归一化          | 小批量 CNN 训练                     | 不依赖 batch size，灵活性强     |

---

### 总结与选择:

- **BatchNorm**：适用于 **大批量训练**，尤其是卷积网络中。对于小批量训练或单样本，效果不佳。
- **LayerNorm**：更适合 **小批量训练或单样本训练**，尤其是 NLP 和 RNN 中的模型，因其不依赖于 mini-batch 的大小。
- **GroupNorm**：适用于 **小批量训练的卷积网络**，尤其是在 BatchNorm 不稳定的情况下提供了更灵活的选择。

这三种归一化方法在深度学习中都有广泛应用，根据具体的场景选择合适的归一化方法可以提高模型的训练效果和稳定性。



# 权重正则化

Dropout、层正则化、批次正则化、L1、L2：通过对模型参数（权重）的额外约束来减少过拟合的风险。这种约束通常通过在模型的损失函数中添加正则化项实现
- L1：在损失函数中添加权重的绝对值之和乘以一个正的超参数。L1 正则化推动模型参数向零稀疏，促使一些权重变为精确的零，也就是鼓励稀疏模型。
- L2(权重衰减)：在损失函数中添加权重的平方和乘以一个正的超参数。L2 正则化推动模型参数向零，但相对于 L1 正则化，它不会将权重精确地变为零。
- Dropout：正则化技术、训练时期使得部分神经元失活，既可以对输入数据的 Dropout ，又能对隐层的 Dropout

- 为何能解决过拟合？通过 Bagging 方式随机抽取不同的训练数据，训练不同的决策树以构成随机森林，采用投票方式完成决策任务。 而 Dropout 可以被视为一种极端形式的 Bagging ，设置丢弃神经元的概率之后，每次训练数据的时候都相当于是在训练一个新的模型，
- 训练阶段为了保证输入和保持不变，将没有置为 0 的结果值都除了保留概率，保证了输入的和不变，也保证了输出的期望 x 不变

# 文本特征表示



![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/4ee6e2ef4a39490bae1d181feecd3386~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=zr9fVX5dg1BAO7Xwq6HYqLZV66g%3D)


### tf-idf


TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

TF-IDF的主要思想是：如果某个单词在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。


**词频（TF）** **表示词条（关键字）在文本中出现的频率**。

**逆向文件频率 (IDF)** ：某一特定词语的IDF，可以由**总文件数目除以包含该词语的文件的数目**，**再将得到的商取对数得到**。

如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。


**TF-IDF实际上是：TF * IDF**，某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语。

TF-IDF算法非常容易理解，并且很容易实现，但是其简单结构并没有考虑词语的语义信息，无法处理一词多义与一义多词的情况。


### ngram
n-gram 是指从文本中提取的 n 个连续的词或字符组成的子序列，`n` 表示子序列中包含的词或字符的数量。n-gram 用于捕捉上下文信息，n 的值可以是 1（unigram）、2（bigram）、3（trigram）等。

#### 示例：
对于文本 `"猫 喜欢 吃 鱼"`，不同 n 值的 n-gram 示例：
- **unigram（n=1）**: `['猫', '喜欢', '吃', '鱼']`
- **bigram（n=2）**: `['猫 喜欢', '喜欢 吃', '吃 鱼']`
- **trigram（n=3）**: `['猫 喜欢 吃', '喜欢 吃 鱼']`

通过 n-gram，模型能够捕捉到更丰富的上下文信息，如 bigram 可以识别出 "喜欢 吃" 是一个常见的短语。

### one-hot
One-Hot 编码是一种将分类变量转化为二进制向量的方式，用于将离散数据（如单词、类别标签等）表示为计算机可处理的数值形式。对于 n 个类别，one-hot 编码将每个类别表示为长度为 n 的向量，其中一个位置为 1，其他位置为 0。

#### 示例：
假设我们有一个词汇表：`["猫", "狗", "鱼"]`。对于句子 `"猫 喜欢 吃 鱼"`，使用 one-hot 编码表示 "猫" 和 "鱼"：
- **"猫"** 的 One-Hot 编码：`[1, 0, 0]`
- **"鱼"** 的 One-Hot 编码：`[0, 0, 1]`

这种编码方法虽然简单，但对于大规模的词汇表，one-hot 编码会导致稀疏高维向量，因此需要注意其计算效率问题。


### 总结
- **TF-IDF** 用于衡量单词在文档中的重要性，常用于文本分析与信息检索。
- **n-gram** 用于提取文本的连续子序列，捕捉上下文依赖信息。
- **One-Hot** 将分类变量编码为二进制向量，常用于表示离散类别数据。

### 整数编码
### 向量编码

### Warm-start embedding

###  NNLM 


2003年第一次用神经网络来解决语言模型，能够得到语言序列的概率分布，当时语言模型的问题主要是纬度灾难、向量稀疏、计算量巨大，使用 NNLM 能解决这些问题，而且模型能提升语言的泛化性，并且能得到副产品词向量。

        公式 y = b + Wx + Utanh(d+Hx) 共三层输入层、隐藏层、输出层，另外输入层还有个 word embedding 矩阵需要训练，隐藏层的输入需要将所有 lookup 出来的词向量进行拼接

        缺点就是模型结构简单、计算复杂(尤其输出层)，softmax 低效
        
        
    


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/d78a5423ddfe44c2bcd6d18c23ed8fde~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=dZmtmzQhitk5R8x4ZkWV%2F7uPxA8%3D)


![Screen Shot 2024-09-15 at 20.32.37.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/7220611781404155a7a8610a17e1202e~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=JKc31hloZiU2LYU3NNXJF8InvFY%3D)

https://blog.51cto.com/Lolitann/5333935


### GloVe

GloVe的思想和word2vec的思想接近，但是GloVe是从全局的角度出发，构建一个基于上下文窗口的共现计数矩阵，每一次的训练都是基于整个语料库进行分析的，所以它才叫 Global Vector，而 word2vec 则是从一个个句子出发进行训练的

https://blog.csdn.net/jesseyule/article/details/99976041

### Word2vec

**Word2vec 是 Word Embedding 的方法之一**。他是 2013 年由谷歌的 Mikolov 提出了一套新的词嵌入方法。Word2vec，是用来产生词向量的相关模型。这些模型为浅而双层的神经网络，用来训练以重新建构语言学之词文本。模型以词表现，并且需猜测相邻位置的输入词，在word2vec中词袋模型假设下，词的顺序是不重要的。训练完成之后，word2vec模型可用来映射每个词到一个向量，可用来表示词对词之间的关系，该向量为神经网络之隐藏层。

## Word2vec 的 2 种训练模式

CBOW(Continuous Bag-of-Words Model)和Skip-gram (Continuous Skip-gram Model)，是Word2vec 的两种训练模式。下面简单做一下解释：

**CBOW**

通过上下文来预测当前值。相当于一句话中扣掉一个词，让你猜这个词是什么。


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/b85e7af851fd4ed6992530a80532dd0e~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=7%2FuShrOmWdi5w7hI%2BsNuD43wtUY%3D)


**Skip-gram**

用当前词来预测上下文。相当于给你一个词，让你猜前面和后面可能出现什么词。


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/98ba159e52494f72add359ec27ed2c15~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=BZtgNod2cEx5uMXSWJG7W4LkkJg%3D)

**优化方法**

为了提高速度，Word2vec 经常采用 2 种加速方式：

0.  Negative Sample（负采样）
0.  Hierarchical Softmax


优点：

0.  由于 Word2vec 会考虑上下文，跟之前的 Embedding 方法相比，效果要更好（但不如 18 年之后的方法）
0.  比之前的 Embedding方 法维度更少，所以速度更快
0.  通用性很强，可以用在各种 NLP 任务中

![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/4e51c02d2380408f866941d7b64956a8~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=C4cqSb5W7vztEqYxHKqcilve4Yo%3D)

缺点：

0.  由于词和向量是一对一的关系，所以多义词的问题无法解决。
0.  Word2vec 是一种静态的方式，虽然通用性强，但是无法针对特定任务做动态优化


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/6552c798abde4d17b759ff54d8362c89~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=HrGBdQ8OlNhUBAn7peZMkTFetaY%3D)


这种方式在 2018 年之前比较主流，但是随着 [BERT](https://easyai.tech/ai-definition/bert/)、GPT 系列的出现，这种方式已经不算效果最好的方法了。



31. 预训练模型
32. 迁移学习、微调、局部微调、整体微调

# 过拟合和欠拟合
- 过拟合：过拟合是指模型在训练数据上表现得比在测试数据上更好，但当面对未见过的数据时，性能却下降的现象，原因有：训练数据太少、特征冗余、数据噪声太多、数据分布不均匀、模型太复杂、训练时间太长。解决办法：充足有效的训练数据、降低模型复杂度、数据归一化、权重正则化（L1、L2、Dropout）、数据增强。
- 欠拟合：欠拟合是指模型在训练和测试数据上的性能都较差，未能很好地捕捉数据的模式，原因有：数据特征不足、数据量太小、模型复杂度低、训练时间短、超参数选择不当（学习率）

# 超参数选择策略
超参数选择或者超参数调整，主要包括模型和算法超参数，Keras Tuner 中有四种工具可选：`RandomSearch`, `Hyperband`, `BayesianOptimization`, and `Sklearn`，先使用训练数据找出最优超参数，然后使用训练数据用找出来的最优超参数去重新训练模型

# 前向传播和反向传播

# RNN
RNN：权重计算 https://juejin.cn/post/6972340784720773151

    -   输入门权重矩阵 `W_i`：`units * (input_dim + units)`
    -   输入门偏置向量 `b_i`：`units`
    -   输出权重矩阵 `W_o`：`units * units`
    -   输出偏置向量 `b_o`：`units`
    -   Total 参数数量为上述各项之和。

# LSTM
LSTM：权重参数计算 (units * (input_dim + units)  + units ) * 4 

https://juejin.cn/post/6973082167970627620

    -   输入门权重矩阵 `W_i`：`units * (input_dim + units)`
    -   输入门偏置向量 `b_i`：`units`
    -   遗忘门权重矩阵 `W_f`：`units * (input_dim + units)`
    -   遗忘门偏置向量 `b_f`：`units`
    -   细胞状态权重矩阵 `W_c`：`units * (input_dim + units)`
    -   细胞状态偏置向量 `b_c`：`units`
    -   输出门权重矩阵 `W_o`：`units * (input_dim + units)`
    -   输出门偏置向量 `b_o`：`units`
    -   Total 参数数量为上述各项之和。

![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/bf9c8e82555241bcb9a1b3de59a49649~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=keEWpMhUQTWyRf0%2BV5SDjQky580%3D)
![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/9421f89ae19a4914aa8372caaca254f2~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=Bcf1t2uoyymhn1KU2MGiGpRF3R0%3D)

# GRU
 **GRU 单元：** reset_after=False 也就是 SimpleRNN， 3 * (units * (input_dim + units) + units)

    -   重置门权重矩阵 `W_r`：`units * (input_dim + units)`
    -   重置门偏置向量 `b_r`：`units`
    -   更新门权重矩阵 `W_z`：`units * (input_dim + units)`
    -   更新门偏置向量 `b_z`：`units`
    -   新状态权重矩阵 `W_h`：`units * (input_dim + units)`
    -   新状态偏置向量 `b_h`：`units`

    CuDNNGRU 為了加速計算因此計算方式與一般 RNN 有稍微不同 tf2 默认 reset_after=True ，计算每个门的时候多一组偏置项。  3 * (units * (input_dim + units) + units + units)

    -   重置门权重矩阵 `W_r`：`units * (input_dim + units)`
    -   重置门偏置向量 `b_r`：`units + units`
    -   更新门权重矩阵 `W_z`：`units * (input_dim + units)`
    -   更新门偏置向量 `b_z`：`units + units`
    -   新状态权重矩阵 `W_h`：`units * (input_dim + units)`
    -   新状态偏置向量 `b_h`：`units + units`


# 文本生成

训练数据制作、模型选择、选则token三种方式（选最大，按照概率分布抽样、温度控制temperature选择1则退化为第一种方式）


# seq2seq 

### 优化技巧

    - 使用更加复杂度的 RNN 变体
    - 双向的特征拼接
    - 更优化的预训练 Embedding
    - 使用 Attention
    - 如果是翻译类任务，可以使用不同语种翻译训练过程，提升某一语种的翻译性能
    
### seq2seq 相似度计算

加入 attention ，时间复杂度 `O(m+m\*t)`，没有 attentino 时间复杂度为 `O(m+t)` 。
    - decoder 时候使用两个参数矩阵对状态 si 与每个时刻的状态输出 hi 进行非线性变化，然后进行 softmax
    - decoder 时候使用算出 hi 和 si 的非线性变化 ki 和 qi ，然后两者求内积，最后进行 softmax

### Seq2Seq 做机器翻译

当输入句子的单词在 20 个附近时的效果最佳，当输入句子的单词超过 20 个的时候，效果会持续下降，这是因为 Encoder 会遗忘某些信息。而在 Seq2Seq 的基础上加入了 Attention 机制之后，在输入单词超过 20 个的时候，效果也不会下降。

### self-attention

### cross-attention


# BERT ：Bidirectional Encoder Representations from Transformers 

其实 BERT 的目的就是预训练 [Transformers](https://link.juejin.cn/?target=https%3A%2F%2Farxiv.org%2Fpdf%2F1706.03762.pdf "https://arxiv.org/pdf/1706.03762.pdf") 模型的 Encoder 网络。三个基本任务分别是：对遮盖的 15% token 进行预测 ，判断两句话是不是相邻的（包括正负样本），第三个任务是混合起来前两个。

    110M 参数-340M参数 ，主要区别在 layer, hidden, heads
    
    
### BERT 输出
BERT 输出有 word_id（包含了 \[CLS\]:101 ，\[SEP\]: 10），mask，type_id ，可以输出 pool value， encoded value（每一层Block 的输出，包括最后一层的输出【B，S，H】）

    建议从参数较少的小型 BERT 开始使用，因为它们的微调速度更快。 如果您喜欢小型模型但精度更高，可以用 ALBERT 。 如果您想要更高的准确性，可以用经典的 BERT 之一如 Electra、Talking Heads 或 BERT Expert。除此之外还有规模更大精度更高的版本，但是无法在单个 GPU 上进行微调，需要使用 TPU 。

# GPT- Generative Pre-trained Transformer

![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/2ba10af6316d41369aeff17edb19b9cd~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=ZaWBm7HO%2BVfDHus77B7gKg2QumU%3D)

### GPT 
2018年，参数量 1.17 亿 ，训练数据 5GB，。 由 12 层简化的 Transformer 的 Decoder Block 组成，每个 Block 中有 Masked Multi self-Attention 和 Dense ，序列长度 512 。GPT-1的训练分为无监督的预训练和有监督的模型微调，而需要注意的另外一个细节，是fine-tune的loss函数，既包含了下游任务的loss，也包含了语言模型的loss（预测下一个单词），这么做的目的是在做垂直领域任务的时候，保持着自己本身的这种语言模型的性质，不要把语言本身给忘掉。

主要解决得是监督任务得两个主要弊端：需要大量标签数据，模型领域化无法真正泛化去理解语言。通过对输入和输出的不同处理方式，在很多任务上 SOTA ，可以完成文本分类、相似度计算、问答、推理等任务。



### GPT-2
2019年，结构上无太大变化，只是使用了更多的网络参数和更大的数据集，在 40 G 的 WebText 数据集上进行训练，目标是使用无监督的预训练模型做有监督的任务，作者认为当一个语言模型的容量足够大时，它就足以覆盖所有的有监督任务，即训练完翻译的任务后，也就是学会了该数据范围的问答。参数量最低 117M 参数，最大 1542M 参数量，约 15 亿，主要区别是Block 的 layer 和 hidden ，序列长度扩大到 1024。在众多任务中获得 SOTA ，初步具备了zero-shot 和 few-shot能力。

GPT-2 进行模型调整的主要原因在于，随着模型层数不断增加，梯度消失和梯度爆炸的风险越来越大。 GPT-2的最大贡献是验证了通过海量数据和大量参数训练出来的 GPT 模型有迁移到其它类别任务中而不需要额外的训练的能力。


### GPT-3
2020年，参数量 1750 亿，数据 45 TB 过滤出 570GB ，模型结构没有变化，只是注意力头增多，具备 zero-shot 或者 few-shot ，在下游任务中表现出色，从理论上讲GPT-3也是支持fine-tuning的，但是fine-tuning需要利用海量的标注数据进行训练才能获得比较好的效果，但是这样也会造成对其它未训练过的任务上表现差，所以GPT-3并没有尝试fine-tuning。

    GPT-3的本质还是通过海量的参数学习海量的数据，然后依赖transformer强大的拟合能力使得模型能够收敛。基于这个原因，GPT-3学到的模型分布也很难摆脱这个数据集的分布情况。
    
缺点有：
- 输出错误或者不合逻辑，对于 PROMPT 无法正确理解
- 可能包含一些非常敏感的内容，例如种族歧视，性别歧视，宗教偏见等；
- 受限于transformer的建模能力，GPT-3并不能保证生成的一篇长文章的连贯性，存在下文不停重复上文的问题。
- 数据量和参数量的骤增并没有带来智能的体验。
    
### GPT3.5（InstructGPT）
GPT-3纵然很强大，但是对于人类的指令理解的不是很好，3H 指出了训练 GPT3.5目标是Helpful、Honest、Harmless，原文分为了三个步骤进行训练：有监督微调SFT，奖励模型训练RM，强化学习训练RLHF；实际上可以把它拆分成两种技术方案，一个是有监督微调（SFT），一个是基于人类反馈的强化学习（RLHF）

![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/8bedd44e7b774ef8a31a1c5705bbe1ba~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=nwJ1%2BW%2BFbCA07ciI5CBJiAZPyf8%3D)


SFT:人类标注的<prompt,answer>数据来Fine-tune GPT-3模型，以使其初步具备理解人类prompt中所包含意图，并根据这个意图给出相对高质量回答的能力。这一步骤中包含了1.2万条训练数据。

RM：RM结构是将SFT训练后的模型的最后的嵌入层去掉后的模型，是一个回归模型。它的输入是prompt和Reponse，输出是奖励值。通过人工标注数据来训练回报模型。 随机抽样prompt，并使用第一阶段Fine-tune好的冷启动模型，生成K个不同的回答。然后，标注人员根据相关性、信息性和有害信息等标准，对K个结果进行排序，生成排序结果数据。接下来，研究者使用这个排序结果数据进行pair-wise learning to rank训练模式，训练回报模型。RM模型接受一个输入<prompt,answer>，给出评价回答质量高低的回报分数Score。对于一对训练数据<answer1,answer2>，假设人工排序中answer1排在answer2前面，那么Loss函数则鼓励RM模型对<prompt,answer1>的打分要比<prompt,answer2>的打分要高。

RLHF：随机抽取 PROMPT，让模型来生成回答。用RM模型对回答进行评估，并给出一个分数作为回报。训练LLM模型生成的答案能够获得高分数，根据得到的回报分数来使用 PPO 算法更新模型的参数，以便让模型生成更好的回答。

PPO 的核心思想之一是通过对比新策略和旧策略之间的差异来更新策略。训练策略网络时，通过最大化优势函数，即新策略相对于旧策略的增益来更新网络参数。但在更新时，为了保持策略的变化不会过大通常会引入一个 Kullback-Leibler（KL）散度的约束，也就是让LM的输出在RM的评估体系中拿到高分，同时不能失去LM输出通顺文本能力，即优化后的模型输出和原模型输出的KL散度越接近越好。而chatgpt 中的策略函数就是预训练GPT得到的生成模型 ， 价值函数则是后续通过人工标注数据训练的打分模型。当然chatgpt对PPO损失做了一定的修改，增加了语言生成模型的损失，希望生成模型生成的句子越通顺越好。


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/4041553fb2854026a0710990d5b30587~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=h61DCz3%2BUkfpbKkc8fgD7hgj%2F0w%3D)

51. 指示学习（Instruct Learning）和提示（Prompt Learning）学习

    指示学习和提示学习的目的都是去挖掘语言模型本身具备的知识。指示学习的优点是它经过多任务的微调后，也能够在其他任务上做 zero-shot ，而提示学习都是针对一个任务的,泛化能力不如指示学习:

    提示学习：激发语言模型的**补全能力**,如：给女朋友买了这个项链，她很喜欢，这个项链太XX了。
    指示学习：激发语言模型的**理解能力**,如：这句话的情感是非常正向的：给女朋友买了这个项链，她很喜欢。

52. Transformers:Transformer 是 Seq2Seq 模型，包括了一个 Encoder 和一个 Decoder 。Transformer 不是 RNN 模型，它没有循环结构。Transformer 完全基于 Attention 和 Self-Attention 搭建而成。

    - Transformer 的 Encoder 网络就是靠 6 个 Block 搭建而成的，每个 Block 中有 Multi-Head Self-Attention Layer 和 Dense Layer ，输入序列大小是【d，m】，输出序列大小是【d，m】。
    - Transformer 的 Decoder 网络也是靠 6 个 Block 搭建而成的，每个 Block 中有 Masked Multi-Head Self-Attention Layer、 Multi-Head Attention Layer 和 Dense Layer ，输入序列有两个，一个序列大小是【d，m】，另一个序列大小为【d，t】，输出序列大小是【d，t】。

53. 维特比算法
54. XLA（Accelerated Linear Algebra）是一种针对线性代数计算的加速器，旨在优化高性能计算和深度学习模型的执行速度。XLA 可以优化计算图，使其更适应硬件加速器，提高深度学习模型的训练。XLA（Accelerated Linear Algebra）默认是启用的，而且无需额外的配置。TensorFlow 2.x 在默认情况下会自动使用 XLA 来优化计算图，以提高深度学习模型的执行效率。
55. PolynomialDecay

![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/67cef1ac82a4481eaa3e2edbe41a2f12~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=iMgr6y38ANmtmoLvfDijSS0LwFM%3D)

56. 迁移学习（Transfer Learning）、微调（fine-tuning）、one-shot、few-shot

    **迁移学习（Transfer Learning）** 通过在一个任务上学到的知识来改善在另一个相关任务上的性能。源领域和目标领域之间的任务通常是相关但不同的。如翻译，例如学会了骑自行车的人也能较快的学会骑电动车。较为常用的一种迁移学习方式是利用预训练模型进行微调。为了解决下面问题：

    -   一些研究领域只有少量标注数据，且数据标注成本较高，不足以训练一个足够鲁棒的神经网络。
    -   大规模神经网络的训练依赖于大量的计算资源，这对于一般用户而言难以实现。
    -   应对于普适化需求的模型，在特定应用上表现不尽如人意。

    **微调（Fine-tuning）** 是迁移学习的一种具体实践方式。在微调中，我们首先在源领域上训练一个模型，然后将这个模型应用到目标领域上，但不是从头开始训练。相反，我们对模型的一部分（或全部）进行调整以适应目标领域的任务。通常，这包括冻结模型的一些层，然后对未冻结的层进行微小的调整。

    **联系：**

    -   微调是迁移学习的一种策略，迁移学习并不仅限于微调,其他迁移学习的策略包括特征提取、共享表示学习等。微调是其中最常见的一种方法之一，特别在深度学习中，通过在预训练模型的基础上进行微调，可以更好地适应目标任务。
    -   在迁移学习中，微调通常是指在源任务上训练的模型参数的调整，以适应目标任务。
    -   微调涉及到从一个任务到另一个任务的知识传递，这与迁移学习的目标一致。

    **One-shot Learning：**: 在一次学习中，模型只能从一个样本（一张图片、一个数据点等）中学到目标任务的知识。 通常应用于样本非常有限、甚至只有一个样本的情况，要求模型能够在仅有极少信息的情况下完成任务。

    **Few-shot Learning：** ： 在少次学习中，模型从几个（通常是很少的）样本中学到目标任务的知识。Few-shot 可能包含从两到几十个样本的情况。 使用于数据较为稀缺的情况，但相对于 One-shot Learning 来说，Few-shot Learning 允许更多的样本用于训练，提供了一定的上下文和信息。

    **zero-shot** 则是不提供示例，只是在测试时提供任务相关的具体描述。实验结果表明，三个学习方式的效果都会随着模型容量的上升而上升，且few shot > one shot > zero show。


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/96567c4abd154c34989ffc3a49542453~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=aAyYnrc%2BGqrhPcQ2NO4AJ0B7v9w%3D)
57. 语言模型是这样一个模型：**对于任意的词序列，它能够计算出这个序列是一句话的概率** https://zhuanlan.zhihu.com/p/32292060
58. 典型的大规模数据处理流程


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/e113d817a8534268a89fb14d6fa4c9cb~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=%2B10LzAwX0%2Bi7kpPPDlb8wXqWvIw%3D)
59. sampling methods  ： https://zhuanlan.zhihu.com/p/453286395
    - GreedySampler
    - BeamSampler
    - Temperature Sampling
    - RandomSampler
    - TopKSampler
    - TopPSampler
    
 
 在解码采样生成序列的过程中，有多种算法可以决定下一个生成的元素（例如单词或字符）。以下是一些常见的采样算法及其原理和使用场景：

### **贪婪采样（Greedy Sampling）**
   - **原理**：在每一步，选择概率最高的token作为下一步的输出。
   - **使用场景**：当生成速度是关键要求，或者只需要非常确定性的输出时使用。
   - **优点**：简单，快速。
   - **缺点**：可能陷入局部最优，缺乏多样性。

### **束搜索（Beam Search）**
   - **原理**：在每一步，保留前k个最可能的序列，并在此基础上扩展，最终选择得分最高的序列。
   - **使用场景**：需要平衡速度和多样性的场景，例如机器翻译。
   - **优点**：比贪婪采样更有可能找到全局最优解。
   - **缺点**：计算成本随束宽增加而增加。

### **随机采样**
   - **原理**：根据概率分布随机选择token。
   - **使用场景**：需要创造性文本或训练变分自编码器（VAE）时使用。
   - **优点**：可以生成更多样化的文本。
   - **缺点**：难以控制输出质量，可能生成无意义的文本。

### **Top-k Sampling**
   - **原理**：在每一步，从概率最高的k个token中随机选择一个。
   - **使用场景**：需要在贪婪采样和随机采样之间取得平衡，增加多样性。
   - **优点**：比贪婪采样有更高的多样性，同时避免了随机采样的不确定性。
   - **缺点**：可能会生成不太可能的token。

### **Top-p Sampling（Nucleus Sampling）**
   - **原理**：累积概率分布，选择累积概率小于p的token集合，然后从这个集合中根据概率分布进行采样。
   - **使用场景**：需要生成更连贯和有意义的文本。
   - **优点**：能够生成更多样化且连贯的文本。
   - **缺点**：计算上比贪婪采样复杂。

**哪个算法比较好？**

这取决于具体的应用场景和需求：
- 如果需要快速且确定性的输出，**贪婪采样**可能是最佳选择。
- 如果需要考虑更多可能性并提高生成质量，**束搜索**是一个更好的选择。
- 如果需要高度创造性的文本，**随机采样**可能更合适。
- 如果需要在多样性和确定性之间取得平衡，可以尝试**Top-k Sampling**。
- 如果需要生成高质量且连贯的文本，**Top-p Sampling**可能是最好的选择。

在实际应用中，可能需要根据模型的特点和具体任务的需求，调整采样策略以达到最佳效果。


  
 
        
        
        
# Elasticsearch 
1. 倒排索引
2. BM25
3. 架构
4. 优化技巧
5. 



# python
1. 基本语法
2. flask + gunicorn


# 神经网络压缩与加速:

1. 减少其存储和计算成本变得至关重要
2. 不是所有神经元都起作用，仅仅只有少部分（5-10%）权值参与着主要的计算，仅仅训练小部分的权值参数就可以达到和原来网络相近的性能
3. 在移动设备上运行


    ![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/1a3087cd2ade4de2a7977266629cb703~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=2rwKsHyRWHJIi6LkAriLAdG1LoM%3D)

# 机器学习十大常见算法（唐宇迪）


    

# CPU/GPU推理加速

1. 量化
2. 剪枝
3. ONNX-CPU
4. ONNX-GPU


# Tensorflow 框架
1. 分布式训练
2. 剪枝
3. 量化
4. 保存模型，只保存权重，保存模型和权重，模型可以保存为 .keras 、SavedModel、.h5，如果需要在 TensorFlow 生态系统中进行跨平台部署，则推荐使用 SavedModel 格式；如果只是保存模型结构和权重，HDF5 可以是一个不错的选择。 Keras 格式则提供了在 TensorFlow 中使用 Keras API 时的一种保存和加载方式。
5.  NumPy 数组和 tf.Tensors 之间最明显的区别是：一是张量可以由加速器内存（如 GPU、TPU）支持。二是张量是不可变的。
20. cache() ：在整个数据集上进行缓存，以便在多次迭代中重复使用数据，从而提高性能。
21. prefetch :在迭代数据时，可以在 CPU 处理当前批次的数据的同时，异步预加载下一批次的数据到 GPU 上，提高数据加载和模型训练的并行度。
22. Multi-GPU distributed training：https://keras.io/guides/distributed_training_with_tensorflow/
 




# LLAMA 
 

## 摘要

- 一般来说更多的参数将导致更好的性能，但是实际证明在给定计算预算的情况下，**最佳性能并非由最大的模型实现，而是由在更多数据上训练的小型模型实现**，一个7B模型在超过1T个token的训练效果比一个用200B个token训练的10B模型好。这一现象打破了“模型越大性能越好”的刻板印象，还提供了更加高效且经济的模型训练策略。算力紧缺的今天，性能优秀的小模型显得尤为可贵。
- 给定目标性能水平，尽管训练一个大型模型以达到一定的性能水平可能更便宜，但一个经过更长时间训练的小型模型在推理时最终会更经济。
- LLaMA 参数范围从**7B到65B**，与现有最佳的大型语言模型（LLMs）相比具有竞争力
- 我们仅**使用公开可用的数据**，这使得我们的工作可以与开源兼容
- **LLaMA-13B vs GPT-3：** **尽管LLaMA-13B（130亿）的规模仅为GPT-3（1750亿参数）的十分之一，但在大多数****基准测试****中，其性能却优于GPT-3。**
-   **LLaMA-7B：** 可以在**单个** **GPU** **上运行**， 使大模型的获取和研究更容易，而不再只是少数几个大厂的专利。
-   **LLaMA-65B：** 在高端系列上，LLaMA-65B 也与最佳的大语言模型（如 Chinchilla 或 PaLM-540B）性能相当。


## 训练数据


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/1cb302a7b5ca483baa8b32e2946f961f~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=p0CRK1V2%2FFt35%2FTTsf8uZvZFuFY%3D)

- 2017年到2020年5年的web爬虫数据 CommonCrawl 、从 CommonCrawl 中过滤出来的 C4、github、wiki、books 、arxiv、高质量问答数据 stackexchange
- 使用字节对编码（BPE）算法对数据进行分词，具体使用了SentencePiece的实现。值得注意的是，我们将所有数字拆分为单个数字，并使用字节来分解未知的UTF-8字符。大模型一般都会有专门的分词表（tokenizer.json）来定义词汇与整数之间的映射关系
- 训练数据集有几种不同来源，涵盖了多个领域。
- 整个训练数据集在 tokenization 后包含大约 1.4T 个 token。
- 对于大多数训练数据，每个 token 在训练期间仅使用一次（1 epochs）；
- 维基百科和书籍是个例外，会被使用两次（2 epochs）。


## 架构变化

我们的网络采用了转换器（transformer）架构（Vaswani等人，2017），取消Encoder，仅保留Decoder：这一设计使得模型结构更为简洁，专注于生成和解码任务。另外借鉴了随后提出的不同模型（如PaLM）中的各种改进方法。

![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/14d32df6dfa046b0bc723c33fb6b5a02~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=NYr11ae%2FPWUnmxEowcAcZa4tgCI%3D)

### RMSNorm归一化函数
有助于提高模型的训练稳定性和收敛速度，我们对每个转换器子层的输入进行归一化，而不是对输出进行归一化。

RMSNorm是一种特定的归一化方法，它采用了均方根（Root Mean Square）的方式来计算缩放因子。通过对输入向量进行RMSNorm处理，可以使得不同尺度的特征具有相同的权重，从而提高了模型的稳定性和泛化能力。


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/b3525de5094847eda8c9a04ec892bdc7~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=h070mV7WO0XmwWnpiTPT93PBCsk%3D)

### SwiGLU激活函数

我们用SwiGLU激活函数替代ReLU非线性函数，这是由Shazeer（2020）提出的，旨在提高性能。我们使用的维度是2/3*4d，而不是PaLM中的4d。

### 旋转位置嵌入（RoPE）

移除了绝对位置嵌入，在Q和K上使用RoPE旋转式位置编码：这种位置编码方式能够更好地捕捉序列中的位置信息，提高模型的表达能力。


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/4dcc59958abd49cb8bc67f431782ee8d~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=rvAd7u2tH8Jk0ORB%2BKLq%2BxPZADo%3D)




### 优化器
-   使用 AdamW 优化器（Loshchilov 和 Hutter，2017）对模型进行训练，具体超参数 ：β1=0.9,β2=0.95。
-   使用一个  cosine learning rate schedule，最终的 学习率 达到了最大学习率的 10％。 
-   使用 0.1 的权重衰减（weight decay）和 1.0 的梯度裁剪（ gradient clipping ）。 
-   使用 2,000 个 warmup steps，并根据模型大小来调整 learning rate 和 batch size ，如图所示。

![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/29db0669f0c948909875fa72eb24a40b~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=l20vTdVmiDRZBfJ6Deu50PjxM6g%3D)


## 优化训练速度

-   首先，使用 **causal multi-head attention** 的一个高效实现来减少内存占用和运行时。 优化原理：由于语言建模任务存在因果特性，因此可以不存储注意力权重（attention weights），不计算那些已经被掩码（masked）的 key/query scores。
-   为进一步提高训练效率，我们通过 checkpoint 技术， 减少了在反向传播期间需要重新计算的激活数量。
-   保存了计算成本高昂的激活，例如线性层的输出。实现方式是手动实现 Transformer 层的反向函数，而不用 PyTorch autograd。
-   为了充分受益于这种优化，需要通过模型和序列并行（model and sequence parallelism）来减少模型的内存使用。

优化成果：
-   这些优化使得在训练 65B 参数的模型时，在  2048 个 A100 80 GB GPU  上能处理约 380 tokens/second/GPU 
-   这意味着  1.4T token  的数据集上训练大约需要  21 天 。


#  LLAMA2
![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/68852d62fa684f20a4a85fda624b0faf~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=P%2BRc6ulyVen0pn2rFopKYFhHfVY%3D)

-   三种尺寸供选择：7B、13B和70B。其中，7B和13B沿用了Llama 1的经典架构，而70B模型则采用了创新的分组查询注意力（GQA）架构。
-   LLAMA2 在2万亿个标记上进行了训练，且上下文长度是 Llama 1 的两倍。而经过微调的模型，也在超过100万条人工标注的数据上进行了精进训练。
-    观察了 Llama 2模型的评估性能，发现它在许多外部基准测试中都优于其他开源语言模型。这些测试涵盖了推理、编码、熟练度和知识等多个方面。
 
![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/6d06cf3f2cfc4ebb8b1c79b22147a6ca~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=krlO02XvX%2FKBW3sLgyFMBicDD%2Bk%3D)

## 改进


### 什么是 kv cache？

KV Cache（键值缓存）是一种在Transformer模型的解码器推理过程中使用的优化技术。它主要针对自回归模型，在生成文本时，通过缓存之前计算的键（Key）和值（Value）信息，减少重复计算，从而提高推理效率。

在自回归推理中，模型每次生成一个新的token，都会将之前的文本和新生成的token一起输入模型进行下一次的计算。这样做效率很低，因为之前的token对应的键值对信息在每次推理时都会被重复计算。KV Cache技术通过存储这些键值对信息，使得在生成新token时，只需要考虑当前token与已缓存键值对的交互，而不需要重新计算之前的token信息。

KV Cache的工作原理可以概括为以下几点：
1. 在模型的推理过程中，为每个token计算查询（Query）、键（Key）和值（Value）。
2. 将计算得到的键和值缓存起来，存储在所谓的`past_key_values`中。
3. 当生成下一个token时，只计算这个新token的查询，然后使用缓存中的键和值来计算注意力得分和输出。
4. 随着生成的token不断增加，缓存中的键和值也会不断更新，以包含新token的信息。

KV Cache的优势在于：
- 提升推理速度：通过避免重复计算，减少了计算量，从而加快了生成速度。
- 减少计算资源消耗：由于减少了重复计算，相应地也减少了浮点运算数（FLOPs）。
- 保持模型性能：缓存的键值对信息不会影响模型的最终输出，因此不会降低模型性能。

然而，KV Cache也会占用一定的内存空间，因为需要存储键和值信息。随着序列长度的增加，占用的内存也会线性增长。尽管如此，KV Cache在实际应用中仍然是一个非常有效的优化手段。
 

 

### 分组查询注意力（GQA）

- https://blog.csdn.net/baoyan2015/article/details/137968408
- https://zhuanlan.zhihu.com/p/667259791


Llama 2 在 70B 模型上采用了使用了 Group query attention 来节省 cache，这一优化减少了模型的计算量和内存占用，提高了模型的效率。


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/d05b1a898fb74afeab2896be4bd96600~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=w5hB7bQjzVny5cFh5BISiPnp%2BKo%3D)

GQA有三种变体：

-   GQA-1：一个单独的组，等同于 Multi-Query Attention (MQA)。所有的Q共享同一套K和V 。
-   GQA-H：组数等于头数，基本上与 Multi-Head Attention (MHA) 相同。每个注意力头都有自己的一套完整的Q（Query）、K（Key）、V（Value）
-   GQA-G：一个中间配置，具有G个组，平衡了效率和表达能力。每组共享K和V，但每组内不同的Q仍然有自己的Q矩阵。


出现背景：

- MQA 虽然推理快，但是有明显的质量下降问题，MHA 质量好，但是推理速度慢，所以折中之后诞生了 GQA 。
- 减少头的数量，也就是减少kv cache的size，达到减小带宽的压力的目的，那么MQA推理速度势必更快。

效果：

性能提升主要来自于kv cache的size减小，那么kv cache占用的显存就变小，那么我们LLM服务可以处理的请求数量就更多，batchsize更大，吞吐量就变大。












































# 其他
- `leetcode 经典 200 题`
- 掘金、CSDN
- stable diffusion 
- 论文，架构，人才，年度优秀员工
- 熟悉主流 LLM 框架结构以及核心算法原理，如 GPT、LLAMA、QWEN 等，以及 agent、rag、rlhf 等
- 数据制备（数据清洗、去重、过滤、数据质量指标筛选）、模型训练优化、模型推理优化、性能评测和迭代、服务部署
- 常见的微调技术、LORA、QLORA、RLHF 等
- 常见的语言模型 tansformer 、bert、gpt、 t5 等
- 熟悉 deepspeed 、accelerator 等主流并发多机多卡训练框架 
- 文生图的理论知识和项目经验
- https://github.com/wangdayaya/aimoneyhunter


![image.png](https://p3-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/55300c487e7546eda4c09f79185c1fbc~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1728633495&x-signature=G2W2e9riuXGqa16F6J%2Bb9pliwi0%3D)