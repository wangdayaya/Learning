# 前言
本文使用简单的  Transformer Block 实现对影评文本数据 IMDB 的情感分类任务。 
# 数据
1.  这里定义了两个关键的超参数：

    -   `vocab_size`：表示词汇表的大小，即允许在文本数据中使用的不同的单词数量。
    -   `maxlen`：表示文本序列的最大长度，超过这个长度的序列将被截断，不足这个长度的序列将被填充。

1.  使用内置函数从 IMDB 电影评论文本数据集加载训练集和测试集。这是一个情感分类任务，其中评论标签被标记为积极或消极。结果只保留在数据中出现频率最高的前`vocab_size`个单词，其他单词将被忽略。这有助于限制词汇表的大小以降低模型的复杂性。

1.  使用`tf.keras.utils.pad_sequences`函数在训练集和测试集的文本序列末尾填充或截断到指定的`maxlen`长度，以确保它们具有相同的长度，以满足神经网络要求的输入大小。

```
vocab_size = 20000
maxlen = 200

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = tf.keras.utils.pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
x_test = tf.keras.utils.pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')
print(len(x_train), "个训练样本")
print(len(x_test), "个测试样本")
```
结果打印：

```
25000 个训练样本
25000 个测试样本
```

# 嵌入
这段代码定义了一个名为 `Embed` 的自定义 Keras 层，用于实现 `Transformer` 模型中的`嵌入层（Embedding Layer）`。嵌入层将输入序列中的每个 `token` 和它们的`位置`嵌入到一个连续向量空间中，为 Transformer 模型提供适当的输入表示。这是 Transformer 模型中非常重要的一部分，用于捕捉序列数据中的`语义信息和位置信息`，主要参数如下。
-   `maxlen`：表示输入序列的最大长度，用于确定位置嵌入的维度。
-   `vocab_size`：表示词汇表的大小，即不同 token 的数量。
-   `embed_dim`：表示嵌入向量的维度，决定了最后得到的词向量维度。

```
class Embed(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
```

# Transformer Block
这段代码定义了一个名为 `Block` 的自定义 Keras 层，用来表示 `Transformer` 模型中的一个基本运算块（block）。每个基本块包含了两部分：`多头自注意力层（Multi-Head Attention）`和`前馈神经网络层（Feed-Forward Neural Network）`，以及一些为了防止过拟合的`层标准化（LayerNormalization）`、 `Dropout` 和`残差连接`等操作。

该 Block 部分通过多头自注意力和前馈神经网络实现了序列数据的`特征提取`和`表示学习`，经过多个这样的基本块可以层层堆叠形成`编码器`和`解码器`，这里只是简单当作一个提取序列数据的层。主要参数如下：
-   `embed_dim`：表示向量的维度。
-   `num_heads`：表示多头自注意力中的注意力头数量。
-   `rate`：Dropout 丢弃比例，在训练期间应用以防止过拟合。
```
class BLock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(embed_dim)])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(ffn_output + out1)
```

# 训练
这里主要是对模型进行编译和训练，选择 `Adam` 作为模型的优化器，损失函数选择`稀疏分类交叉熵（sparse_categorical_crossentropy）`，评估指标选择`准确率（accuracy）`。这里为了防止过拟合我设置了回调函数 `EarlyStopping` ，这个回调函数在验证集上监视 `val_accuracy` ，并在连续两个 epoch 后指标没有提升时停止训练。
```
model = tf.keras.Sequential([
    Input(shape=(maxlen,)),
    Embed(maxlen, vocab_size, 32),
    BLock(32, 4),
    GlobalAveragePooling1D(),
    Dropout(0.1),
    Dense(20, activation='relu'),
    Dropout(0.1),
    Dense(2, activation='softmax')
])
callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_accuracy')]
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), callbacks=callbacks)
```

结果打印，经过了 `3` 轮训练后提前停止了训练，最终的验证集准确率能达到 `84%` ：

    Epoch 1/10
    782/782 [==============================] - 9s 10ms/step - loss: 0.4148 - accuracy: 0.7963 - val_loss: 0.3230 - val_accuracy: 0.8642
    Epoch 2/10
    782/782 [==============================] - 11s 14ms/step - loss: 0.2207 - accuracy: 0.9165 - val_loss: 0.4794 - val_accuracy: 0.8006
    Epoch 3/10
    782/782 [==============================] - 12s 15ms/step - loss: 0.1476 - accuracy: 0.9477 - val_loss: 0.4001 - val_accuracy: 0.8434