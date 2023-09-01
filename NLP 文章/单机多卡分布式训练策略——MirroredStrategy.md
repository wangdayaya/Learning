# 前言
分布式训练是一种用于在多个`设备`或`机器`上同时训练深度学习模型的技术，它有助于`减少训练时间`，允许使用更多数据`更快训练`大模型。分布式训练重点关注`数据并行性`，本次试验使用的是`单机多卡`的分布式训练策略，也就是 `MirroredStrategy` 。通常单台机器上的有 1-8 个 GPU ， 这也是研究人员和小公司最常见的配置。

# MirroredStrategy 简介
 tf.distribute.MirroredStrategy 的步骤如下：

-   训练开始前，该策略在 N 个 GPU 上各复制一份完整的`模型备份`；
-   每次训练传入一个 batch 的数据，将数据`分成 N 份`，分别传入 N 个计算设备（即数据并行）；
-   N 个计算设备在自己的内存中分别计算自己所获得数据的模型`梯度`；
-   使用分布式计算的 `All-reduce` 操作，在所有 GPU 间高效交换梯度数据并进行`求和`，使得最终每个设备都有了所有设备的梯度之和；
-   使用梯度求和的结果更新各个 GPU 中的模型权重；
-   因为该策略是`同步`的，所以只有当所有设备均更新模型后，才进入下一轮训练。

# 虚拟出 4 个 2G 的GPU
1. 这里先查找主机系统中可用的物理 GPU 设备，因为我这里只有一块 `4090` ，所以结果肯定是包含只有一个物理 GPU 的列表，并将它们存储在 `physical_devices` 列表中。
1.  将我们唯一的物理 GPU 设备 `physical_devices[0]` 划分成了四个虚拟 GPU 设备，每个虚拟 GPU 的内存限制被设置为 `2048MB` 。这样是为了模拟一个`单机多卡`的分布式环境，方便我们试验 `MirroredStrategy` 策略。
```
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    physical_devices[0],
    [
        tf.config.LogicalDeviceConfiguration(memory_limit=2048),
        tf.config.LogicalDeviceConfiguration(memory_limit=2048),
        tf.config.LogicalDeviceConfiguration(memory_limit=2048),
        tf.config.LogicalDeviceConfiguration(memory_limit=2048),
    ]
)

logical_devices = tf.config.list_logical_devices('GPU')
print(f'从一张物理 GPU 中虚拟出 {len(logical_devices)} 个逻辑 GPU')
```
结果打印：

    从一张物理 GPU 中虚拟出 4 个逻辑 GPU

# 数据准备

这里主要是准备用于训练神经网络的文本数据集，并对数据进行一些预处理，具体如下：

1. 设置每个训练批次的大小为 `128`，训练的总轮数为 `5` 。 从指定的 URL 下载 wiki 文本数据集，并将其解压缩到本地。
5. `train_ds`、`val_ds` 和 `test_ds`：这三个变量分别用于表示训练、验证和测试数据集。对每个数据集进行了类似的处理步骤：
   -  通过过滤器函数，剔除长度小于 100 个字符的文本行，以排除短文本。
   -  对数据进行随机`洗牌`，以打乱样本的顺序，有助于模型的训练。
   -  将数据批次大小设置为 `BATCH_SIZE` 。
   -  将数据`缓存`，以提高数据加载的效率。
   -  使用`预取策略`，允许在模型训练时异步加载下一个批次的数据，以减少训练时的等待取数据的时间。
 


```
BATCH_SIZE = 128
EPOCHS = 5
keras.utils.get_file(origin="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip", extract=True, )
wiki_dir = os.path.expanduser("~/.keras/datasets/wikitext-2/")
train_ds = (tf.data.TextLineDataset(wiki_dir + 'wiki.train.tokens').filter(lambda x: tf.strings.length(x) > 100).shuffle(buffer_size=500).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE))
val_ds = (tf.data.TextLineDataset(wiki_dir + 'wiki.valid.tokens').filter(lambda x: tf.strings.length(x) > 100).shuffle(buffer_size=500).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE))
test_ds = (tf.data.TextLineDataset(wiki_dir + 'wiki.test.tokens').filter(lambda x: tf.strings.length(x) > 100).shuffle(buffer_size=500).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE))
```


# 分布式训练

这里介绍训练一个 BERT 掩码语言模型（Masked Language Model，MLM），并使用分布式训练策略 `tf.distribute.MirroredStrategy` 进行训练。具体如下：

1.  `TRAINING_STEP` 定义了记录了每个 epoch 总的训练步骤。`lr_schedule`定义了一个学习率衰减策略，学习率在训练过程中从初始值（0.0001）线性地衰减到结束值（0.0），衰减的步数由 `TRAINING_STEP` 决定，也就是每个 epoch 过后学习率进行衰减下调。
1.  `callbacks`：定义一些回调函数，用于在训练过程中执行特定的操作。包括了早停（`EarlyStopping`）和记录训练日志（`TensorBoard`）。
1.  创建一个 MirroredStrategy ，用于多 GPU 分布式训练。 在 MirroredStrategy 的作用域内定义模型和训练过程。这意味着模型同时将在多个 GPU 上进行训练。在作用域中创建一个 `BERT` 掩码语言模型，其中包括预训练的 BERT 模型，并将其`最后一层池化层`设置为不可训练。
1.  编译模型，定义了损失函数为`稀疏分类交叉熵`、优化器为 `Adam` 和评估指标为`稀疏分类准确率`。


```
TRAINING_STEP = sum(1 for _ in train_ds.as_numpy_iterator()) * EPOCHS
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.0001, decay_steps=TRAINING_STEP, end_learning_rate=0.)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.TensorBoard('./logs')]

strategy = tf.distribute.MirroredStrategy()
print(f"可用的分布式训练的 GPU 设备有 {strategy.num_replicas_in_sync} 个")
with strategy.scope():
    model_d = keras_nlp.models.BertMaskedLM.from_preset("bert_tiny_en_uncased")
    model_d.get_layer("bert_backbone").get_layer("pooled_dense").trainable = False
    model_d.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer=tf.keras.optimizers.Adam(lr_schedule),
                    weighted_metrics=tf.keras.metrics.SparseCategoricalAccuracy())
    model_d.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)
model_d.evaluate(test_ds, verbose=1)
```
结果打印，理论上训练时间是与显卡数量成反比，卡越多训练时间越快，但是在小数据集中效果不是很明显，因为多块显卡之间的通信、复制、同步信息都会耗时，在实际训练中还要保证既要跑满每张显卡又不能 OOM ：

    可用的分布式训练的 GPU 设备有 4 个
    Epoch 1/5
    120/120 [==============================] - 48s 270ms/step - loss: 1.9297 - sparse_categorical_accuracy: 0.0579 - val_loss: 1.7024 - val_sparse_categorical_accuracy: 0.1913
    Epoch 2/5
    120/120 [==============================] - 29s 241ms/step - loss: 1.6517 - sparse_categorical_accuracy: 0.1644 - val_loss: 1.4522 - val_sparse_categorical_accuracy: 0.2798
    Epoch 3/5
    120/120 [==============================] - 29s 240ms/step - loss: 1.5088 - sparse_categorical_accuracy: 0.2163 - val_loss: 1.3278 - val_sparse_categorical_accuracy: 0.3198
    Epoch 4/5
    120/120 [==============================] - 29s 240ms/step - loss: 1.4406 - sparse_categorical_accuracy: 0.2370 - val_loss: 1.2749 - val_sparse_categorical_accuracy: 0.3361
    Epoch 5/5
    120/120 [==============================] - 29s 241ms/step - loss: 1.4113 - sparse_categorical_accuracy: 0.2448 - val_loss: 1.2603 - val_sparse_categorical_accuracy: 0.3402
    15/15 [==============================] - 4s 112ms/step - loss: 1.2633 - sparse_categorical_accuracy: 0.3531
 
可以看出损失在下降，准确率在提升，可以使用更大的 epoch 继续进行训练。使用 tensorboard 查看训练过程 loos 的变化过程如下：


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b2ed0c83f4a443a2bbae8ce6be5eab8a~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=421&h=307&s=18805&e=png&b=fdfdfd)

