# 概要

获取相似的图像是当下搜索引擎的一个重要功能，在本次任务中，我使用 LSH 算法，在预训练图像分类模型 Bit 基础上，实现简单的图像搜索功能。实现过程比较简单，容易理解，是个值得上手练习的案例。

# 数据准备

*   我们这里用到的数据是 `food101` 数据集，下载可能会需要很长的时间，并且要保证网络稳定。我们从中选取了三种类型的数据进行本次的任务，包括 `32 (dumplings)`、`53 （hamburger）`，`55 （hot_dog）`三种类别的图像。
*   在进行数据处理过程中，主要是将图像调整为大小为 `(256, 256)` 的尺寸，并且将图像数据归一化到 `[0, 1]` 范围内。
*   处理完数据之后进行混洗，选出 80% 作为训练集，20% 作为验证集。

<!---->

    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np
    import time
    import tensorflow_datasets as tfds
    import tensorflow_hub as hub
    from tqdm import tqdm
    import random

    train_ds = tfds.load( "food101", split="train", as_supervised=True)   # 75750
    ds = []
    for (image, label) in train_ds:
        label = label.numpy()
        if label == 55 or label == 53 or label == 32:
            image = tf.image.resize(image, (256,256))
            image = image / 255.
            image = image.numpy()
            ds.append([image, label])
    random.shuffle(ds)
    N = int(len(ds)*0.8)
    train_images, train_labels= zip(*ds[:N])
    val_images, val_labels = zip(*ds[N:]) 

这里是一个工具类，主要是为了随机选取 16 张图像，展示成 4x4 的图像形式。

    def show(images):
        if type(images) == tf.Tensor:
            images = images.numpy()
        images = images[:16]
        plt.figure(figsize=(5, 5))
        for i in range(len(images)):
            image = images[i]
            plt.subplot(4, 4, i + 1)
            plt.imshow(image)
            plt.axis('off')
        plt.show()

    show(train_images)

效果如下：

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8a5b5dadd30b4b51a4b9f24d6577ef17~tplv-k3u1fbpfcp-watermark.image?)

# 模型准备

*   使用`hub.KerasLayer`函数从指定的 URL 加载预训练模型，该模型是 Bit 模型在 ImageNet 21K 数据集上进行预训练后的结果，专门用于食物分类任务，我这里主要是将其作为特征提取器。
*   使用`tf.keras.Sequential`创建一个序列模型，序列模型是一系列网络层的线性堆叠。首先，使用`tf.keras.layers.Input`定义一个输入层，指定输入图像的形状为 (256, 256, 3) 。接下来，将之前加载的 Bit 模型(`bit_model`)添加到序列模型中，作为特征提取器。最后，添加一个归一化层`tf.keras.layers.Normalization`，用于对特征向量进行归一化处理，使其具有均值为 0 ，方差为 1 的标准正态分布。

<!---->

    bit_model  = hub.KerasLayer("https://tfhub.dev/google/experts/bit/r50x1/in21k/food/1")
    embedding_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((256, 256, 3)),
            bit_model,
            tf.keras.layers.Normalization(mean=0, variance=1, name='normalization'),
        ],
        name="embedding_model",
    )

# 工具

## bool2int 函数

该函数的作用是将一个布尔数组转换为对应的整数值。函数的输入参数`x`是一个布尔数组，表示一个哈希码的二进制形式。通过迭代布尔数组中的每个元素，对为`True`的位置进行位运算，将其对应的位设置为 1 ，并将所有位的值相加，得到最终的整数值。

    def bool2int(x):
        y = 0
        for i, j in enumerate(x):
            if j:
                y += 1 << i
        return y

## hash\_func 函数

*   该函数将输入的嵌入向量进行哈希处理。函数的输入参数`embedding`是一个图像经过编码的向量大小是 `(1,2048)` ，参数`random_vectors`是随机向量，大小是 `(2048,8)`。
*   将两个向量进行相乘得到一个结果向量大小是 (1, 8)，将其转换为布尔形式。
*   最后，调用`bool2int`函数将布尔数组转换为对应的整数哈希码，并将结果以列表的形式返回。

这段代码主要用于将嵌入向量进行哈希处理，将连续的嵌入空间映射到离散的哈希空间中，以便进行快速的相似度搜索或索引。

    def hash_func(embedding, random_vectors):
        embedding = np.array(embedding)

        # Random projection.
        bools = np.dot(embedding, random_vectors) > 0
        return [bool2int(bool_vec) for bool_vec in bools]

## Table 类

Table 类主要是用于构建哈希表，并实现了添加数据和查询数据的功能。random\_vectors 是一个随机生成的大小为 (8,2048) 的向量，用于和图像的特征向量结合生成哈希码。对于每个图像在经过哈希之后，将其添加到 table 字典中，可能同一个哈希值对应多张图像

*   初始化`Table`对象时，需要指定哈希表的大小(`hash_size`)和特征向量的维度(`dim`)。`random_vectors` 是一个随机生成的大小为 (`dim`, `hash_size`) 的向量，用于和图像的特征向量结合生成哈希码。`table` 字典用于用于存储数据。

*   `add`方法用于向哈希表中添加图像。每个图像使用其 id 和其 label 来命名，如 `0_55` 。通过调用`hash_func`函数，将图像特征向量和随机向量结合映射为哈希码。在 table 中如果哈希码对应的桶已存在，则将当前条目添加到桶中；否则，创建新的桶，并将条目添加到桶中。

*   `query`方法用于根据图像特征向量进行查询操作，返回与特征向量相似的条目列表。输入参数为图像经过模型提取的特征向量`vectors`。通过调用`hash_func`函数计算出哈希码。遍历哈希码列表，对每个哈希码，在哈希表中查找对应的桶。如果桶存在，则将桶中的条目添加到结果列表中，最后返回结果列表。

<!---->

    class Table:
        def __init__(self, hash_size, dim):
            self.table = {}
            self.hash_size = hash_size
            self.random_vectors = np.random.randn(hash_size, dim).T

        def add(self, id, vectors, label):
            entry = {"id_label": str(id) + "_" + str(label)}
            hashes = hash_func(vectors, self.random_vectors)
            for h in hashes:
                if h in self.table:
                    self.table[h].append(entry)
                else:
                    self.table[h] = [entry]

        def query(self, vectors):
            hashes = hash_func(vectors, self.random_vectors)
            results = []
            for h in hashes:
                if h in self.table:
                    results.extend(self.table[h])
            return results

## LSH

这段代码定义了一个`LSH` 类，它是基于多个哈希表构建的索引结构。

*   初始化`LSH`对象时，需要指定哈希表的大小(`hash_size`)、特征向量的维度(`dim`)和哈希表的数量(`num_tables`)。创建`num_tables`个`Table`对象，并将其存储在列表`tables`中。

*   `add`方法用于向所有哈希表中添加内容。输入参数包括`id`、`vectors`（图像特征向量）、`label`（标签）。遍历每个哈希表，调用对应的`Table`对象的`add`方法，将该对象经过计算添加到每一个的哈希表中，由于每个哈希表中的 random\_vectors 不同，所以计算出来的哈希值也不同。

*   `query`方法用于根据图像特征向量进行查询操作，返回与特征向量相似的条目列表。输入参数为图像特征向量`vectors`。遍历每个哈希表，调用每个哈希表的`query`方法，获取相似的图像列表，追加到最终结果中。通过多个的哈希表索引，可以提高相似度搜索的效率。

<!---->

    class LSH:
        def __init__(self, hash_size, dim, num_tables):
            self.num_tables = num_tables
            self.tables = []
            for i in range(self.num_tables):
                self.tables.append(Table(hash_size, dim))

        def add(self, id, vectors, label):
            for table in self.tables:
                table.add(id, vectors, label)

        def query(self, vectors):
            results = []
            for table in self.tables:
                results.extend(table.query(vectors))
            return results

## BuildLSHTable

这段代码定义了一个`BuildLSHTable`类，用于构建和查询 LSH 哈希表。

*   初始化`BuildLSHTable`对象时，需要指定哈希表的大小(`hash_size`)、特征向量的维度(`dim`)和哈希表的数量(`num_tables`)。`prediction_model` 是一个用于提取特征向量的模型，也就是我们前面定义的embedding\_model 。 `concrete_function`参数用于指定是否使用具体函数来提取特征向量。

*   `train`方法用于填充 LSH 对象。将训练数据中的每张图像使用`prediction_model`提取图像的特征向量。然后调用`LSH`对象的`add`方法，将特征向量、标签和唯一标识符添加到 LSH 对象中的每一个哈希表中。

*   `query`方法用于在 LSH 对象，找到与输入图像相似的图像。使用`prediction_model`提取输入图像的特征向量。调用`LSH`对象的`query`方法，获取与特征向量相似的图像。统计相同哈希值对应的相似图片的个数，并对计数结果除 dim 作为该哈希值的相似度。

```

class BuildLSHTable:
    def __init__( self, prediction_model, concrete_function=False,  hash_size=8, dim=2048, num_tables=10, ):
        self.hash_size = hash_size
        self.dim = dim
        self.num_tables = num_tables
        self.lsh = LSH(self.hash_size, self.dim, self.num_tables)
        self.prediction_model = prediction_model
        self.concrete_function = concrete_function

    def train(self, training_files):
        for id, training_file in enumerate(training_files):
            image, label = training_file
            if len(image.shape) < 4:
                image = image[None, ...]
            if self.concrete_function:
                features = self.prediction_model(tf.constant(image))[ "normalization" ].numpy()
            else:
                features = self.prediction_model.predict(image)
            self.lsh.add(id, features, label)

    def query(self, image, verbose=True):
        if len(image.shape) < 4:
            image = image[None, ...]
        if self.concrete_function:
            features = self.prediction_model(tf.constant(image))[  "normalization" ].numpy()
        else:
            features = self.prediction_model.predict(image)
        results = self.lsh.query(features)
        if verbose:
            print("Matches:", len(results))
        counts = {}
        for r in results:
            if r["id_label"] in counts:
                counts[r["id_label"]] += 1
            else:
                counts[r["id_label"]] = 1
        for k in counts:
            counts[k] = float(counts[k]) / self.dim
        return counts
```

# 训练

这里主要是使用训练数据进行训练，也就是哈希表的填充，过程比较简单。

    training_files = zip(train_images, train_labels)
    lsh_builder = BuildLSHTable(embedding_model)
    lsh_builder.train(training_files)

打印：

    1/1 [==============================] - 1s 722ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    ...
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 17ms/step

# 效果展示

这里主要是随机对 5 张图片进行了相似图片的搜索，我们可以直接看结果图。第一列是输入的 5 张图片，后面几列是根据相似度从高到低展示出来的图片，效果还是可以的，对于汉堡、包子、热狗基本能搜索出来相似的图片。

```

images = train_images + val_images
labels = train_labels + val_labels

def plot_images(images, labels):
    plt.figure(figsize=(20, 10))
    columns = 5
    for (i, image) in enumerate(images):
        ax = plt.subplot(len(images) // columns + 1, columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + "Label: {}".format(labels[i]))
        else:
            ax.set_title("Similar Image # " + str(i) + "\nLabel: {}".format(labels[i]))
        plt.imshow(image.astype("float"))
        plt.axis("off")


def visualize_lsh(lsh_class):
    idx = np.random.choice(len(val_images))
    image = val_images[idx]
    label = val_labels[idx]
    results = lsh_class.query(image)

    candidates = []
    labels = []
    overlaps = []

    for idx, r in enumerate(sorted(results, key=results.get, reverse=True)):
        if idx == 4:
            break
        image_id, label = r.split("_")[0], r.split("_")[1]
        candidates.append(images[int(image_id)])
        labels.append(label)
        overlaps.append(results[r])

    candidates.insert(0, image)
    labels.insert(0, label)

    plot_images(candidates, labels)

for _ in range(5):
    visualize_lsh(lsh_builder)
```

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f6d87fb2fb3645d58a9873f79c6222c0~tplv-k3u1fbpfcp-watermark.image?)

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a986305ab42a431fa3ae4ad12ad457ca~tplv-k3u1fbpfcp-watermark.image?)

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ac074e4b1cdb43458e0ab1b4a43009b7~tplv-k3u1fbpfcp-watermark.image?)

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a0c8de6ceff1409da187f3d191758563~tplv-k3u1fbpfcp-watermark.image?)

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0e295d62201b4d449f0ea88e71bfa3e7~tplv-k3u1fbpfcp-watermark.image?)
