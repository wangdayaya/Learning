# 前文

这里使用孪生结构的深度学习网络模型，实现了对卫星图像对进行相似度判断的任务，需要准备如下：
 

*   tensorflow-gpu==2.10.0
*   python==3.10
*   地图数据 <https://huggingface.co/datasets/huggan/maps>

# 数据处理

## 加载数据

首先需要说明的是我自己将数据进行了处理，放到了 maps 目录之下，总共有 6582 张图像，每张图像会通过随机旋转得到两个不同的图像（这里只是为了实现简单的任务，只是对原图像进行了旋转，如果是比较复杂的任务，需要将图像经过反转、放缩、仿射、裁切等操作），并且这三个图像的名字是三个相连的数字，也就是说文件夹中每 3 个相连的图片是有关系的一组图片。

这里主要是从目录中读取所有的图像，原图像的大小是 (600,600)，数据量太大了，耗费内存和后期的 GPU ，所以将每张图像的大小调整为 (128,128) ，并且将图像的值都进行归一化，便于后期的模型计算。

<!---->

    data = []
    directory = 'maps'
    N = len(os.listdir(directory))
    for i in tqdm(range(N)):
        filename = str(i)+'.jpg'
        filepath = os.path.join(directory, filename)
        image = cv2.imread(filepath)
        image = cv2.resize(image,(128,128))
        data.append(image/255.)

## 制作图片对

因为我们要计算图片之间的相似度，所以我们要制作图片对，每对图片有的是相似的，标签设置为 0，有的是完全不相关的，标签设置为 1 。这里的代码通过以下形式制作图片对：

*   在一组三个相连的相似图片中随机一对图片，标签设置为 0
*   选择两个完全一样的图片，标签设置为 0
*   选择不相关的一对图片，标签设置为 1

如果对图片进行了更加复杂的操作，整体的思路还是按照上面的逻辑来制作图片对，只要图像的内容有关系标签就要设置为 0 ，如果没有关系标签就设置为 1 。

<!---->

    def make_pairs(ds):
        pairs = []
        labels = []
        N = len(ds)
        numbers = list(range(N))
        d = collections.defaultdict(int)
        for i in range(0, N, 3):
            L = range(i, i+3)
            for j in range(i, i+3):
                d[j] = min(N-1, random.choice(L))
        for i in tqdm(range(N)):
            
            x1 = ds[i]
            random_i = d[i]
            x2 = ds[random_i]
            pairs += [[x1, x2]]
            labels += [0]
            
            available_numbers = numbers[:i-3] + numbers[i+3:]
            random_j = random.choice(available_numbers)
            x3 = ds[random_j]
            pairs += [[x1, x3]]
            labels += [1]
            
            pairs += [[x1, x1]]
            labels += [0]
        return np.array(pairs).astype("float32"), np.array(labels).astype("float32")
    pairs,labels = make_pairs(data[:2000])

## 展示图片函数

这个函数是个工具函数，主要是将每对图片展示出来，进行预览。

    def show(images):
        plt.figure(figsize=(5, 5))
        for i, image in enumerate(images):
            plt.subplot(1, 2, i + 1)
            plt.imshow(image)
            plt.axis('off')
        plt.show()
    show(pairs[0])

这里是相同的图片对。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/9027932fdfaa40d59e90f1c4afdbcb14~tplv-k3u1fbpfcp-watermark.image?)

这里是不相同的图片对。

    show(pairs[1])

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/40c2f93c0b364bd490db9bd440d88a16~tplv-k3u1fbpfcp-watermark.image?)

这里还是相同的图片对，只是经过了旋转。

    show(pairs[3])

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6211e2928e4240df86259c88e1ca0582~tplv-k3u1fbpfcp-watermark.image?)

选取其中 90% 的数据作为训练集，剩下的 10% 的数据作为验证集。

    split_num = int(len(pairs) * 0.9)   
    x_train_1 = pairs[:split_num, 0]  
    x_train_2 = pairs[:split_num, 1]
    labels_train = labels[:split_num]

    x_val_1 = pairs[split_num:, 0]   
    x_val_2 = pairs[split_num:, 1]
    labels_val = labels[split_num:]

# 模型搭建和编译

## 孪生结构模型

这段代码定义了一个具有孪生网络结构的 Siamese 网络，用于比较两个输入图像之间的相似性。通过共享权重的方式，将两个输入图像分别提取为特征向量，然后计算它们之间的欧氏距离。最后，使用 sigmoid 激活函数将距离映射到 `[0, 1]` 的范围，表示两个输入图像的相似度。下面是对代码的逻辑进行解释：

1.  定义了一个 `euclidean_distance` 函数，用于计算欧氏距离。该函数接受一个包含两个向量的列表作为输入，并计算两个向量之间的欧氏距离。
2.  创建一个输入层 `input`，其输入形状为 `(128, 128, 3)`，用于接收图像输入。 使用 `tf.keras.layers.BatchNormalization()`对输入数据进行批量归一化处理。 使用 `layers.Conv2D`进行卷积操作，使用 `tanh` 作为激活函数。 使用 `layers.AveragePooling2D()`执行平均池化操作，。
3.  继续多次类似的卷积和池化操作。使用 `layers.Flatten()` 将特征压平后对展平后的数据进行批量归一化处理。 使用 `layers.Dense()`添加一个全连接层，创建 `embedding_network` 模型，也就是将输入的图像进行了编码，得到了128 维的特征向量。
4.  定义两个输入层 `input_1` 和 `input_2`，它们都有形状 `(128, 128, 3)`，用于接收两个待比较的输入图像。将两个输入分别通过 `embedding_network` 模型，得到输出 `tower_1` 和 `tower_2`，也就是两个图像的特征图像。使用 `layers.Lambda` 将 `euclidean_distance` 函数应用到 `tower_1` 和 `tower_2` 上，计算它们之间的欧氏距离。对欧氏距离的结果进行批量归一化处理。使用 `layers.Dense()`添加一个全连接层，得到输出层 `output_layer`，也就是最后的预测概率。

这段代码定义了一个具有孪生网络结构的 Siamese 网络，用于比较两个输入图像之间的相似性。通过共享权重的方式，将两个输入图像分别提取为特征向量，然后计算它们之间的欧氏距离。最后，使用 sigmoid 激活函数将距离映射到  `[0, 1]` 的范围，表示两个输入图像的相似度。

```
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

input = layers.Input((128, 128, 3))
x = tf.keras.layers.BatchNormalization()(input)
x = layers.Conv2D(64, (3, 3), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = layers.Dense(128, activation="tanh")(x)
embedding_network = keras.Model(input, x)

input_1 = layers.Input((128, 128, 3))
input_2 = layers.Input((128, 128, 3))
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

```

## 损失函数

这里定义了一个对比损失函数（contrastive loss）用于训练孪生网络。对比损失函数常用于孪生网络中的图片相似性任务。具体逻辑如下：

1.  定义了内部函数 `contrastive_loss`，它接受两个参数 `y_true` 和 `y_pred`，分别表示真实标签和预测值。
2.  使用 `tf.math.square(y_pred)`计算预测值的平方，。
3.  使用 `tf.math.square(tf.math.maximum(margin - y_pred, 0))` 计算边界值的平方。这里使用了 `tf.math.maximum` 函数来限制边界值大于零。
4.  使用公式 `(1 - y_true) * square_pred + y_true * margin_square` 计算对比损失函数。这里使用了逐元素相乘和相加的计算方式。
5.  使用 `tf.math.reduce_mean` 函数对所有损失值求均值，得到最终的损失值。
6.  最后，返回 `contrastive_loss` 函数作为外部函数 `loss` 的结果。

<!---->

    def loss(margin=1):
        def contrastive_loss(y_true, y_pred):
            square_pred = tf.math.square(y_pred)
            margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
            return tf.math.reduce_mean( (1 - y_true) * square_pred + (y_true) * margin_square )
        return contrastive_loss

## 模型编译

这行代码编译了孪生网络模型 `siamese`，使用了之前定义的对比损失函数作为 loss ，设置优化器为 "RMSprop"，这是一种常用的优化算法，用于更新网络参数以最小化损失函数。 设置评估指标为准确率,在训练过程中，模型将根据准确率来评估模型的性能。

<!---->

    siamese.compile(loss=loss(margin=1), optimizer="RMSprop", metrics=["accuracy"])

# 训练

    history = siamese.fit(
        [x_train_1, x_train_2], labels_train,
        validation_data=([x_val_1, x_val_2], labels_val),
        batch_size=8,
        epochs=30,
    )

过程打印：

    Epoch 1/30
    675/675 [==============================] - 9s 12ms/step - loss: 0.2402 - accuracy: 0.5887 - val_loss: 0.1635 - val_accuracy: 0.8283
    ...
    Epoch 8/30
    675/675 [==============================] - 8s 12ms/step - loss: 0.0884 - accuracy: 0.8896 - val_loss: 0.0354 - val_accuracy: 0.9517
    ...
    Epoch 15/30
    675/675 [==============================] - 8s 12ms/step - loss: 0.0821 - accuracy: 0.8935 - val_loss: 0.0338 - val_accuracy: 0.9583 
    ... 
    Epoch 30/30
    675/675 [==============================] - 8s 12ms/step - loss: 0.0706 - accuracy: 0.9081 - val_loss: 0.0315 - val_accuracy: 0.9550

训练过程的准确率如图所示。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8567e6017e7440abbabfb677875f967a~tplv-k3u1fbpfcp-watermark.image?)

# 效果测试

使用训练好的模型将验证集的预测概率结果映射为 0 或者 1 。

<!---->

    predictions = siamese.predict([x_val_1, x_val_2])
    def threshold_array(array):
        array[array > 0.2] = 1
        array[array <= 0.2] = 0
        return array

    arr_thresholded = threshold_array(predictions)
    arr_thresholded

我们将验证集图像以及预测的结果进行展示。索引为 6 的验证集数据和预测标签如下，可以看出这两个图片与预测值是对的，很明显这两个是相关的图像，尽管左边的图像相对于右边的图像有些倾斜。

        print(arr_thresholded[6])
        show([x_val_1[6], x_val_2[6]])

<!---->

    [0.]

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/333818765aa44d9bb7a1de067da75722~tplv-k3u1fbpfcp-watermark.image?)

索引为 7 的验证集数据和预测标签如下，可以看出这两个图片与预测值是对的，很明显这两个是不相关的图像，左边是城市，右边是海岸。

    print(arr_thresholded[7])
    show([x_val_1[7], x_val_2[7]])

<!---->

    [1.]

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7d662657417342cdb300fce44b7aa0fd~tplv-k3u1fbpfcp-watermark.image?)

索引为 9 的验证集数据和预测标签如下，可以看出这两个图片与预测值是对的，很明显这两个是相关的图像，右边的图像是左边图像经过向左旋转得到的。

    print(arr_thresholded[9])
    show([x_val_1[9], x_val_2[9]])

<!---->

    [0.]

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2c9406968ee7443fba8d81159d957aaf~tplv-k3u1fbpfcp-watermark.image?)

# 不足

本文只是实现了一个简单的通过孪生结果网络模型对图片进行相似性判断的任务，我们还可以继续对数据集进行更加复杂的图像处理，比如遮盖、模糊、反转等操作，再进行相似性的判断，这样更能用于实际任务当中。另外模型的复杂度也有待提升等等，总之需要改进的地方还有很多

# 感谢

<https://huggingface.co/datasets/huggan/maps>
