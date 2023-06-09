# 前文

最近遇到了一个需求，就是在一批街景图片数据中，输入一张图片进行比较精确的图片搜索。之前本来想使用孪生网络来解决这个问题，但是[孪生网络](https://juejin.cn/post/7241744188261711909)(上一篇文章写了这个，感兴趣的同学可以前往)需要同时输入一对图片，这是个缺陷（也可能是我的能力有限想偏了），好像不能解决我的问题。我的需求只能是输入一张图片，然后在图片库中进行搜索，所以经过试验，想到了使用 `tfhub` 中预训练的模型 `MobileNet` 对图片库中所有的街景图片先进行特征提取，然后将特征提取结果存入向量引擎 `Milvus` 中，最后输入待搜索图片的特征向量，找到最相似的 N 张图片。

# 准备

*   在网上下载了 100 张左右的大小几乎都为 1080 x 810 的街景图片，然后对每张图片进行了 3 次的随机适当旋转操作来增加数据的丰富性，最终获得了 432 张图片
*   安装 Docker（目前 Milvus 主要是通过 Docker 启动的，所以这个要提前安装好）并启动
*   安装 Milvus 并启动
*   安装 tensorflow-gpu=2.10
*   安装 tensorflow-hub=0.13.0
*   安装 python=3.8
*   MobileNet 预训练模型
*   4090 显卡

# 数据加载

这段代码主要是用于读取指定目录下的图像文件，并将它们的文件名和图像数据保存在一个列表中。下面是对代码的详细解释：

1.  因为后续要使用的库较多，所以这里一次性导入所需的库，如：`os`、`cv2`、`tqdm`、`PIL`、`numpy`、`tensorflow_hub`、`tensorflow`和`matplotlib.pyplot`等库。
2.  使用`os.listdir`函数获取指定目录下的所有文件名，并保存在`fileNames`列表中，这个列表中的文件名在后续中会用到。
3.  遍历`fileNames`列表中的每个文件名，通过`os.path.join`函数将目录路径和文件名拼接起来，形成完整的文件路径，接着使用`cv2.imread`函数读取图像文件，统一将每张图片通过使用`cv2.resize`函数将图像的尺寸调整为指定的大小（224x224），最后将每张图片的的文件名和图片数据组成的子列表添加到`ds`列表中。

<!---->

        import os,cv2
        from tqdm import tqdm
        from PIL import Image
        import numpy as np
        import tensorflow_hub as hub
        import tensorflow as tf
        from tensorflow.keras import layers
        from tensorflow import keras
        import matplotlib.pyplot as plt

        ds = []
        directory = 'street'
        fileNames = os.listdir(directory)
        for i,filename in tqdm(enumerate(fileNames)):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            image = cv2.resize(image,(224 ,224))
            ds.append([filename, image])
部分图片效果如下，有原始图像还有经过旋转处理后面的图像。

 
![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0d9191e2dcae490fbe7c803537d100f0~tplv-k3u1fbpfcp-watermark.image?)
# 模型搭建

这段代码定义了一个函数`ComputeModel()`，用于构建一个基于 MobileNet V2 的特征提取模型，并返回该模型。具体代码含义如下：

1.  使用`layers.Input()`函数创建一个输入层，指定输入的形状为`(224, 224, 3)`，即 224x224大小的 RGB 图像，因为想要使用预训练模型 MobileNet 要求输入是这个大小。
2.  使用`hub.KerasLayer()`函数创建一个基于 MobileNet V2 的特征提取层。输出形状为`1280` 大小的维度，也就是每张图像会被压缩成 `1280` 维向量。设置`trainable=False`表示不会训练该层的参数。
3.  使用`keras.models.Model()`函数创建一个模型，将输入层和特征提取层作为输入和输出构建起来。
4.  创建该模型的实例 `encoder` 。

<!---->

        def ComputeModel():
            input = layers.Input((224, 224, 3))
            output = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", output_shape=[1280],trainable=False)(input)
            compression_model = keras.models.Model(inputs=input, outputs=output)
            return compression_model
        encoder = ComputeModel() 

# 特征提取

这段代码对数据集`ds`中的每张图像进行特征提取，并将提取到的特征保存在`im_features`列表中。具体代码含义如下：

1.  遍历数据集`ds`的中的每个元素，`ds[i][1]`表示第`i`张图像的像素矩阵。`ds[i][1][np.newaxis, :]`是为了让每张图片的像素形状符合模型的输入，也就是`(1, 224, 224, 3)`的输入张量。
2.  将转换后的输入张量传入特征提取模型`encoder`得到图像的特征向量。`image_features.numpy()`是为了将特征向量转换为 NumPy 数组。然后使用`np.squeeze(image_features.numpy())`将特征向量中的维度为 1 的维度去除，得到形状为大小为 `1024`的特征向量。
3.  将特征向量添加到`im_features`列表中。

<!---->

        im_features = []
        for i in tqdm(range(len(ds))):
            image_features = encoder(ds[i][1][np.newaxis, :])
            im_features.append(np.squeeze(image_features.numpy()))

# Milvus 录入特征向量

这段代码演示了如何使用`pymilvus`库将图像特征向量存储到 `Milvus` 数据库中。具体解释含义如下：

1.  定义了一些变量：
    *   `field_id`：表示数据集中每张图像的唯一标识字段，也可以认为是主键。
    *   `field_name`：表示数据集中每张图像的名称字段。
    *   `field_embedding`：表示数据集中每张图像的特征向量字段。
    *   `collection_name`：表示在 Milvus 中创建的集合名称，也可以理解为表名。
    *   `vector_dim`：表示特征向量的维度，因为我们之前通过模型提取出来的向量都是 1024 维，所以这里每个向量也是 1024 维
2.  连接到Milvus数据库。 检查是否已经存在名为`collection_name`的集合，如果存在，则删除该集合，方便后面重新进行数据的插入。
3.  定义集合的字段模式，包括`field_id`、`field_name`和`field_embedding`，并创建集合的模式。
4.  根据模式创建一个名为`images`的表。
5.  准备插入数据的实体列表：

    *   第一个列表包含了图像的唯一标识数据。
    *   第二个列表包含了图像的名称数据。
    *   第三个列表包含了图像的特征向量数据。
6.  使用`images.insert`方法将实体数据插入到表中。

<!---->

    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections
    from pymilvus.orm import utility
    field_id = "field_id"
    field_name = "name"
    field_embedding = "embedding"
    collection_name = "images_vgg"
    vector_dim = im_features[0].shape[0]

    print("连接 milvus")
    connections.connect(host="127.0.0.1", port=19530)

    if utility.has_collection(collection_name):
        print("删除数据")
        utility.drop_collection(collection_name)
    print("插入数据")
    fields = [
        FieldSchema(name=field_id, dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=128 ),
        FieldSchema(name=field_embedding, dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
    ]
    schema = CollectionSchema(fields, "embeddings of images")
    images = Collection(collection_name, schema)
    entities = [
        [i for i in range(len(ds))],
        [d[0] for d in ds],
        [v for v in im_features]
    ]
    insert_result = images.insert(entities)
    print("插入完毕")

# Milvus 搜索

这段代码展示了如何使用 `Milvus` 数据库进行图像的精确搜索。 具体代码解释如下：

1.  加载名为 `collection_name` 的 Milvus 集合表。
2.  将待搜索的向量 `vectors_to_search` 构造为一个二维数组。
3.  设置搜索参数 `search_params`，使用 L2 距离度量（欧式距离），并设置 `nprobe` 参数为 10 。`nprobe` 控制着在搜索过程中要探测的索引节点数目，较高的 `nprobe` 值可以提高搜索的召回率，但也会增加搜索时间，可以根据具体情况进行权衡。
4.  调用 `images.search()` 方法执行搜索操作，将搜索结果存储在 `result` 中。
5.  打印每个结果的距离、文档 ID 和对应的文件名。

<!---->

        images = Collection(collection_name)
        try:
            images.load()
        except Exception as e:
            pass
        vectors_to_search = entities[-1][0][np.newaxis, :]
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        result = images.search(vectors_to_search, field_embedding, search_params, limit=5, output_fields=[field_name])
        distances = result[0].distances
        ids = result[0].ids
        for distance, idx in zip(distances, ids):
            print(f"Distance: {distance}, Document id: {idx}, name: {fileNames[idx]}")

结果打印如下，可以看到结果还是比较准确的，最接近的图片肯定是自身 0.jpg ，之后就是进行旋转之后的结果图 0\_rotated\_0.jpg 、0\_rotated\_2.jpg 、0\_rotated\_1.jpg ，都能找出来，这个效果基本能满足我的需求了。

    输入向量的原始文件名为 0.jpg
    Distance: 0.0, Document id: 0, name: 0.jpg
    Distance: 52.11998748779297, Document id: 1, name: 0_rotated_0.jpg
    Distance: 83.90057373046875, Document id: 3, name: 0_rotated_2.jpg
    Distance: 101.7999496459961, Document id: 2, name: 0_rotated_1.jpg
    Distance: 147.64035034179688, Document id: 17, name: 102_rotated_2.jpg

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b22fbc7d1c134ce8b1af6a3d6d92e94b~tplv-k3u1fbpfcp-watermark.image?)
使用最后一张图片进行搜索，代码如下：

    print('输入向量的原始文件名为', fileNames[-1])
    vectors_to_search = entities[-1][-1][np.newaxis, :]
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    result = images.search(vectors_to_search, field_embedding, search_params, limit=5, output_fields=[field_name])
    distances = result[0].distances
    ids = result[0].ids
    for distance, idx in zip(distances, ids):
        print(f"Distance: {distance}, Document id: {idx}, name: {fileNames[idx]}")

结果打印如下，可以看到最相似的肯定是图片自身 9\_rotated\_2.jpg ，其次就是相关的原图和经过旋转的图 9.jpg 、9\_rotated\_0.jpg、9\_rotated\_1.jpg ，效果符合预期。

    输入向量的原始文件名为 9_rotated_2.jpg
    Distance: 0.0, Document id: 431, name: 9_rotated_2.jpg
    Distance: 58.83830261230469, Document id: 388, name: 9.jpg
    Distance: 82.89167022705078, Document id: 429, name: 9_rotated_0.jpg
    Distance: 88.25995635986328, Document id: 430, name: 9_rotated_1.jpg
    Distance: 162.87728881835938, Document id: 63, name: 16_rotated_1.jpg

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/23c1b10a0d8044769205d13a93f80e72~tplv-k3u1fbpfcp-watermark.image?)

# 后记

本文对图片只是进行了旋转操作，其实还可以进行裁切、遮挡、反转、仿射、模糊等操作，增加数据的多样性和数量，本文的后续工作也会向这个方向努力，研究在复杂情况下的街景图片精确搜索。

# 感谢

*   [百度街景数据](https://image.baidu.com/search/index?ct=201326592\&z=\&tn=baiduimage\&word=%E8%A1%97%E6%99%AF\&pn=\&spn=\&ie=utf-8\&oe=utf-8\&cl=2\&lm=-1\&fr=\&se=\&sme=\&width=1080\&height=810\&cs=\&os=\&objurl=\&di=\&gsm=5a\&dyTabStr=MCwzLDYsMSw0LDUsMiw3LDgsOQ%3D%3D)
*   [tensorflow hub 中的 MobileNet 预训练模型](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4)
*   [Milvus](https://milvus.io/)
*   [Docker](https://www.docker.com/)
