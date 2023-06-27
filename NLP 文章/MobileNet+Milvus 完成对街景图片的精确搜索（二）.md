# 前文

书接上回，这次我对图像进行了复杂的变换操作，然后使用相关技术完成图片的精确搜素。

# 数据处理

首先我在百度中搜集了 107 张街景图片，然后使用下面的代码对每张图片进行了随机的处理，其中可选的有六种处理手段：

-   `rotate_gen`：随机旋转角度范围为 0-90 度
-   `width_gen`：随机水平平移比例为 0-0.3
-   `height_gen`：随机垂直平移范围为 0-0.3
-   `zoom_gen`：随机缩放范围为 0-0.5
-   `vertical_gen`：随机垂直翻转
-   `horizontal_gen`：随机水平翻转

并且为了方便后期的使用，在每张命名中会体现出来是从哪个操作处理得到的结果。如 0_rotate_gen.jpg 表示的是对原始 0.jpg 图片进行随机旋转得到的图片，其他的以此类推，最后的结果得到有 107 张原始图片，同时每张原始图片会生成对应的 6 张经过 6 中操作得到的图片结果。


```
    import tensorflow as tf
    import os
    from tqdm import tqdm
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import numpy as np

    rotate_gen = ImageDataGenerator(rotation_range=90, )     # 随机旋转角度范围
    width_gen = ImageDataGenerator(width_shift_range=0.3,)   # 随机水平平移范围
    height_gen = ImageDataGenerator(height_shift_range=0.3,) # 随机垂直平移范围
    zoom_gen = ImageDataGenerator(zoom_range=0.5,)           # 随机缩放范围
    vertical_gen = ImageDataGenerator(  vertical_flip=True, )       # 垂直反转
    horizontal_gen = ImageDataGenerator(  horizontal_flip = True)   # 水平反转

    source_dir = 'street2'
    names = ['rotate_gen', 'height_gen', 'width_gen', 'zoom_gen','vertical_gen','horizontal_gen']
    L = os.listdir(source_dir)
    for filename in tqdm(L):
        source_path = os.path.join(source_dir, filename)
        if filename.endswith('.jpg'):
            for i,datagen in enumerate([rotate_gen, width_gen, height_gen, zoom_gen, vertical_gen, horizontal_gen]):
                target_path = os.path.join(source_dir, filename.split(".")[0] + f"_{names[i]}.jpg")
                image = tf.keras.preprocessing.image.load_img(source_path)
                image_array = tf.keras.preprocessing.image.img_to_array(image)
                image_array = np.expand_dims(image_array, axis=0)
                augmented_images = datagen.flow(image_array, batch_size=1)
                augmented_image = next(augmented_images)[0].astype(np.uint8)
                augmented_image = tf.keras.preprocessing.image.array_to_img(augmented_image)
                augmented_image.save(target_path)
                print(f"已增强文件: {target_path}")
```

下面是对图片 2.jpg 以及处理后的结果进行展示。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b2a4076fe68f415881e262f3550c5f7b~tplv-k3u1fbpfcp-watermark.image?)


# 特征提取


对每个文件进行迭代处理，使用`enumerate`函数获取文件的索引`i`和文件名`filename`。构建文件的完整路径`filepath`。使用`cv2.imread`函数读取图像文件，并将图像大小调整为`(224, 224)`。将文件名和图像数据作为一个列表`[filename, image]`添加到列表`ds`中。最终，列表`ds`中的每个元素包含了文件名和对应的图像数据，用于进一步处理和分析。

    ds = []
    directory = 'street2'
    fileNames = os.listdir(directory)
    for i,filename in tqdm(enumerate(fileNames)):
        filepath = os.path.join(directory, filename)
        image = cv2.imread(filepath)
        image = cv2.resize(image,(224 ,224))
        ds.append([filename, image])

这段代码的目的是创建一个用于图像编码的模型。定义一个函数`ComputeModel()`，用于创建图像编码模型。该模型基于 MobileNet V2 架构，使用预训练模型的权重。输入是一个形状为`(224, 224, 3)`的图像，输出是一个长度为 1280 的特征向量。在这个模型中，预训练的权重是不可训练的。通过这部分代码，创建的图像编码器模型，这个模型可以在后续的任务中用于图像特征提取和表示学习。


    from tensorflow.keras import layers
    import tensorflow_hub as hub
    from tensorflow import keras

    def ComputeModel():
        input = layers.Input((224, 224, 3))
        output = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", output_shape=[1280],trainable=False)(input)
    #     output = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5", output_shape=[1280],trainable=False)(input)
        compression_model = keras.models.Model(inputs=input, outputs=output)
        return compression_model
    encoder = ComputeModel() 
    

这段代码的作用是使用之前定义的图像编码器模型`encoder`对图像进行编码，并将编码后的特征向量存储在列表`im_features`中。需要注意的是在每次迭代中，取出图像数据`ds[i][1]`，要使用`np.newaxis`在维度上扩展，使其成为形状为`(1, 224, 224, 3)`的张量。

    im_features = []
    for i in tqdm(range(len(ds))):
        image_features = encoder(ds[i][1][np.newaxis, :])
        im_features.append(np.squeeze(image_features.numpy()))

# Milvus 搜索

这段代码的目的是使用Pymilvus库与Milvus进行连接，并创建一个 Milvus 集合来存储图像的特征向量。首先定义了字段 field_id、 name、embedding 以及表名 images_vgg 和向量长度 vector_dim ，并使用`connections.connect`函数与 Milvus 建立连接。在这里，指定了 Milvus 服务器的主机为"127.0.0.1"，端口为 19530 。

    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections
    from pymilvus.orm import utility
    field_id = "field_id"
    field_name = "name"
    field_embedding = "embedding"
    collection_name = "images_vgg"
    vector_dim = im_features[0].shape[0]

    print("连接 milvus")
    connections.connect(host="127.0.0.1", port=19530)

通过调用`utility.has_collection`函数检查是否已经存在名为`collection_name`的集合。如果集合已经存在，则打印提示信息"删除数据"并调用`utility.drop_collection`函数删除该集合。接着定义了一个包含三个字段的集合。`field_id`是 INT64 类型的字段，当作主键保存序列 id ；`field_name`是 VARCHAR 类型的字段，最大长度为 128 ，保存图片名字；`field_embedding`是 FLOAT_VECTOR 类型的字段，维度长度为`vector_dim` ，用来保存图片的向量特征。然后使用`CollectionSchema`类创建一个集合模式对象`schema`，该对象包含了字段的定义和集合的描述。然后使用预先准备好的数据组织成 entities 列表，将 entities 中的数据插入集合中，在插入之后将集合读取到内存中，以供后续的向量搜索。

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
    print("进行搜索")
    images = Collection(collection_name)
    try:
        images.load()
    except Exception as e:
        pass

这里定义了一个名为`random_search`的函数。在函数内部，通过随机选择一个索引`random_idx`来获取一个随机图像文件名和特征向量。`limit`变量用于指定搜索相似结果的数量限制。然后定义了一个搜索参数字典`search_params`，其中`metric_type`表示使用的距离度量方式为 L2 范数，`params`中的`nprobe`参数指定了使用的探测器数量。然后使用这些配置在 milvus 中进行相似图片的搜索，最后将最相似的 6 张图片及其文件名、与原图的距离都打印出来。

    import random
    def random_search():
        random_idx = random.choice([i for i in range(len(fileNames))])
        limit = 6
        print('输入向量的原始文件名为', fileNames[random_idx])
        vectors_to_search = im_features[random_idx][np.newaxis, :]
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        result = images.search(vectors_to_search, field_embedding, search_params, limit=limit, output_fields=[field_name])
        distances = result[0].distances
        ids = result[0].ids
        result = [fileNames[random_idx]]
        for distance, idx in zip(distances, ids):
            result.append(fileNames[idx])
            print(f"Distance: {distance}, Document id: {idx}, name: {fileNames[idx]}")
        show(result)
    random_search()

下面展示的是多次执行的结果效果。从最后的多次测试效果来看，输入的图片是原始图片、缩放图片、旋转图片、水平平移、水平反转的情况下，可以召回大部分的相关图片，但是对于垂直平移、垂直反转的情况下，召回效果比较差。此情况的出现可能是由于图片特征提取的预训练模型没有经过微调导致的，也可能是在图片在经过多种处理操作时产生的部分“垂直纹路”或者“水平纹路”对结果产生了影响。还需要后续的进一步试验。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e74484baa9f348d3a3282f9375ab3839~tplv-k3u1fbpfcp-watermark.image?)

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/deb3fc5a1f8e4b639d7c0f6f356e9438~tplv-k3u1fbpfcp-watermark.image?)


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/03dc4ed901114702be3f45bc7c783b5c~tplv-k3u1fbpfcp-watermark.image?)


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3c4dcea0602c4bb7bbe986e4aaa356f8~tplv-k3u1fbpfcp-watermark.image?)



![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/17d2f2ee0ba74548b3ec439690ec8219~tplv-k3u1fbpfcp-watermark.image?)




![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/73ccc0027b1a4163a89ef188d9446fbb~tplv-k3u1fbpfcp-watermark.image?)




![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0876b86ccab14019bb9e85355a45a3ce~tplv-k3u1fbpfcp-watermark.image?)

# 结尾

从结果上来看还是没能达到我的预期，我认为主要的影响部分是对图像的特征提取，尤其是对经过各种操作之后的图像的特征提取。借此机会我也想请教各位大佬，如何进一步提升效果，如果要训练模型该如何进行实操，如果要微调预训练模型该如何设计目标和实现过程，在此感谢。



# 感谢

-   [百度街景数据](https://link.juejin.cn/?target=https%3A%2F%2Fimage.baidu.com%2Fsearch%2Findex%3Fct%3D201326592%26z%3D%26tn%3Dbaiduimage%26word%3D%25E8%25A1%2597%25E6%2599%25AF%26pn%3D%26spn%3D%26ie%3Dutf-8%26oe%3Dutf-8%26cl%3D2%26lm%3D-1%26fr%3D%26se%3D%26sme%3D%26width%3D1080%26height%3D810%26cs%3D%26os%3D%26objurl%3D%26di%3D%26gsm%3D5a%26dyTabStr%3DMCwzLDYsMSw0LDUsMiw3LDgsOQ%253D%253D "https://image.baidu.com/search/index?ct=201326592&z=&tn=baiduimage&word=%E8%A1%97%E6%99%AF&pn=&spn=&ie=utf-8&oe=utf-8&cl=2&lm=-1&fr=&se=&sme=&width=1080&height=810&cs=&os=&objurl=&di=&gsm=5a&dyTabStr=MCwzLDYsMSw0LDUsMiw3LDgsOQ%3D%3D")
-   [tensorflow hub 中的 MobileNet 预训练模型](https://link.juejin.cn/?target=https%3A%2F%2Ftfhub.dev%2Fgoogle%2Ftf2-preview%2Fmobilenet_v2%2Ffeature_vector%2F4 "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")
-   [Milvus](https://link.juejin.cn/?target=https%3A%2F%2Fmilvus.io%2F "https://milvus.io/")
-   [Docker](https://link.juejin.cn/?target=https%3A%2F%2Fwww.docker.com%2F "https://www.docker.com/")