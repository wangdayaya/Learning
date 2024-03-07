# 前言

使用 ESPCN (Efficient Sub-Pixel CNN) 模型对低分辨率的图像，进行超分辨率处理。

# 效果展示
lowres 表示低分辨率图像，highres 表示高分辨率图像，prediction 表示模型预测的高分辨率图像，可以看出模型在生成高分辨率图像过程中确实发挥了作用。



 
![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/28bded8db9d14490bd919f992fc8ee3b~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=640&h=480&s=339249&e=png&b=fcf6f6)


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7542b6428f39419ba26212bf1af86222~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=640&h=480&s=360302&e=png&b=fcf7f6)

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/073b75264f7c4e89b77c60db05ed7fa4~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=640&h=480&s=358472&e=png&b=fdf8f7)


    PSNR of low resolution image and high resolution image is 25.4162
    PSNR of predict and high resolution is 26.8309
    
![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ee3192776bed4bfabe34f40d246895c6~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=640&h=480&s=305040&e=png&b=fbf2f1)



![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/38d85ab88a424ce9a8e97e759f60f035~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=640&h=480&s=336488&e=png&b=fbf2f1)


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6bc2c55cd5b74cb7ab7662c6e1710f32~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=640&h=480&s=325112&e=png&b=fcf3f2)

    PSNR of low resolution image and high resolution image is 24.5984
    PSNR of predict and high resolution is 26.2234

# 模型原理

在 `SRCNN` 和 `DRCN` 中，低分辨率图像都是先通过上采样插值得到与高分辨率图像同样的大小再作为网络输入，这意味着卷积操作在较高的分辨率上进行，相比于在低分辨率的图像上计算卷积会`降低效率`。 [`ESPCN`](https://arxiv.org/pdf/1609.05158.pdf) 提出一种在低分辨率图像上直接计算卷积得到高分辨率图像的高效率方法。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ef6b088727f647bcac39782f01d4b4f3~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1022&h=262&s=212187&e=png&b=fdfcfc)

ESPCN 的核心概念是`亚像素卷积层(sub-pixel convolutional layer)`。如上图所示，网络的输入是原始低分辨率图像，通过若干卷积层以后，得到的特征图像大小与输入图像一样，但是特征通道为 `r^2` 。将每个像素的 `r^2` 个通道重新排列成一个 `r x r` 的区域，对应于高分辨率图像中的一个 `r x r` 大小的子块，从而大小为`r^2 x H x W `的特征图像被重新排列成 `1 x rH x rW`  大小的高分辨率图像。这个变换虽然被称作 sub-pixel convolution , 但实际上并没有卷积操作。总之亚像素卷积层包含两个过程，分别是普通的`卷积层`和后面的`排列像素`的步骤。 

通过使用 sub-pixel convolution , 图像从低分辨率到高分辨率放大的过程，插值函数被隐含地包含在前面的卷积层中，可以自动学习到。只在最后一层对图像大小做变换，前面的卷积运算由于在低分辨率图像上进行，因此效率会较高。

# 数据处理

- 我自己生成了一批数据，我这里是放在了 `D:\pythonProject\HKYModel\data2` 目录之下。
- 因为数据集中已经分好了`训练集`和`测试集`，所以直接使用函数进行本地数据的读取即可得到 `train_ds` 和 `valid_ds`
- 将 `train_ds` 和 `valid_ds` 中的图片都做归一化操作


```
root_dir = "D:\pythonProject\HKYModel\BSR\BSDS500\data"
crop_size = 300
upscale_factor = 3
input_size = crop_size // upscale_factor
batch_size = 8
train_ds = image_dataset_from_directory(root_dir, batch_size=batch_size, image_size=(crop_size, crop_size), validation_split=0.2,  subset="training", seed=1337,  label_mode=None)
valid_ds = image_dataset_from_directory(root_dir, batch_size=batch_size,  image_size=(crop_size, crop_size),  validation_split=0.2, subset="validation",  seed=1337, label_mode=None)
def scaling(input_image):
    input_image = input_image / 255.
    return input_image
train_ds = train_ds.map(scaling)
valid_ds = valid_ds.map(scaling)

```
- `process_input` 函数接受输入图像和输入大小作为参数，并且将输入图像转换为 `YUV` 颜色空间。YUV 颜色空间包含了亮度（Y）和色度（U、V）信息。`tf.image.rgb_to_yuv` 函数用于将 RGB 彩色图像转换为 YUV 颜色空间。接着确定最后一个维度的索引。这个索引被用来沿着颜色通道轴（通常是最后一个维度）拆分输入张量，得到 Y、U、V 三个通道的张量。我们从拆分后的张量中只提取亮度通道 Y，并使用 `tf.image.resize` 函数将其调整为指定的输入大小，调整大小的方法是 `"area"`。
- `process_target` 函数也是类似的，它也将输入图像转换为 YUV 颜色空间，并提取出亮度通道 Y。但不同的是，它并没有调整图像的大小，只是返回了亮度通道 Y。
```
dataset = os.path.join(root_dir, "images")
test_path = os.path.join(dataset, "test")
test_img_paths = sorted([os.path.join(test_path, fname) for fname in os.listdir(test_path) if fname.endswith(".jpg")])
def process_input(input, input_size):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")
def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y
train_ds = train_ds.map(lambda x: (process_input(x, input_size), process_target(x))).prefetch(buffer_size=32)
valid_ds = valid_ds.map(lambda x: (process_input(x, input_size), process_target(x))).prefetch(buffer_size=32)
```

# 模型

 

- `DepthToSpace` 类继承自 `layers.Layer`，表示一个深度转换空间的层，用于实现深度转换空间操作。 `get_config` 方法用于获取层的配置信息。这个方法被调用以保存层的配置，以便在需要序列化模型时可以重新创建相同的层实例。`call` 方法实现了层的前向传播逻辑，在这个方法中，它接受一个输入张量 `input`，然后执行深度转换空间操作。具体地，它首先获取输入张量的形状信息，然后按照 `block_size` 分块重排张量，并最终返回转换后的张量。
- `get_model` 函数用于创建一个 Keras 模型。在这个函数中，它接受两个参数：`upscale_factor` 和 `channels`。`upscale_factor` 表示上采样因子，`channels` 表示输入图像的通道数。在模型中，它使用了一系列的卷积层构建了一个深度卷积神经网络。然后，通过 `DepthToSpace` 层来实现深度转换空间操作，以实现图像的上采样。最后，通过 `keras.Model` 类构建了一个 Keras 模型，指定了输入和输出，返回了这个模型。

 


```
class DepthToSpace(layers.Layer):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config
    def call(self, input):
        batch, height, width, depth = ops.shape(input)
        depth = depth // (self.block_size**2)
        x = ops.reshape(input, [batch, height, width, self.block_size, self.block_size, depth])
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(x, [batch, height * self.block_size, width * self.block_size, depth])
        return x
def get_model(upscale_factor=3, channels=1):
    conv_args = {"activation": "relu",  "kernel_initializer": "orthogonal", "padding": "same"}
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(512, 5, **conv_args)(inputs)
    x = layers.Conv2D(256, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor**2), 3, **conv_args)(x)
    outputs = DepthToSpace(upscale_factor)(x)
    return keras.Model(inputs, outputs)
```

# 训练

- 自定义回调函数类 ESPCNCallback ，在每个 epoch 开始时调用 `on_epoch_begin` 方法，它初始化了一个列表 `self.psnr`，用于存储每个 epoch 的峰值信噪比（PSNR）。在每个 epoch 结束时调用 `on_epoch_end` 方法。它计算了当前 epoch 的平均 PSNR ，并打印输出。每隔 20 个 epoch 就利用模型生成了一个预测图像，并通过 `plot_results` 函数绘制了这个预测图像，用于观察模型的生成效果。在每个测试集的 batch 结束时调用`on_test_batch_end` 方法，它计算了当前 batch 的 PSNR ，并将其添加到 `self.psnr` 列表中。
- 另外创建了两个额外的 Keras 回调函数：`early_stopping_callback` 用于在训练过程中实施 early stopping 策略，如果在连续 5 个 epoch 中损失没有降低，则停止训练；`model_checkpoint_callback` 用于在训练过程中保存模型的最佳参数。
- 使用 Adam 优化器和均方误差作为损失函数。
- 使用 `model.fit` 函数进行模型的训练。指定了训练数据集 `train_ds`，并设置了训练的 epochs 数目为 200，并且设置了之前定义的回调函数作为回调参数。
```
class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        if epoch % 20 == 0:
            prediction = upscale_image(self.model, self.test_img)
            plot_results(prediction, "epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))


early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=5)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath="ESPCN/checkpoint.keras", save_weights_only=False, monitor="loss",  mode="min", save_best_only=True, )
model = get_model(upscale_factor=upscale_factor, channels=1)
model.summary()
callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.MeanSquaredError())
model.fit(train_ds, epochs=200, callbacks=callbacks, validation_data=valid_ds, verbose=2)
```

日志打印：
```
Epoch 1/200
2024-03-06 16:14:17.804215: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
Mean PSNR for epoch: 22.44
50/50 - 5s - loss: 0.0226 - val_loss: 0.0058 - 5s/epoch - 105ms/step
Epoch 2/200
Mean PSNR for epoch: 23.57
50/50 - 1s - loss: 0.0064 - val_loss: 0.0043 - 1s/epoch - 21ms/step
...
Epoch 29/200
Mean PSNR for epoch: 26.75
50/50 - 1s - loss: 0.0025 - val_loss: 0.0022 - 996ms/epoch - 20ms/step
Epoch 30/200
Mean PSNR for epoch: 26.53
50/50 - 1s - loss: 0.0025 - val_loss: 0.0023 - 992ms/epoch - 20ms/step
Epoch 31/200
Mean PSNR for epoch: 26.18
50/50 - 1s - loss: 0.0025 - val_loss: 0.0023 - 987ms/epoch - 20ms/step
```




# 参考

- https://github.com/wangdayaya/DP_2023/blob/main/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/Image%20Super-Resolution%20using%20an%20Efficient%20Sub-Pixel%20CNN.py
- https://blog.csdn.net/sinat_36197913/article/details/104823216
- https://blog.csdn.net/sinat_36197913/article/details/104823216