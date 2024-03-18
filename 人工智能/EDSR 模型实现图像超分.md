# 前言

主要介绍使用 `Enhanced Deep Residual Networks for Single Image Super-Resolution（EDSR）`模型完成图像超分修复。

# 模型介绍
`Enhanced Deep Residual Networks for Single Image Super-Resolution（EDSR）`是一种用于单图像超分辨率`（Single Image Super-Resolution，SISR）`的深度学习模型，其目标是通过从低分辨率图像生成高分辨率图像来改善图像质量。有如下特点：


1.  **深度残差网络（Deep Residual Network）** ：EDSR 模型基于`深度残差网络`的架构，来学习图像的残差映射，从而使得模型能够更轻松地学习到复杂的映射关系，减轻了`梯度消失和梯度爆炸`等训练问题。
1.  **增强的深度残差网络（Enhanced Deep Residual Network）** ：EDSR 通过对传统深度残差网络进行改进，包括`更深的网络结构、更多的残差块、更大的过滤器大小`等，以增加模型的表示能力和学习能力。
1.  **超分辨率任务（Super-Resolution Task）** ：EDSR 主要用于单图像超分辨率任务，即从低分辨率输入图像生成高分辨率输出图像。在训练阶段，模型通过学习输入图像与对应的高分辨率目标图像之间的映射关系来进行训练。在推理阶段，模型可以根据输入的低分辨率图像生成对应的高分辨率图像。

# 数据准备

我们使用 `DIV2K` 数据集，这是一个著名的单图像超分辨率数据集，包含 `1000` 张具有各种场景的图像，其中 `800` 张图像用于训练，`100` 张图像用于验证，`100` 张图像用于测试。 其中低分辨率有 `x2` 倍、`x3` 倍、`x4` 倍三种缩小因子的低分辨率图像，我们这里选用 x4 的图像作为低分辨率图像。

下面是三个辅助函数：
- `flip_left_right(lowres_img, highres_img)`：这个函数用于实现图像的左右翻转操作。 首先通过 tf.random.uniform() 生成一个随机数 rn，范围在 0 到 1 之间。接着通过 tf.cond() 条件函数，当 rn 小于 0.5 时，返回原始的 lowres_img 和 highres_img ；当 rn 大于等于 0.5 时，对 lowres_img 和 highres_img  分别进行左右翻转操作。最终返回处理后的图像数据。
- `random_rotate(lowres_img, highres_img)`：这个函数用于实现随机旋转图像的操作。首先通过 tf.random.uniform() 生成一个随机整数 rn，范围在 0 到 3 之间（旋转次数）。然后分别对 lowres_img 和 highres_img 使用 tf.image.rot90  函数，将图像顺时针旋转90度的次数等于 rn 。最终返回处理后的图像数据。
- `random_crop(lowres_img, highres_img, hr_crop_size=96, scale=4)`：这个函数用于实现随机裁剪图像的操作，同时保持低分辨率图像和高分辨率图像的对应关系。首先根据给定的高分辨率裁剪尺寸 hr_crop_size 和放大倍数 scale 计算出低分辨率图像的裁剪尺寸 lowres_crop_size 。然后通过 tf.random.uniform() 生成两个随机整数 lowres_width 和 lowres_height，分别表示低分辨率图像裁剪区域的左上角坐标。接着根据裁剪区域坐标，从原始的 lowres_img 和 highres_img 中分别裁剪出对应的低分辨率图像和高分辨率图像。最终返回处理后的图像数据。


这个函数的作用是用上面三个辅助函数创建一个用于训练或测试的数据集对象。它从输入的 dataset_cache 数据集开始，如果是训练阶段则进行 `random_crop 、random_rotate、flip_left_right、repeate` 等数据处理和增强操作，如果是测试阶段则只进行 `random_crop` 的数据处理操作。

```
def dataset_object(dataset_cache, training=True):
    ds = dataset_cache.map(lambda lowres, highres: random_crop(lowres, highres, scale=4), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(16)
    if training:
        ds = ds.repeate()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
```




# 模型搭建

这段代码定义了一个基于残差块的超分辨率模型EDSR（Enhanced Deep Super-Resolution），其中包含了训练步骤、预测步骤以及用于构建模型的辅助函数。

1. `EDSRModel` 类:
   - `train_step` 方法: 该方法重写了 `tf.keras.Model` 的 `train_step` 方法，用于定义训练步骤。在每个训练步骤中，输入数据 `data` 就是 `x` 和 `y`，然后使用梯度带（GradientTape）计算模型预测 `y_pred`，并计算损失。最后，更新模型的权重和度量指标，并返回度量指标的结果。
   - `predict_step` 方法: 该方法重写了 `tf.keras.Model` 的 `predict_step` 方法，用于定义预测步骤。在每个预测步骤中，输入 `x` 被转换为 `float32` 类型，并经过模型进行预测得到超分辨率图像。然后，对预测结果进行裁剪、取整、转换为 `uint8` 类型，并返回超分辨率图像。

2. 辅助函数：
   - `ResBlock` 函数: 定义了一个残差块，包括两个卷积层和一个跳跃连接（通过 `Add` 层实现）。这个函数用于构建模型中的残差块。
   - `Upsampling` 函数: 定义了一个上采样函数，使用卷积层进行上采样，并通过 `tf.nn.depth_to_space` 函数实现深度到空间的转换。这个函数用于模型中的上采样操作。
   - `make_model` 函数: 该函数用于构建整个EDSR模型。首先定义了输入层，然后进行图像缩放处理。接着定义了一系列残差块，并通过跳跃连接将它们连接起来。最后进行上采样和输出处理，得到最终的超分辨率图像。函数返回一个EDSR模型对象。
 
```
class EDSRModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_varibales
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, x):
        x = tf.cast(tf.expand_dims(x, axis=0).tf.float32)
        super_resolution_img = self(x, training=False)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        super_resolution_img = tf.round(super_resolution_img)
        super_resolution_img = tf.squeeze(tf.cast(super_resolution_img, tf.uint8), axis=0)
        return super_resolution_img


def ResBlock(inputs):
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    return x


def Upsampling(inputs, factor=2, **kwargs):
    x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor)
    x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(x)
    x = tf.nn.depth_to_space(x, block_size=factor)(x)
    return x


def make_model(num_filters, num_of_residual_blocks):
    input_layer = layers.Input(shape=(None, None, 3))
    x = layers.Rescaling(scale=1.0 / 255)(input_layer)
    x = x_new = layers.Conv2D(num_filters, 3, padding="same")(x)
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock(x_new)
    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])
    x = Upsampling(x)
    x = layers.Conv2D(3, 3, padding="same")(x)
    output_layer = layers.Rescaling(scale=255)(x)
    return EDSRModel(input_layer, output_layer)
```

# 模型训练

1.  `early_stopping_callback`：创建一个 EarlyStopping 回调对象，如果指标在设定的 10 次 epoch 内没有改善，则停止训练。
1.  `model_checkpoint_callback`：创建一个 ModelCheckpoint 回调对象，它用于在训练过程中保存模型的权重，并且只保存在验证集上性能最好的模型。
```
early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath="EDSR/checkpoint.keras", save_weights_only=False, monitor="loss",  mode="min", save_best_only=True, )
model = make_model(num_filters=64, num_of_residual_blocks=16)
optim_edsr = keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[5000], values=[1e-4, 5e-5]))
model.compile(optimizer=optim_edsr, loss="mae", metrics=[PSNR])
model.fit(train_ds, epochs=50, steps_per_epoch=200, validation_data=val_ds, callbacks=[early_stopping_callback, model_checkpoint_callback])
```
日志打印：
```
Epoch 1/50
200/200 [==============================] - 7s 20ms/step - loss: 26.4669 - PSNR: 19.8638 - val_loss: 14.3105 - val_PSNR: 23.2535
Epoch 2/50
200/200 [==============================] - 3s 13ms/step - loss: 12.8289 - PSNR: 25.4992 - val_loss: 11.6453 - val_PSNR: 26.0368
...
Epoch 46/50
200/200 [==============================] - 2s 12ms/step - loss: 7.2581 - PSNR: 32.2279 - val_loss: 7.3768 - val_PSNR: 31.8386
Epoch 47/50
200/200 [==============================] - 2s 12ms/step - loss: 7.3122 - PSNR: 32.0149 - val_loss: 7.2080 - val_PSNR: 31.2420
Epoch 48/50
200/200 [==============================] - 2s 12ms/step - loss: 7.2122 - PSNR: 33.6319 - val_loss: 7.1458 - val_PSNR: 31.3648
```
# 效果展示

从下面的展示样例中可以看出来。模型确实拥有了将图像超分的能力。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/bfa758f4b7ae494687da0c919411e63d~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=859&h=416&s=614043&e=png&b=ebe6e5)


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b9176428473a4425b6e452aea02195a6~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=863&h=416&s=476597&e=png&b=f8e8e5)


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1d6b46d1ab2b4eadab68b3bf3602e7d3~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=865&h=418&s=624835&e=png&b=f8f0eb)

# 参考
https://github.com/wangdayaya/DP_2023/blob/main/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/Enhanced%20Deep%20Residual%20Networks%20for%20single-image%20super-resolution.py