# 前言
使用 tensorflow 2.10 实现 AutoEncoder 模型，还原生成 MNIST 图片。

# 数据
1.  从 TensorFlow 的 Keras 库中加载 MNIST 数据集。MNIST 数据集包含了手写数字的灰度图像。`load_data()`函数会返回两个元组，分别包含了训练数据和测试数据。因为本文只关心图像本身，不关心标签，所以我们用下划线`_`来表示标签。
1.  将加载的图像数据进行标准化，也就是将像素值缩放到 `0` 到 `1` 之间。MNIST 图像的像素值范围通常是 `0` 到 `255` 之间的整数，所以我们将每个像素值除以 `255.0` 来将其缩放到 0 到 1 之间的浮点数，有助于神经网络更好地学习图像特征，也有利于网络的快速收敛。
2.  MNIST 图像数据中每个图像的尺寸为 `28x28` 像素，但在训练神经网络时，我们需要将图像数据展平成一维向量，因此将训练和测试数据的形状从 `(B, 28, 28)` 变为 `(B, 784)`。
```
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
```

# 模型

我们定义了一个简单的 `AutoEncoder` 模型，其中包括一个`编码器`和一个`解码器`，并设置了模型的优化器和损失函数。具体如下：

1. 将模型 `AutoEncoder` 的输入大小设置为 `(784,)` ，对应于上面转化后的每个图像展平后的像素数。
1.  然后定义了编码器部分，它是一个包含一个全连接层。该层有 `16` 个神经元，使用 `ReLU` 作为激活函数。编码器主要是为了对图像进行压缩和编码。
1.  接着定义了解码器部分，它也是一个全连接层，它的输出维度与编码器的输入数据维度相同`（784维）`。解码器使用 `Sigmoid` 激活函数，以确保输出值在 [0, 1] 范围内，因为我们输入的图像数据值在这个范围中。解码器的作用就是将编码器的特征尽量还原回原图像。
1.  最后编译 `AutoEncoder` 模型，使用 `Adam` 作为优化器，使用`均方误差（Mean Squared Error，MSE）`作为损失函数。

```
input_dim = 784
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(16, activation='relu')(input_layer)
decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder)
autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')
```
# 训练


1.  对 `AutoEncoder` 模型进行 `10` 个次循环训练，每次循环只进行一个`epoch`训练，这里是为将每一个 `epoch` 的训练结果进行保存，便于后期观察训练效果。并使用批量大小为 `256` 的样本数据进行训练。训练数据和验证数据都是`x_train`，这是因为 `AutoEncoder` 的目标是学习如何将输入数据编码并尽量还原回去。
1.  在每个训练迭代之后，使用训练好的 `AutoEncoder` 模型对测试数据`x_test`进行预测，得到了`decoded_img`，即重构后的图像。
1.  在每次循环中取部分个测试图片原图和预测图，绘制成对比的效果图并保存到本地。因为有 `10` 次循环，所以最终会生成 `10` 张图片。

```
for i in range(10):
    autoencoder.fit(x_train, x_train, epochs=1, batch_size=256, validation_data=(x_test, x_test))
    decoded_img = autoencoder.predict(x_test)
    n = 10
    plt.figure(figsize=(18, 4))
    for j in range(n):
        ax = plt.subplot(2, n, j + 1)
        plt.imshow(x_test[j].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, j + 1 + n)
        plt.imshow(decoded_img[j].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(f'example{i}.png')
```
训练日志打印：
```
235/235 [==============================] - 1s 2ms/step - loss: 0.0812 - val_loss: 0.0501
313/313 [==============================] - 0s 594us/step
235/235 [==============================] - 0s 2ms/step - loss: 0.0435 - val_loss: 0.0381
313/313 [==============================] - 0s 521us/step
...
235/235 [==============================] - 1s 2ms/step - loss: 0.0241 - val_loss: 0.0235
313/313 [==============================] - 0s 547us/step
235/235 [==============================] - 0s 2ms/step - loss: 0.0236 - val_loss: 0.0230
313/313 [==============================] - 0s 563us/step
```


# 效果
这里主要是将上面生成的 `10` 张图片合成一个 gif ，以 `1` 帧/秒的速率播放，可以看出随着模型的训练，模型生成的图片的还原程度逐步提升，如果增加模型架构的复杂度，生成效果还可以进一步提升。
```
image_files = [f'example{i}.png' for i in range(10)]
images = []
fig = plt.figure()
def update(frame):
    img = plt.imread(image_files[frame])
    plt.imshow(img)
    plt.axis('off')
    images.append([plt.imshow(img, animated=True)])
ani = animation.FuncAnimation(fig, update, frames=len(image_files), repeat=False)
output_gif = "output.gif"
ani.save(output_gif, writer='pillow', fps=1)
```

效果展示：

![output.gif](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f35db20d3b644a1db7a27bdf797739f9~tplv-k3u1fbpfcp-watermark.image?)