# 前言

[上一篇文章](https://juejin.cn/post/7262003094161571899)中介绍了使用了剪枝、量化、轻量化来压缩模型大小的具体内容，但是对于一个模型的性能指标不仅仅有模型大小，还有准确率和速度，而且对于这三种技术的单用和混用也没有直观的概念，所以我索性花时间研究比对了一番，让大家也能有个直观的感受，如果不想看中间过程，可以直接跳到最后看结果表格。

创作不易，点个赞再走呗👍👍

# 基础模型

基础模型就是训练一个简单的卷积网络，完成对 `mnist` 数据集的分类任务，这个模型结构会沿用到后面所有的对比模型中，不会发生变化。关键代码如下：

```
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])
```

# 剪枝微调模型

训练好上面的`基础模型`，在此基础上使用剪枝技术进行`微调训练`。最后将微调之后的模型进行保存，供后面测试使用。关键代码如下：

```
validation_split = 0.1
num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=end_step)}
model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model_for_pruning.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
```

什么是`剪枝技术`呢？在剪枝微调训练过程中，会将对结果不怎么起作用的权重减小为 `0` ，实现模型稀疏化。稀疏的模型更易于压缩，而且我们在推断过程中因为有很多零而减少计算，另外模型精度非常接近原模型，所以理论上使用该技术可以实现将`模型的大小压缩`、`推理速度的提升`和`保证模型精度`。

# 剪枝优化+轻量化模型

训练好上面的 `剪枝微调模型`，我们要使用轻量化技术来对模型进行进一步的处理，将轻量化后的模型进行保存，供后面测试使用，关键代码如下：

```
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()
with open('剪枝优化+轻量化模型.tflite', 'wb') as f:
    f.write(pruned_tflite_model)
```

`TFLite` 是 tensorflow 为了将模型部署在`边缘设备`的工具包，可以把训练好的模型通过`转化`、`部署`和`优化`三个步骤，达到`提升运算速度，减少内存、显存占用`等效果。 TFlite 主要由 `Converter` 和`Interpreter` 组成。`Converter` 负责把 `TensorFlow` 训练好的模型转化，转化的同时还完成了对网络的优化，如量化。`Interpreter` 则负责把 tflite 格式的模型部署到边缘设备并高效地执行推理过程，简单来说，Converter 负责`打包优化模型`，Interpreter 负责`高效易用地执行推理`。

# 量化感知微调模型
训练好上面的 `基础模型` ，在此基础上进行量化感知微调训练，将训练好的模型保存起来，供后面测试使用，关键代码如下：

```
q_aware_model = tfmot.quantization.keras.quantize_model(model)
q_aware_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
q_aware_model.fit(train_images, train_labels,  batch_size=batch_size, epochs=epochs, validation_split=0.1)
```
什么是量化技术？tensorflow 中有两种量化技术：
- 训练后量化技术：训练后量化是直接将训练好的模型转变为 tflite 格式的模型。
- 训练中量化感知技术（我们这里使用到的）：又叫量化感知技术，这是一种`伪量化`。它在已经训练好的模型基础上再进行`量化微调训练`，完成某些`识别和统计`工作，以便于在后续转换为 `tflite` 格式减少精度损失，也就是说只有在转变为 tflite 后才真正是`量化后的模型`，此时只是在为真正的量化做准备，现在仍然是 `tensorflow` 模型而不是 `tflite` 模型。


# 量化感知优化+轻量化模型

在训练好上面的 `量化感知微调模型` 后，在此基础上使用轻量化技术来对模型进行进一步的处理，将轻量化后的模型进行保存，供后面测试使用，关键代码如下：

```
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
```

这里肯定会有人有疑问？`量化技术`和`轻量化`很像，而且轻量化过程中也有量化这个内容，他们之间有什么区别和联系。我的理解是，其实他们只是名字恰好相同，使用的`阶段不同`，量化技术主要是对模型进行一些个性化或者全局性的标记，这些准备就绪之后，使用轻量化技术将 `tensorflow` 模型转换为能在边缘设备上运行的 `tflite` 模型。

# 剪枝优化+量化优化+轻量化

在训练好的 `基础模型` 上面先后使用`剪枝`和`量化`，最后使用`轻量化`技术来对模型进行进一步的处理，将轻量化后的模型进行保存，供后面测试使用，这里就是将上面几部分代码混合，代码较多不再展示，详见文末的链接。

# 结果对比
### 使用 1 个 batch 推理测试

我们在 CPU 上以`基准模型`推理一个 batch （`32个样本`）的`准确率、耗时和模型文件大小`为准，用其他的模型的结果与其进行比对发现：
- 模型文件大小得到了`大幅度压缩`
- 在单独使用剪枝或者量化技术对`推理速度确实有提升`
- 准确率有损耗，但是可接受
- 综合来看混合使用所有技术后准确率、耗时和模型文件大小三个指标优化效果最好

另外有几点需要特殊说明：

- 这里的测试数据总归很少，耗时和准确率`不具有说服力`。
- 量化感知微调模型只是在基准模型之上微调的“伪量化”模型，里面有一些标记或者统计信息，所以模型大小不降反增。

表1
| 模型 | 准确率 |耗时/秒 | 耗时减少 |模型文件大小/字节 |模型文件大小减小|
| --- | --- |--- | --- | --- | --- |
| 基准模型 | 0.96875 |  0.072 |1倍|78117  |1倍 |
| 剪枝微调模型 | 0.96875 | 0.0379 |1.8倍| 25584 | 3.053倍|
| 量化感知微调模型 | 1.0 | 0.062025 | 1.16倍|79774 |0.97倍 |
| 剪枝优化+轻量化模型 |0.9375  | 0.00499 |14.42倍 |24851 | 3.143倍|
| 量化感知优化+轻量化模型 | 0.9375  | 0.01199 |6.005倍| 17690 |4.415倍 |
| 剪枝优化+量化感知优化+轻量化 | 0.96875 | 0.0030 |24倍| 8044 | 9.71倍|

 

### 使用 313 个 batch 未经优化的推理测试

我们在 CPU 上以基准模型推理所有数据，也就是 313 个 batch 的`准确率、耗时和模型文件大小`为准，这份测试`数据的数量较大`结果有说服力。用其他的模型的结果与其进行比对发现，总体上模型文件大小都有所减少，但是`最后两个模型`的`耗时却增加`了，我在经过代码调试之后发现主要有以下原因：

- 基准模型使用内置的 `predict() 函数`进行推理，这个函数本身就是设计用来进行`批量推理`的高性能函数，内部对于多批次数据的`预取`做了优化，并且在推理的时候是在 CPU `多核`上进行的，所以速度很快。
- 轻量化后的 tflite 模型主要靠 ```interpreter.invoke()``` 的调用来进行耗时的推理，我这里只是简单使用` for 循环`对每个 batch 进行推理，耗时累计，从上面的表1中我们知道，虽然在`单个 batch 上的推理速度 tflite 完胜`，但是对于多批次数据的推理，没有做任何的优化，就会出现耗时增加的现象。

表2
| 模型 | 准确率 |耗时/秒 | 耗时减少|模型文件大小/字节 |模型文件大小减小|
| --- | --- |--- | --- | --- | --- |
| 基准模型 | 0.9699 |  0.3200 |  1倍|78117  | 1倍 |
| 剪枝微调模型 | 0.9719 | 0.27707 |  1.15倍| 25584 | 3.05倍 |
| 量化感知微调模型 | 0.97315 | 0.3164 | 1.011倍 |79774 |  0.979倍 |
| 剪枝优化+轻量化模型 | 0.9707  | 0.28508 | 1.122倍|24851 |   3.143倍|
| 量化感知优化+轻量化模型 | 0.9777  | 3.52852 | 0.090倍|17690 | 4.415倍|
| 剪枝优化+量化感知优化+轻量化 |0.9723 |  0.4626 | 0.691倍|8044 |  9.71倍 |

### 多线程下使用 313 个 batch 的推理测试（只针对轻量化的 3 个模型）

重新训练了以下四个模型，在 CPU 上以基准模型推理所有数据，也就是 313 个 batch 的准确率、耗时和模型文件大小为准，我们可以看出来耗时和模型文件大小都明显减小了，准确率的损耗不大。

表3
| 模型 | 准确率 |耗时/秒 | 耗时减少|模型文件大小/字节 |模型文件大小减小|
| --- | --- |--- | --- | --- | --- |
| 基准模型 | 0.97345 |  0.3202 |  1倍|78093  | 1倍 |
| 剪枝优化+轻量化模型 | 0.9707  | 0.2515 | 1.273倍|24851 |   3.1424倍|
| 量化感知优化+轻量化模型 | 0.9777  | 0.2339 | 1.368倍|17690 | 4.414倍|
| 剪枝优化+量化感知优化+轻量化 |0.9684 |  0.2660 | 1.2037倍|8342 |  9.3614倍 |
 
 # 总结
 
 通过上面的三张表格我们可以总结以下两个要点：
 
-  使用剪枝、量化、轻量化等技术确实对推理速度和模型大小有明显改善，准确率虽有损耗但在可接受范围之内
-  要使用轻量化后的 tflite 模型进行大批量数据推理最好使用多线程来进行优化，推理速度会有明显的提升
 
 # 参考
 
-  https://zhuanlan.zhihu.com/p/66346329
-  https://blog.csdn.net/qq_30683995/article/details/100934739
- https://github.com/wangdayaya/DP_2023/tree/main/NLP%20%E6%96%87%E7%AB%A0/tensorflow%20%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E5%8F%8A%E5%AF%B9%E6%AF%94
- https://www.tensorflow.org/model_optimization/guide