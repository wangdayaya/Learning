# 常规方式训练基准模型
这段代码使用 TensorFlow 构建了一个简单的 CNN 模型，并在 MNIST 数据集上进行训练和评估，具体如下：

1.  加载 MNIST 数据集：使用 Keras 的 `mnist.load_data()` 函数加载 MNIST 数据集，将训练集和测试集分别赋值给 `train_images`、`train_labels`、`test_images` 和 `test_labels`。
1.  数据预处理：将图像数据归一化到 [0, 1] 的范围，通过除以 255.0，将像素值从整数转换为浮点数。这是一种常见的数据预处理步骤，用于提高模型的训练效果。
1.  构建 CNN 模型：使用 Keras 的 `Sequential` 模型构建一个简单的 CNN 模型。该模型包含一个输入层、一个卷积层、一个最大池化层、一个扁平化层和一个全连接层。卷积层用于提取图像特征，最大池化层用于降低特征图的空间尺寸，扁平化层将特征图展平成一维向量，全连接层用于输出类别得分。
1.  编译模型：通过调用 `model.compile()` 方法编译模型，指定优化器为 Adam，损失函数为稀疏交叉熵（Sparse Categorical Crossentropy），并使用准确率作为评估指标。
1.  训练模型：使用 `model.fit()` 方法对模型进行训练。将训练数据 `train_images` 和 `train_labels` 传入，并指定训练周期（epochs）为 4，以及验证集的拆分比例为 0.1。
1.  评估模型：使用 `model.evaluate()` 方法评估模型在测试集上的准确率，并将准确率打印出来作为基准测试准确率。
1.  保存模型：通过调用 `tf.keras.models.save_model()` 方法，将训练好的基准模型保存到一个临时文件中，并打印保存文件的路径。
 
```
    import tempfile
    import os
    import tensorflow as tf
    import numpy as np
    from tensorflow import keras

    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28,28)),
        keras.layers.Reshape(target_shape=(28,28,1)),
        keras.layers.Conv2D(filters=12, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=4, validation_split=0.1)
    _, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print('基准测试准确率：', baseline_model_accuracy)
    _, keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model, keras_file, include_optimizer=False)
    print('保存基准模型到', keras_file)
```

结果打印：

```
Epoch 1/4
1688/1688 [==============================] - 5s 2ms/step - loss: 0.3055 - accuracy: 0.9138 - val_loss: 0.1280 - val_accuracy: 0.9653
Epoch 2/4
1688/1688 [==============================] - 4s 2ms/step - loss: 0.1308 - accuracy: 0.9626 - val_loss: 0.0899 - val_accuracy: 0.9747
Epoch 3/4
1688/1688 [==============================] - 4s 2ms/step - loss: 0.0953 - accuracy: 0.9723 - val_loss: 0.0767 - val_accuracy: 0.9813
Epoch 4/4
1688/1688 [==============================] - 4s 2ms/step - loss: 0.0764 - accuracy: 0.9778 - val_loss: 0.0653 - val_accuracy: 0.9823
基准测试准确率： 0.977400004863739
保存基准模型到 C:\Users\13900K\AppData\Local\Temp\tmponrm9q25.h5
```


# 使用剪枝技术微调训练好的模型


这段代码使用 TensorFlow Model Optimization（tfmot）库中的剪枝（pruning）功能来对一个已有的 Keras 模型进行剪枝，并在剪枝后对修剪后的模型进行训练和评估。剪枝原理简单理解就是训练过程中，在迭代一定次数后，便将接近 0 的权重都置为 0 ，以达到对模型剪枝的作用，反复重复这个过程，直到模型权重达到目标的稀疏度。这样模型训练完成后，模型里面指定比例的权重皆为 0 ，当我们在后面进行压缩时，模型便可以得到很大程度的压缩空间，而且在推断过程中也减少了很多的计算量。具体如下：

1.  导入所需的库：导入 `prune_low_magnitude` 函数，用于对模型进行剪枝。
1.  定义剪枝相关参数：代码定义了用于剪枝的一些参数，包括 `batch_size`（批量大小）、`epochs`（训练周期）、`validation_split`（验证集拆分比例）等。`num_images` 计算了训练集中用于剪枝的图像数量。`end_step` 表示剪枝结束的步数，通过向上取整计算得到。
1.  定义剪枝策略：使用 `tfmot.sparsity.keras.PolynomialDecay` 定义了剪枝策略，这是一种多项式衰减的剪枝策略。初始稀疏度为 0.50，最终稀疏度为 0.8，表示将模型从稀疏性 0.5 部分开始，到达到稀疏性为 0.8 时候结束，稀疏就代表着多少权重变成 0 。剪枝开始的步数为 0，结束的步数为之前计算得到的 `end_step`。训练中每隔一定频率修建一次模型，一般默认是 100 个 step 。
1.  应用剪枝：通过调用 `prune_low_magnitude` 函数，将定义好的剪枝策略应用到原始的模型 `model` 上，得到一个修剪后的模型 `model_for_pruning`。
1.  编译修剪后的模型：对修剪后的模型 `model_for_pruning` 调用 `compile` 方法，指定优化器为 Adam，损失函数为稀疏交叉熵（Sparse Categorical Crossentropy），并使用准确率作为评估指标。
1.  定义回调函数：创建一个回调函数列表 `callbacks`，其中包含 `tfmot.sparsity.keras.UpdatePruningStep()` 回调函数，这个回调函数将在每个训练步骤中更新剪枝。
1.  训练修剪后的模型：使用 `model_for_pruning` 进行训练，通过 `fit` 方法传入训练数据和回调函数。训练周期为 `epochs`，验证集拆分比例为 `validation_split`。
1.  评估修剪后的模型：通过 `evaluate` 方法评估修剪后的模型在测试集上的准确率，并将准确率打印出来。同时，之前定义的基准测试准确率 `baseline_model_accuracy` 也会被打印出来，用于与修剪后的模型进行对比。
 
```
    import tensorflow_model_optimization as tfmot

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    batch_size = 128
    epochs = 2
    validation_split = 0.1
    num_images = train_images.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.8, begin_step=0, end_step=end_step)
    }
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
    ]
    model_for_pruning.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=callbacks)
    _, model_for_pruning_accuracy = model_for_pruning.evaluate(test_images, test_labels, verbose=0)
    print('基准测试准确率:', baseline_model_accuracy) 
    print('修剪测试准确率:', model_for_pruning_accuracy)
```    
    
结果打印，可以看到剪枝后的准确率只是下降了 0.005 左右：

```
Epoch 1/2
422/422 [==============================] - 2s 4ms/step - loss: 0.0898 - accuracy: 0.9744 - val_loss: 0.0781 - val_accuracy: 0.9802
Epoch 2/2
422/422 [==============================] - 2s 4ms/step - loss: 0.0838 - accuracy: 0.9754 - val_loss: 0.0799 - val_accuracy: 0.9793
基准测试准确率: 0.977400004863739
修剪测试准确率: 0.972599983215332
```

# 使用剪枝技术将模型缩小 3 倍

这段代码用于计算和比较 TensorFlow 模型在压缩后的文件大小。具体如下：

1.  `get_gzipped_model_size` 函数：这个函数用于计算文件的 gzip 压缩大小。它将传入的文件进行 gzip 压缩，然后返回压缩后的文件大小。
1.  去除模型中的剪枝：使用 `tfmot.sparsity.keras.strip_pruning` 函数从 `model_for_pruning` 中去除不能训练的权重，得到一个剪枝后的新模型 `model_for_export`。
1.  保存修剪后的 Keras 模型：使用 `tf.keras.models.save_model` 函数将修剪后的 Keras 模型 `model_for_export` 保存到一个临时文件 `pruned_keras_file` 中。
1.  转换并保存修剪后的 TFLite 模型：使用 `tf.lite.TFLiteConverter.from_keras_model` 将修剪后的 Keras 模型转换为 TFLite 格式，并将其保存到一个临时文件 `pruned_tflite_file` 中。
1.  获取压缩后的文件大小：通过调用 `get_gzipped_model_size` 函数，分别计算基线 Keras 模型、修剪后的 Keras 模型和修剪后的 TFLite 模型在 gzip 压缩后的文件大小，并将结果打印出来。

 ```
    import os
    import zipfile

    def get_gzipped_model_size(file):
        _, zipped_file = tempfile.mkstemp('.zip')
        with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(file)
        return os.path.getsize(zipped_file)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    _, pruned_keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('保存上一步中裁剪的Keras模型到:', pruned_keras_file)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()
    _, pruned_tflite_file = tempfile.mkstemp('.tflite')
    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)
    print('保存裁剪的 TFLite 模型到:', pruned_tflite_file)
    print("gzip 压缩后的基线 Keras 模型的大小: %.2f bytes" % (get_gzipped_model_size(keras_file)))
    print("gzip 压缩后的修剪后的 Keras 模型的大小: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
    print("gzip 压缩后的修剪后的 TFlite 模型的大小: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
 ```
结果打印，可以看到经过剪枝的模型大小缩小了 3 倍左右：

```
保存裁剪的 TFLite 模型到: C:\Users\13900K\AppData\Local\Temp\tmp2ul613rh.tflite
gzip 压缩后的基线 Keras 模型的大小: 78212.00 bytes
gzip 压缩后的修剪后的 Keras 模型的大小: 25787.00 bytes
gzip 压缩后的修剪后的 TFlite 模型的大小: 24859.00 bytes
```

# 使用剪枝和量化将模型缩小 10 倍
这段代码主要执行以下操作：

1.  创建 TFLite 转换器：使用 `tf.lite.TFLiteConverter.from_keras_model` 方法，将已经剪枝的 Keras 模型 `model_for_export` 创建为 TFLite 转换器对象 `converter`。
1.  设置量化优化：通过 `converter.optimizations = [tf.lite.Optimize.DEFAULT]` 设置 TFLite 转换器的优化选项为量化优化。量化（Quantization）是一种技术，用于减少模型的存储空间和推理时的计算量，从而提高模型在移动设备等资源受限环境下的性能。
1.  进行量化和剪枝：通过调用 `converter.convert()` 方法，将修剪后的 Keras 模型进行量化和剪枝转换，得到量化和剪枝后的 TFLite 模型 `quantized_and_pruned_tflite_model`。
1.  保存量化和剪枝后的 TFLite 模型：通过将 `quantized_and_pruned_tflite_model` 的内容写入临时文件 `quantized_and_pruned_tflite_file` 中，将量化和剪枝后的 TFLite 模型保存到该文件中。
1.  打印模型文件大小：通过调用 `get_gzipped_model_size` 函数，分别计算基线 Keras 模型（`keras_file`）和量化和剪枝后的 TFLite 模型（`quantized_and_pruned_tflite_file`）在 gzip 压缩后的文件大小，并将结果打印出来。
 ```
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()
    _, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')
    with open(quantized_and_pruned_tflite_file, 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)
    print('将量化和修剪后的 TFLite 模型保存到:', quantized_and_pruned_tflite_file)
    print("gzip 压缩后的基线 Keras 模型的大小: %.2f bytes" % (get_gzipped_model_size(keras_file)))
    print("gzipped 修剪和量化 TFlite 模型的大小: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))
```

结果打印，可以看到模型的大小缩小了 10 倍左右：

```
将量化和修剪后的 TFLite 模型保存到: C:\Users\13900K\AppData\Local\Temp\tmpuajkil8d.tflite
gzip 压缩后的基线 Keras 模型的大小: 78212.00 bytes
gzipped 修剪和量化 TFlite 模型的大小: 8244.00 bytes
```


这段代码用于评估修剪和量化后的 TFLite 模型在测试集上的准确率。具体如下：

1.  `evaluate_model` 函数：这个函数用于评估 TFLite 模型在测试集上的准确率。首先从 `interpreter` 中获取输入和输出的索引，然后遍历测试集中的图像数据 `test_images`，对每张图像进行预测。将图像转换为适合模型输入的格式，即在第0维增加一个维度，并将图像数据转换为浮点型。然后，设置 `interpreter` 的输入张量为当前图像，调用 `interpreter.invoke()` 进行推理，获取输出张量，并将输出的数字类别作为预测结果。将所有预测结果存储在 `prediction_digits` 列表中。最后，计算预测准确率并返回。
1.  创建 TFLite 解释器：通过 `tf.lite.Interpreter` 构造函数，使用 `quantized_and_pruned_tflite_model` 初始化一个 TFLite 解释器 `interpreter`。
1.  分配张量：调用 `interpreter.allocate_tensors()` 方法来为模型的输入和输出张量分配内存。
1.  评估修剪和量化后的 TFLite 模型：通过调用 `evaluate_model(interpreter)` 函数，将 TFLite 解释器 `interpreter` 作为参数传入，对修剪和量化后的 TFLite 模型在测试集上进行准确率评估，并将结果赋值给 `test_accuracy`。
1.  打印准确率：将修剪后的模型在测试集上的准确率 `model_for_pruning_accuracy` 和修剪和量化后的 TFLite 模型在测试集上的准确率 `test_accuracy` 打印出来。

```
    import numpy as np

    def evaluate_model(interpreter):
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        prediction_digits = []
        for i, test_image in enumerate(test_images):
            if i % 1000 == 0:
                print('Evaluated on {n} results so far.\n'.format(n=i))
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)
            interpreter.invoke()
            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_digits.append(digit)
        prediction_digits = np.array(prediction_digits)
        accuracy = (prediction_digits == test_labels).mean()
        return accuracy

    interpreter = tf.lite.Interpreter(model_content=quantized_and_pruned_tflite_model)
    interpreter.allocate_tensors()
    test_accuracy = evaluate_model(interpreter)
    print('修剪后的模型测试准确率:', model_for_pruning_accuracy)
    print('修剪和量化后的 TFLite 模型测试准确率:', test_accuracy)
```
结果打印，可以看到经过剪枝和量化后的准确率和之前相差无几：

```
Evaluated on 0 results so far.
Evaluated on 1000 results so far.
Evaluated on 2000 results so far.
Evaluated on 3000 results so far.
Evaluated on 4000 results so far.
Evaluated on 5000 results so far.
Evaluated on 6000 results so far.
Evaluated on 7000 results so far.
Evaluated on 8000 results so far.
Evaluated on 9000 results so far.
修剪后的模型测试准确率: 0.972599983215332
修剪和量化后的 TFLite 模型测试准确率: 0.9727
```