# 前言

本文使用 cpu 版本的 tensorflow 2.4 ，选用 Keras Tuner 工具以 Fashion 数据集的分类任务为例，完成最优超参数的快速选择任务。

当我们搭建完成深度学习模型结构之后，我们在训练模型的过程中，有很大一部分工作主要是通过验证集评估指标，来不断调节模型的超参数，这是比较耗时耗力的，如果只是不计代价为找到模型最优的超参数组合，我们大可以使用暴力穷举，把所有超参数都搭配组合试用一遍，肯定能找到一组最优的超参数结果。但是现实情况是我们不仅要考虑时间成本，还要考虑计算成本等因素，而 Tuner 工具包可帮助我们省时省力做这件事情，为我们的 TensorFlow 程序选择最佳的超参数集，整个这一找最佳超参数集的过程称为超参数调节或超调。

我们要知道超参数有两种类型：

* 模型超参：也就是能够影响模型的架构参数，例如神经元个数等
* 算法超参：也就是能够影响模型学习算法参数，例如学习率和 epoch 等

# 本文大纲

1. 获取 MNIST 数据并进行处理
2. 搭建超模型结构
3. 实例化调节器并进行模型超调
4. 训练模型获得最佳 epoch
5. 使用最有超参数集进行模型训练和评估

# 实现过程

### 1. 获取 MNIST 数据并进行处理
（1）首先我们要保证 tensorflow 不低于 2.4.0 ，python 不低于 3.8 ，否则无法使用 keras-tuner ，然后使用 pip 安装 keras-tuner 使用即可。

（2）使用 tensorflow 的内置函数从网络获取 Fashion 数据集 。

（3）将整个数据集做归一化操作，加快模型训练的收敛。

	import tensorflow as tf
	from tensorflow import keras
	import keras_tuner as kt
	
	(train_img, train_label), (test_img, test_label) = keras.datasets.fashion_mnist.load_data()
	train_img = train_img.astype('float32') / 255.0
	test_img = test_img.astype('float32') / 255.0

### 2. 搭建超模型

（1）这里主要是定义超模型，在构建用于超调的模型时，除了定义模型结构之外，还要定义超参的可选范围，这种为超调搭建的模型称为超模型。

（2）第一层是将每张图片的输入从二维压缩成一维。

（3）第二层是输出一个维度为 units 的全连接层，units 是我们的神经元个数选择器，我们规定了从 16-256 中随机选择一个可用的整数来进行模型的训练，整数选择的步长为 32 ，并最终能确定一个使得模型能达到最好效果的神经元个数，并且使用了激活函数 relu 来进行非线性变换。

（4）第三层是一个输出 10 个维度向量的全连接层，也就是输出该图片属于这 10 个类别的概率分布。

（5）学习率也是一个需要不断调整的超参数，所以我们使用 learning_rate 当做我们优化器学习率的选择器，从  [1e-2, 1e-3, 1e-4] 中选择能使模型达到最好效果的那个。

（6）编译模型的时候我们选择了最常用的 Adam 优化器，其学习率就是用我们刚才定义好的 learning_rate ，一会在模型学习的过程中会不断随机选择一个学习率。

（7）损失函数选择常见的 SparseCategoricalCrossentropy 。

（8）评估指标选择最简单的准确率 accuracy 。

	def model_builder(hp):
	    model = keras.Sequential()
	    model.add(keras.layers.Flatten(input_shape=(28, 28)))
	    units = hp.Int('units', min_value=16, max_value=256, step=32)
	    model.add(keras.layers.Dense(units=units, activation='relu'))
	    model.add(keras.layers.Dense(10))
	    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
	    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
	                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	                metrics=['accuracy'])
	    return model
	    
### 3. 实例化调节器并进行模型超调

（1）Tuner 中常见的调节器包括：RandomSearch、Hyperband、BayesianOptimization 和 Sklearn。在本文中我们使用 Hyperband 调节器来完成超参数的选择。

（2）我们知道现实中将所有的超参数进行搭配可以形成很多组，这个组数越多，那么最优超参数组合出现的概率也越大，但是与此相悖的是组数越多，在有限资源的情况下，能对每一组超参数进行测试的资源就越少，找到最优组的概率会下降， Hyperband 由此而生，Hyperband 就是假设尽可能出现多的超参数组，并且每组所能得到的资源要尽可能多，从而确保尽可能找到有最优超参数的那个组。

（3）在实际超调过程中，Hyperband 会假设 n 组超参数组合，然后对这 n 组超参数均匀地分配预算并进行验证评估，根据验证结果淘汰一半表现差的超参数组，不断重复上述过程直到找到一个最优超参数组合。

（4）调用函数 Hyperband 将调节器进行实例化，我们需要传入超模型、训练目标和最大的训练 epoch 。

（5）为了训练过程中防止过拟合现象，我们还加入了 EarlyStopping ，当经过 3 次 epoch 都没有优化之后会停止模型训练。

（6）这里就是使用超调实例 tuner 在模型训练过程中，通过调用 model\_builder 函数不断地在 units 、learning\_rate 中使用合适的超参构建新模型，并使用训练集为新模型训练 10 个 epoch ，最后选用训练集 20% 比例的验证集，记录下每个配置下的模型在验证集上表现出的评估指标 val\_accuracy 。

（7）当超调结束之后，我们返回最好的超参选用结果 best\_hps 。

	tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10)
	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
	tuner.search(train_img, train_label, epochs=10, validation_split=0.2, callbacks=[stop_early])
	best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
	print(f"""超调结束， 第一层全连接层的神经元个数建议选为 {best_hps.get('units')} ，优化器学习率建议选为 {best_hps.get('learning_rate')}.""")
	
输出结果为：

	Trial 30 Complete [00h 00m 15s]
	val_accuracy: 0.8845000267028809
	Best val_accuracy So Far: 0.8864166736602783
	Total elapsed time: 00h 03m 04s
	INFO:tensorflow:Oracle triggered exit
	超调结束， 第一层全连接层的神经元个数建议选为176 ，优化器学习率建议选为0.001.

### 4. 训练模型获得最佳 epoch

（1）我们已经通过 tuner 获得了最优的超参数，接下来我们只需要用最优的超参数构建模型，然后使用训练数据对模型进行训练 30 个 epoch 即可，并且使用训练数据的 20% 作为验证集对模型进行效果评估。

（2）我们可以将经过验证集评估得到的每个 epoch 产生的 val_accuracy 都取出来，然后选择其中最大的那个 epoch ，说明当经过 14 次 epoch 就可以达到最佳的模型效果

	model = tuner.hypermodel.build(best_hps)
	history = model.fit(img_train, label_train, epochs=30, validation_split=0.2)
	val_acc_per_epoch = history.history['val_accuracy']
	best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
	print('产生最好的 val_accuracy 是在第 %d 个 epoch ' % (best_epoch,))

输出为：

	Epoch 1/30
	1500/1500 [==============================] - 2s 1ms/step - loss: 0.6338 - accuracy: 0.7770 - val_loss: 0.4494 - val_accuracy: 0.8401
	Epoch 2/30
	1500/1500 [==============================] - 1s 938us/step - loss: 0.3950 - accuracy: 0.8575 - val_loss: 0.3971 - val_accuracy: 0.8497
	...
	Epoch 14/30
	1500/1500 [==============================] - 2s 1ms/step - loss: 0.2027 - accuracy: 0.9229 - val_loss: 0.3150 - val_accuracy: 0.8943
	Epoch 15/30
	1500/1500 [==============================] - 1s 985us/step - loss: 0.1951 - accuracy: 0.9280 - val_loss: 0.3200 - val_accuracy: 0.8912
	...
	Epoch 29/30
	1500/1500 [==============================] - 1s 906us/step - loss: 0.1298 - accuracy: 0.9517 - val_loss: 0.3939 - val_accuracy: 0.8902
	Epoch 30/30
	1500/1500 [==============================] - 1s 951us/step - loss: 0.1194 - accuracy: 0.9561 - val_loss: 0.4027 - val_accuracy: 0.8904
	产生最好的 val_accuracy 是在第 14 个 epoch 
	
### 5. 使用最有超参数集进行模型训练和评估

(1) 经过上面的过程我们已经找到了最好的神经元个数、学习率、以及训练模型的 epoch ，接下来使用这些超参重新实例化新的模型并使用上面的 best_epoch 对其进行训练，仍然选择训练集的 20% 作为验证集对模型效果进行验证 。

	best_model = tuner.hypermodel.build(best_hps)
	best_model.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)
	
（2）我们使用测试集堆模型进行评估。

	eval_result = best_model.evaluate(test_img, test_label)
	print("测试集损失值 %f , 测试集准确率为 %f"% (eval_result[0], eval_result[1]))

输出结果为：

	测试集损失值 0.345943 , 测试集准确率为 0.889400