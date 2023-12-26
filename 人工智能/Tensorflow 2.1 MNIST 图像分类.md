# 前言
之前工作中主要使用的是 Tensorflow 1.15 版本，但是渐渐跟不上工作中的项目需求了，而且因为 2.x 版本和 1.x 版本差异较大，所以要专门花时间学习一下 2.x 版本，本文作为学习 Tensorflow 2.x 版本的开篇，主要介绍了使用 cpu 版本的 Tensorflow 2.1 搭建深度学习模型，完成对于 MNIST 数据的图片分类的任务。

# 本文流程

整个流程如下所示：

* 1.加载数据，处理数据
* 2.使用 keras 搭建深度学习模型
* 3.定义损失函数
* 4.配置编译模型
* 5.使用训练数据训练模型
* 6.使用测试数据评估模型
* 7.展示不使用归一化的操作的训练和评估结果
 
# 主要思路和实现
 
### (1) 加载数据，处理数据
这里是要导入 tensorflow 的包，前提是你要提前安装 tensorflow ，我这里为了方便直接使用的是 cpu 版本的 tensorflow==2.1.0 ，如果是为了学习的话，cpu 版本的也够用了，毕竟数据量和模型都不大。

	import tensorflow as tf

这里是为了加载 mnist 数据集，mnist 数据集里面就是 0-9 这 10 个数字的图片集，我们要使用深度学习实现一个模型完成对 mnist 数据集进行分类的任务，这个项目相当于 java 中 hello world 。

	mnist = tf.keras.datasets.mnist

这里的 (x\_train, y\_train) 表示的是训练集的图片和标签，(x\_test, y\_test) 表示的是测试集的图片和标签。

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

每张图片是 28*28 个像素点（数字）组成的，而每个像素点（数字）都是 0-255 中的某个数字，我们对其都除 255 ，这样就是相当于对这些图片的像素点值做归一化，这样有利于模型加速收敛，在本项目中执行本操作比不执行本操作最后的准确率高很多，在文末会展示注释本行情况下，模型评估的指标结果，大家可以自行对比差异。

	x_train, x_test = x_train / 255.0, x_test / 255.0

### (2) 使用 keras 搭建深度学习模型

这里主要是要构建机器学习模型，模型分为以下几层：

1. 第一层要接收图片的输入，每张图片是 28\*28 个像素点组成的，所以 input_shape=(28, 28)
2. 第二层是一个输出 128 维度的全连接操作
3. 第三层是要对第二层的输出随机丢弃 20% 的 Dropout 操作，这样有利于模型的泛化
4. 第四层是一个输出 10 维度的全连接操作，也就是预测该图片分别属于这十种类型的概率

	
		model = tf.keras.models.Sequential([
		  tf.keras.layers.Flatten(input_shape=(28, 28)),
		  tf.keras.layers.Dense(128, activation='relu'),
		  tf.keras.layers.Dropout(0.2),
		  tf.keras.layers.Dense(10)
		])

### (3) 定义损失函数

这里主要是定义损失函数，这里的损失函数使用到了 SparseCategoricalCrossentropy ，主要是为了计算标签和预测结果之间的交叉熵损失。
	
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

### (4) 配置编译模型
这里主要是配置和编译模型，优化器使用了 adam ，要优化的评价指标选用了准确率 accuracy ，当然了还可以选择其他的优化器和评价指标。

	model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
	
### (5) 	使用训练数据训练模型

这里主要使用训练数据的图片和标签来训练模型，将整个训练样本集训练 5 次。

	model.fit(x_train, y_train, epochs=5) 
 
训练过程结果输出如下：
	
	Train on 60000 samples
	Epoch 1/5
	60000/60000 [==============================] - 3s 43us/sample - loss: 0.2949 - accuracy: 0.9144
	Epoch 2/5
	60000/60000 [==============================] - 2s 40us/sample - loss: 0.1434 - accuracy: 0.9574
	Epoch 3/5
	60000/60000 [==============================] - 2s 36us/sample - loss: 0.1060 - accuracy: 0.9676
	Epoch 4/5
	60000/60000 [==============================] - 2s 31us/sample - loss: 0.0891 - accuracy: 0.9721
	Epoch 5/5
	60000/60000 [==============================] - 2s 29us/sample - loss: 0.0740 - accuracy: 0.9771
	10000/10000 - 0s - loss: 0.0744 - accuracy: 0.9777
 
### (6)  使用测试数据评估模型
 
 
这里主要是使用测试数据中的图片和标签来评估模型，verbose 可以选为 0、1、2 ，区别主要是结果输出的形式不一样，嫌麻烦可以不设置

	model.evaluate(x_test,  y_test, verbose=2)

评估的损失值和准确率如下：

	[0.07444974237508141, 0.9777]
 
### (7)  展示不使用归一化的操作的训练和评估结果

在不使用归一化操作的情况下，训练过程输出如下：

	Train on 60000 samples
	Epoch 1/5
	60000/60000 [==============================] - 3s 42us/sample - loss: 2.4383 - accuracy: 0.7449
	Epoch 2/5
	60000/60000 [==============================] - 2s 40us/sample - loss: 0.5852 - accuracy: 0.8432
	Epoch 3/5
	60000/60000 [==============================] - 2s 36us/sample - loss: 0.4770 - accuracy: 0.8724
	Epoch 4/5
	60000/60000 [==============================] - 2s 34us/sample - loss: 0.4069 - accuracy: 0.8950
	Epoch 5/5
	60000/60000 [==============================] - 2s 32us/sample - loss: 0.3897 - accuracy: 0.8996
	10000/10000 - 0s - loss: 0.2898 - accuracy: 0.9285
 
评估结果输入如下：

	[0.2897613683119416, 0.9285]
	
所以我们通过和上面的进行对比发现，不进行归一化操作，在训练过程中收敛较慢，在相同 epoch 的训练之后，评估的准确率和损失值都不理想，损失值比第（6）步操作的损失值大，准确率比第（6）步操作低 5% 左右。