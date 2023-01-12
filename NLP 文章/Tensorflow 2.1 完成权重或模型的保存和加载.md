***本文正在参加[「金石计划 . 瓜分6万现金大奖」](https://juejin.cn/post/7162096952883019783 "https://juejin.cn/post/7162096952883019783")***


# 前言

本文主要使用 cpu 版本的 tensorflow-2.1 来完成深度学习权重参数/模型的保存和加载操作。

在我们进行项目期间，很多时候都要在模型训练期间、训练结束之后对模型或者模型权重进行保存，然后我们可以从之前停止的地方恢复原模型效果继续进行训练或者直接投入实际使用，另外为了节省存储空间我们还可以自定义保存内容和保存频率。

# 文本大纲

1. 读取数据
2. 搭建深度学习模型
3. 使用回调函数在每个 epoch 后自动保存模型权重
4. 使用回调函数每经过 5 个 epoch 对模型权重保存一次
5. 手动保存模型权重到指定目录
6. 手动保存整个模型结构和权重

# 实现方法

### 1. 读取数据

（1）本文重点介绍模型或者模型权重的保存和读取的相关操作，使用到的是 MNIST 数据集仅是为了演示效果，我们无需关心模型训练的质量好坏。

（2）这里是常规的读取数据操作，我们为了能较快介绍本文重点内容，只使用了 MNIST 前 1000 条数据，然后对数据进行归一化操作，加快模型训练收敛速度，并且将每张图片的数据从二维压缩成一维。

	import os
	import tensorflow as tf
	from tensorflow import keras
	
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
	train_labels = train_labels[:1000]
	test_labels = test_labels[:1000]
	train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
	test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

### 2. 搭建深度学习模型

（1）这里主要是搭建一个最简单的深度学习模型。

（2）第一层将图片的长度为 784 的一维向量转换成 256 维向量的全连接操作，并且用到了 relu 激活函数。

（3）第二层紧接着使用了防止过拟合的 Dropout 操作，神经元丢弃率为 50% 。

（4）第三层为输出层，也就是输出每张图片属于对应 10 种类别的分布概率。

（5）优化器我们选择了最常见的 Adam 。

（6）损失函数选择了 SparseCategoricalCrossentropy 。

（7）评估指标选用了 SparseCategoricalAccuracy 。

	def create_model():
	    model = tf.keras.Sequential([keras.layers.Dense(256, activation='relu', input_shape=(784,)),
	                                 keras.layers.Dropout(0.5),
	                                 keras.layers.Dense(10) ])
	    model.compile(optimizer='adam',
	                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
	    return model



### 3. 使用回调函数在每个 epoch 后自动保存模型权重

（1）这里介绍一种在模型训练期间保存权重参数的方法，我们定义一个回调函数 callback ，它可以在训练过程中将权重保存在自定义目录中 weights_path ，在训练过程中一共执行 5 次 epoch ，每次 epoch 结束之后就会保存一次模型的权重到指定的目录。

（2）可以看到最后使用测试集进行评估的 loss 为 0.4952 ，分类准确率为 0.8500 。

	weights_path = "training_weights/cp.ckpt"
	weights_dir = os.path.dirname(weights_path)
	callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path, save_weights_only=True,  verbose=1)
	model = create_model()
	model.fit(train_images, 
	          train_labels,  
	          epochs=5,
	          validation_data=(test_images, test_labels),
	          callbacks=[callback]) 
 输出结果为：
 
	 val_loss: 0.4952 - val_sparse_categorical_accuracy: 0.8500	         
（3）我们浏览目标文件夹里，只有三个文件，每个 epoch 后自动都会保存三个文件，在下一次 epoch 之后会自动更新这三个文件的内容。

	os.listdir(weights_dir)
	
结果为：

	['checkpoint', 'cp.ckpt.data-00000-of-00001', 'cp.ckpt.index']

（4） 我们通过 create_model 定义了一个新的模型实例，然后让其在没有训练的情况下使用测试数据进行评估，结果可想而知，准确率差的离谱。

	NewModel = create_model()
	loss, acc = NewModel.evaluate(test_images, test_labels, verbose=2)

结果为：

	loss: 2.3694 - sparse_categorical_accuracy: 0.1330
	
（5） tensorflow 中只要两个模型有相同的模型结构，就可以在它们之间共享权重，所以我们使用 NewModel 读取了之前训练好的模型权重，再使用测试集对其进行评估发现，损失值和准确率和旧模型的结果完全一样，说明权重被相同结构的新模型成功加载并使用。

	NewModel.load_weights(checkpoint_path)
	loss, acc = NewModel.evaluate(test_images, test_labels, verbose=2)

输出结果：

	loss: 0.4952 - sparse_categorical_accuracy: 0.8500


### 4. 使用回调函数每经过 5 个 epoch 对模型权重保存一次



（1）如果我们想保留多个中间 epoch 的模型训练的权重，或者我们想每隔几个 epoch 保存一次模型训练的权重，这时候我们可以通过设置保存频率 period 来完成，我这里让新建的模型训练 30 个 epoch ，在每经过 10 epoch 后保存一次模型训练好的权重。

（2）使用测试集对此次模型进行评估，损失值为  0.4047 ，准确率为  0.8680 。

	weights_path = "training_weights2/cp-{epoch:04d}.ckpt"
	weights_dir = os.path.dirname(weights_path)
	batch_size = 64
	cp_callback = tf.keras.callbacks.ModelCheckpoint( filepath=weights_path, 
	                                                  verbose=1, 
	                                                  save_weights_only=True,
	                                                  period=10)
	model = create_model()
	model.save_weights(weights_path.format(epoch=1))
	model.fit(train_images, 
	          train_labels,
	          epochs=30, 
	          batch_size=batch_size, 
	          callbacks=[cp_callback],
	          validation_data=(test_images, test_labels),
	          verbose=1)
   
结果输出为：

	val_loss: 0.4047 - val_sparse_categorical_accuracy: 0.8680   
	
（2）这里我们能看到指定目录中的文件组成，这里的 0001 是因为训练时指定了要保存的 epoch 的权重，其他都是每 10 个 epoch 保存的权重参数文件。目录中有一个 checkpoint ，它是一个检查点文本文件，文件保存了一个目录下所有的模型文件列表，首行记录的是最后（最近）一次保存的模型名称。

（3）每个 epoch 保存下来的文件都包含：

* 一个索引文件，指示哪些权重存储在哪个分片中
* 一个或多个包含模型权重的分片

	
浏览文件夹内容

    os.listdir(weights_dir)

结果如下：

	['checkpoint',
	 'cp-0001.ckpt.data-00000-of-00001',
	 'cp-0001.ckpt.index',
	 'cp-0010.ckpt.data-00000-of-00001',
	 'cp-0010.ckpt.index',
	 'cp-0020.ckpt.data-00000-of-00001',
	 'cp-0020.ckpt.index',
	 'cp-0030.ckpt.data-00000-of-00001',
	 'cp-0030.ckpt.index']
 （4）我们将最后一次保存的权重读取出来，然后创建一个新的模型去读取刚刚保存的最新的之前训练好的模型权重，然后通过测试集对新模型进行评估，发现损失值准确率和之前完全一样，说明权重被成功读取并使用。

	latest = tf.train.latest_checkpoint(weights_dir)
	newModel = create_model()
	newModel.load_weights(latest)
	loss, acc = newModel.evaluate(test_images, test_labels, verbose=2)

结果如下：

	loss: 0.4047 - sparse_categorical_accuracy: 0.8680

### 5. 手动保存模型权重到指定目录


（1）有时候我们还想手动将模型训练好的权重保存到指定的目录下，我们可以使用  save_weights 函数，通过我们新建了一个同样的新模型，然后使用 load_weights 函数去读取权重并使用测试集对其进行评估，发现损失值和准确率仍然和之前的两种结果完全一样。

	model.save_weights('./training_weights3/my_cp')
	newModel = create_model()
	newModel.load_weights('./training_weights3/my_cp')
	loss, acc = newModel.evaluate(test_images, test_labels, verbose=2)

结果如下：

	loss: 0.4047 - sparse_categorical_accuracy: 0.8680
	
### 6. 手动保存整个模型结构和权重

（1）有时候我们还需要保存整个模型的结构和权重，这时候我们直接使用 save 函数即可将这些内容保存到指定目录，使用该方法要保证目录是存在的否则会报错，所以这里我们要创建文件夹。我们能看到损失值为  0.4821，准确率为 0.8460 。
	
	model = create_model()
	model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels), verbose=1)
	!mkdir my_model
	modelPath = './my_model'
	model.save(modelPath)
	
输出结果：

	val_loss: 0.4821 - val_sparse_categorical_accuracy: 0.8460

（2）然后我们通过函数  load_model 即可生成出一个新的完全一样结构和权重的模型，我们使用测试集对其进行评估，发现准确率和损失值和之前完全一样，说明模型结构和权重被完全读取恢复。

	new_model = tf.keras.models.load_model(modelPath)
	loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)

输出结果：

	 loss: 0.4821 - sparse_categorical_accuracy: 0.8460	