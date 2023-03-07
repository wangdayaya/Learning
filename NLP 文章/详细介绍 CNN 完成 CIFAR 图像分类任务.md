

## 准备工作

这里用到的是：

*  tensorflow-cpu 2.4 版本
*  python 3.8 版本 
*  anaconda 自带的 jupyter notebook


## 本文大纲

1. 加载、展示、处理 CIFAR 图像数据
2. 搭建 CNN 模型架构
3. 编译、训练模型
4. 测试模型

## 加载、展示、处理 CIFAR 图像数据

（1）这里国内下载数据可能会报错，所以需要提前在这里下载好数据 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz ，然后将压缩包名字改成  cifar-10-batches-py.tar.gz ，然后放到路径：你的主目录/.keras/datasets 下。

（2）CIFAR10 数据集包含 10 个种类，共 60000 张彩色图片，每类图片有 6000 张。此数据集中 50000 张图片被作为训练集，剩余 10000 张图片作为测试集。每个种类的图片集之间相互独立。
 
（3）我们首先通过 datasets 读取到 CIFAR10 数据集，结果分成了两部分，训练集和测试集，每份数据集中有图片及其对应的标签。我们首先对所有图片数据进行归一化处理，这么做的好处有两点，一是能够使数据有相同的分布，保证不同的图像经过 CNN 后提取的特征的分布基本上趋于一致，梯度下降就很快，进而加快模型收敛，另外可能在一定程度上提升模型的精度。

	import tensorflow as tf
	from tensorflow.keras import datasets, layers, models
	import matplotlib.pyplot as plt
	from tensorflow.keras import regularizers
	
	(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
	train_images, test_images = train_images / 255.0, test_images / 255.0

（4）本数据集中的 10 中分类分别是 ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']，在这里我们随便展现出来了 4 张图片，虽然有点模糊，但是大体是可以区分出来所属类别。

	class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	plt.figure(figsize=(4,4))
	for i in range(100,104):
	    plt.subplot(2,2,i+1-100)
	    plt.xticks([])
	    plt.yticks([])
	    plt.grid(False)
	    plt.imshow(train_images[i])
	    plt.xlabel(class_names[train_labels[i][0]])
	plt.show()
	

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3642263b8ca14644b65d17e6fb30f647~tplv-k3u1fbpfcp-watermark.image?)

## 搭建 CNN 模型架构

这里主要是介绍模型架构组成，主要三个特征提取块，每个块中有卷积层、批正则化层、最大池化层，另外最后加入了 Dropout 层、全连接层、分类输出层，每个特征提取块的详细情况都一样，以第一个特征提取块为例说明如下：

* 第一层主要是接收大小为 (32, 32, 3) 的图片，这里的前两个维度是每张图片的长款像素数量，第三个维度是每个像素点的三个颜色通道，这里主要使用了 64 个大小都为 (3, 3) 的卷积核来完成图像的特征提取工作，总共的可训练参数有个。

* 第二层主要是过了防止过拟合设置批正则化，它先对数据做平移和伸缩变换，将数据伸缩到固定区间范围，其次可以加快模型训练时的收敛速度，使得模型训练过程更加稳定，避免梯度爆炸或者梯度消失，另外还能起到一定的正则化作用，现在几乎不用 Dropout 。**其实这里我有个问题，看下面的模型架构图中每个批正则化都有参数，这个是怎么算出来的呢？还有为什么不可训练的参数有 512 个？有知道的请教教我**。

* 第三层主要是进行了二维的最大池化操作，池化窗口大小为 (2, 2)， 池化层可以有效的缩小特征矩阵的尺寸，从而减少最后连接层的中的参数数量。所以加入池化层也可以加快计算速度和防止过拟合的作用。

* 后面使用了 Flatten 将每张图片的多维特征，压缩成一个长维度的特征，我们在进行全连接层之前还加入了 Dropout 就是为了防止过拟合，最后加入分类输出层进行类型 logit 的预测。

		model = models.Sequential([
		    layers.Conv2D(64, (3, 3),  activation='relu', input_shape=(32, 32, 3)),
		    layers.BatchNormalization(),
		    layers.MaxPooling2D((2, 2)),
		    
		    layers.Conv2D(128, (3, 3), activation='relu'),
		    layers.BatchNormalization(),
		    layers.MaxPooling2D((2, 2)),
		    
		    layers.Conv2D(64, (3, 3), activation='relu'),
		    layers.BatchNormalization(),
		    layers.MaxPooling2D((2, 2)),
		    
		    layers.Flatten(),
		    layers.Dropout(0.5),
		    layers.Dense(32, activation='relu'),
		    layers.Dense(10)
		])
		model.summary()

结果打印：

	Model: "sequential_6"
	_________________________________________________________________
	 Layer (type)                Output Shape              Param #   
	=================================================================
	 conv2d_21 (Conv2D)          (None, 30, 30, 64)        1792      
	                                                                 
	 batch_normalization_3 (Batc  (None, 30, 30, 64)       256       
	 hNormalization)                                                 
	                                                                 
	 max_pooling2d_17 (MaxPoolin  (None, 15, 15, 64)       0         
	 g2D)                                                            
	                                                                 
	 conv2d_22 (Conv2D)          (None, 13, 13, 128)       73856     
	                                                                 
	 batch_normalization_4 (Batc  (None, 13, 13, 128)      512       
	 hNormalization)                                                 
	                                                                 
	 max_pooling2d_18 (MaxPoolin  (None, 6, 6, 128)        0         
	 g2D)                                                            
	                                                                 
	 conv2d_23 (Conv2D)          (None, 4, 4, 64)          73792     
	                                                                 
	 batch_normalization_5 (Batc  (None, 4, 4, 64)         256       
	 hNormalization)                                                 
	                                                                 
	 max_pooling2d_19 (MaxPoolin  (None, 2, 2, 64)         0         
	 g2D)                                                            
	                                                                 
	 flatten_5 (Flatten)         (None, 256)               0         
	                                                                 
	 dropout_7 (Dropout)         (None, 256)               0         
	                                                                 
	 dense_10 (Dense)            (None, 32)                8224      
	                                                                 
	 dense_11 (Dense)            (None, 10)                330       
	                                                                 
	=================================================================
	Total params: 159,018
	Trainable params: 158,506
	Non-trainable params: 512

## 编译、训练模型

这里就是使用训练集和测试集进行模型的训练和验证，速度还是有点慢的，通过打印的结果我们可以看到，最后 accuracy 和 val_accuracy 都在正常进行，基本没有出现过拟合或者欠拟合的风险，只是模型的结构还是很单薄，所以最后的准确率只有 75% 上下，如果用其它专业的大模型，准确率应该在 98% 以上。


	model.compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])
	
	history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

结果打印：

	Epoch 1/10
	1563/1563 [==============================] - 110s 70ms/step - loss: 1.5976 - accuracy: 0.4294 - val_loss: 1.4839 - val_accuracy: 0.4570
	Epoch 2/10
	1563/1563 [==============================] - 110s 71ms/step - loss: 1.2001 - accuracy: 0.5744 - val_loss: 1.3931 - val_accuracy: 0.5108
	......
	Epoch 9/10
	1563/1563 [==============================] - 114s 73ms/step - loss: 0.7181 - accuracy: 0.7515 - val_loss: 0.7916 - val_accuracy: 0.7275
	Epoch 10/10
	1563/1563 [==============================] - 112s 71ms/step - loss: 0.6840 - accuracy: 0.7612 - val_loss: 0.7671 - val_accuracy: 0.7367
	
这里主要展现的是整个训练过程中的，训练集和验证集各自准确率的发展趋势，一般我们可以从图中的曲线可以得知训练的整体情况，如果不符合预期可以进行数据或者模型或者参数的调整，如果符合预期，则可以进行下一步的推理或者预测。

	plt.plot(history.history['accuracy'], label='accuracy')
	plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')
	plt.show()
        

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c232b67f060d4716b0c527a8ed04b77a~tplv-k3u1fbpfcp-watermark.image?)

## 测试模型

使用测试集进行模型的测试工作，因为之前验证集和测试集用的是同一份数据，所以最后的准确率肯定和训练过程的最后的 val_accuracy 是一样的。那么到这里为止，使用卷积神经网络进行 CIFAR 图像的分类任务就告一段落了。


	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	print(test_acc)
	
结果打印：

	313/313 - 5s - loss: 0.7671 - accuracy: 0.7367 - 5s/epoch - 15ms/step
	0.7366999983787537





