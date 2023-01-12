# 前言

本文使用 cpu 的 tensorflow 2.4 完成迁移学习和模型微调，并使用训练好的模型完成猫狗图片分类任务。

预训练模型在 NLP 中最常见的可能就是 BERT 了，在 CV 中我们此次用到了 MobileNetV2 ，它也是一个轻量化预训练模型，它已经经过大量的图片分类任务的训练，里面保存了一个可以通用的去捕获图片特征的模型网络结构，其可以通用地提取出图片的有意义特征。这些特征捕获功能可以轻松迁移到其他图片任务上帮助其完成特征提取工作。

本文的工作，就是在 MobileNetV2 基础上加入我们自定义的若干网络层，通过使用大量的猫狗数据先对新添加的分类器层进行迁移学习，然后结合基础模型的最后几层与我们的自定义分类器一起进行微调训练，就可以轻松获得一个效果很好的猫狗图片分类模型，而不必基于大量数据集从头开始训练一个大型模型，这样会很耗时间。

# 本文大纲

1. 获取数据
2. 数据扩充与数据缩放
3. 迁移学习
4. 微调
5. 预测

# 实现过程

### 1. 获取数据
首先我们要使用 tensorflow 的内置函数，从网络上下载猫狗图片集，训练数据中的猫、狗图片各 1000 张，验证数据中的猫、狗图片各 500 张。每张图片的大小都是（160, 160, 3），每个 batch 中有 32 张图片。

	import matplotlib.pyplot as plt
	import numpy as np
	import os
	import tensorflow as tf
	
	URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
	BATCH_SIZE = 32
	IMG_SIZE = (160, 160)
	
	path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=URL, extract=True)
	PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
	train_dir = os.path.join(PATH, 'train')
	validation_dir = os.path.join(PATH, 'validation')
	train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
	validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
	class_names = train_dataset.class_names

在这里我们挑选了部分图片和标签进行展示。

	plt.figure(figsize=(10, 10))
	for images, labels in train_dataset.take(1):
	    for i in range(3):
	        ax = plt.subplot(3, 3, i + 1)
	        plt.imshow(images[i].numpy().astype("uint8"))
	        plt.title(class_names[labels[i]])
	        plt.axis("off")
	        
《猫狗图片展示》
	        
### 2. 数据扩充与数据缩放

（1）由于我们没有测试集数据，使用 tf.data.experimental.cardinality 确定验证集中有多少个 batch 的数据，然后将其中的 20% 的 batch 编程测试集。

	val_batches = tf.data.experimental.cardinality(validation_dataset)
	test_dataset = validation_dataset.take(val_batches // 5)
	validation_dataset = validation_dataset.skip(val_batches // 5)

（2）为了保证在加载数据的时候不会出现 I/O 不会阻塞，我们在从磁盘加载完数据之后，使用 cache 会将数据保存在内存中，确保在训练模型过程中数据的获取不会成为训练速度的瓶颈。如果说要保存的数据量太大，可以使用 cache 创建磁盘缓存提高数据的读取效率。另外我们还使用 prefetch 在训练过程中可以并行执行数据的预获取。

	AUTOTUNE = tf.data.AUTOTUNE
	train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
	validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
	test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
	
（3）因为我们没有大量的图片数据，我们将现有的图片数据进行旋转或者翻转操作可以增加样本的多样性，这样也有助于减少过拟合现象。RandomFlip 函数将根据 mode 属性将图片进行水平或垂直翻转图像。"可选的有 "horizontal" 、 "vertical" 或 "horizontal_and_vertical" 。"horizontal" 是左右翻转， "vertical" 是上下翻转。RandomRotation 函数通过设置 factor 来讲图片进行旋转，假如 factor=0.2 会将图片在 [-20% * 2pi, 20% * 2pi] 范围内随机旋转。

	data_augmentation = tf.keras.Sequential([
	  tf.keras.layers.RandomFlip('horizontal'),
	  tf.keras.layers.RandomRotation(0.1),
	])	
	
我们这里将随便使用一张图片，使用数据增强的方法对其进行变化，第一张图是原图，其他的都是进行了变化的图片

	for image, _ in train_dataset.take(1):
	    plt.figure(figsize=(10, 10))
	    first_image = image[0]
	    for i in range(3):
	        if  i==0:
	            ax = plt.subplot(3, 3, i + 1)
	            plt.imshow(first_image/ 255)
	            plt.axis('off')
	        else:
	            ax = plt.subplot(3, 3, i + 1)
	            augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
	            plt.imshow(augmented_image[0] / 255)
	            plt.axis('off')
	            
《	            图片旋转和反转》

（4）由于 MobileNetV2 模型的输入值范围是在 [-1, 1] 范围内，但此时我们的图片数据中的像素值处于 [0, 255] 范围内，所以要重新缩放这些像素值， Rescaling 函数可以实现该操作。如果将 [0, 255] 范围的输入缩放到 [0, 1] 的范围，我们可以通过设置参数 scale=1./255 来实现。如果将 [0, 255] 范围内的输入缩放到 [-1, 1] 范围内，我们可以通过设置参数 scale=1./127.5, offset=-1 来实现。

	rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)	
### 3. 迁移学习

（1）我们将 MobileNet V2 模型当做一个基础模型，此模型已基于 ImageNet 数据集进行完美的预训练，我们选择将 MobileNet V2 的倒数第二个网络层，一般来说最后一层就是一个分类器，与最后一层相比，倒数第二层能够保留更丰富的图片特征。在实际操作中我们实例化 MobileNetV2 模型，通过指定 include_top=False 参数，可以加载不包括最顶层的整个预训练网络结果。 该模型的作用就是将（160,160,3）大小的图片转换为（5,5,1280）的特征输出。

（2）第一层是将我们的训练数据都进行数据增强操作，也就是随机的翻转和旋转。

（3）第二层是接收输入为 （160,160,3）大小的图片的输入层。

（4）第三层是我们直接拿来用的 MobileNetV2 ，因为我们要直接使用基础模型的图片特征提取能力，所以为了在训练过程中其权重不发生变化，我们将基础模型中的权重参数都冻结。

（5）第四层是一个池化层，可以将每个 batch 从 (32,5,5,1280) 压缩为 (32,1280) 大小的输出。

（6）第五层是 Dropout ，防止过拟合。

（7）第六层是对该图片一个预测值，也就是 logit 。如果是正数预测标签 1 ，如果是负数预测标签 0 。

（8）我们从模型的 summary 中可以看到，此时我们的模型中的 2259265 个参数被冻结，只有 1281 个参数是可以训练的，它们是分属两个变量的可训练参数，即最后一个全连接层即权重 1280 个可训练参数和偏差 1 个可训练参数。

	IMG_SHAPE = IMG_SIZE + (3,)
	inputs = tf.keras.Input(shape=(160, 160, 3))
	x = data_augmentation(inputs)
	x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
	base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
	base_model.trainable = False
	x = base_model(x, training=False)
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dropout(0.2)(x)
	outputs = tf.keras.layers.Dense(1)(x)
	model = tf.keras.Model(inputs, outputs)
	model.summary()
	
模型结构如下：

	Model: "model_2"
	_________________________________________________________________
	 Layer (type)                Output Shape              Param #   
	=================================================================
	 input_8 (InputLayer)        [(None, 160, 160, 3)]     0         
	                                                                 
	 sequential_2 (Sequential)   (None, 160, 160, 3)       0         
	                                                                 
	 tf.math.truediv_3 (TFOpLamb  (None, 160, 160, 3)      0         
	 da)                                                             
	                                                                 
	 tf.math.subtract_3 (TFOpLam  (None, 160, 160, 3)      0         
	 bda)                                                            
	                                                                 
	 mobilenetv2_1.00_160 (Funct  (None, 5, 5, 1280)       2257984   
	 ional)                                                          
	                                                                 
	 global_average_pooling2d_3   (None, 1280)             0         
	 (GlobalAveragePooling2D)                                        
	                                                                 
	 dropout_2 (Dropout)         (None, 1280)              0         
	                                                                 
	 dense_2 (Dense)             (None, 1)                 1281      
	                                                                 
	=================================================================
	Total params: 2,259,265
	Trainable params: 1,281
	Non-trainable params: 2,257,984
	
模型的中的可训练的 2 个变量为：

	model.trainable_variables
	
结果如下：

	[<tf.Variable 'dense_2/kernel:0' shape=(1280, 1) dtype=float32, numpy=
	 array([[ 0.08899798],
	        [-0.06681276],
	        [ 0.00906871],
	        ...,
	        [-0.00114891],
	        [-0.01134416],
	        [-0.02000826]], dtype=float32)>,
	 <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.03746362], dtype=float32)>]	
	 
	 
（9）选择 Adam 优化器，将我们的学习率设置为 0.0003 。选择 BinaryCrossentropy 作为损失函数。选择常规的 accuracy 作为评估指标。此时我们先对基础模型训练 10 个 epoch 。

	lr = 0.0003
	initial_epochs = 10
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
	              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	              metrics=['accuracy'])
	history = model.fit(train_dataset, validation_data=test_dataset, epochs=initial_epochs)
	
训练过程如下：

	Epoch 1/10
	63/63 [==============================] - 24s 352ms/step - loss: 0.5517 - accuracy: 0.6835 - val_loss: 0.2720 - val_accuracy: 0.8958
	Epoch 2/10
	63/63 [==============================] - 21s 327ms/step - loss: 0.2792 - accuracy: 0.8865 - val_loss: 0.1499 - val_accuracy: 0.9531
	...
	Epoch 9/10
	63/63 [==============================] - 20s 321ms/step - loss: 0.1075 - accuracy: 0.9530 - val_loss: 0.0766 - val_accuracy: 0.9740
	Epoch 10/10
	63/63 [==============================] - 21s 329ms/step - loss: 0.1040 - accuracy: 0.9560 - val_loss: 0.0742 - val_accuracy: 0.9740
	
（10）使用测试数据进行评估

	model.evaluate(validation_dataset)
	
结果如下：

	loss: 0.0664 - accuracy: 0.9765
### 4. 微调

（1）在上面的操作中，我们仅仅在 MobileNetV2 基础模型的顶部添加了一层池化层、一层 Dropout、一层全连接层作为我们的自定义分类器。预训练模型 MobileNetV2 中权重在训练过程中未曾发生过更新。

（2）我们还有一种被称为微调的方法，可以在训练基础模型权重的同时，同时训练我们上面自定义添加的用于分类的分类器。这个微调的训练过程可以将基础模型的通用的图片特征提取能力调整为专门提取本任务中猫狗数据集特征的能力。这个微调的操作只能在我们进行了上面的迁移学习操作之后才能进行，否则如果一开始直接将基础模型和我们自定义的若干层分类器一起进行训练，则会由于随机初始化的分类器导致整个模型的更新梯度太大，从而使得基础模型丧失了其预训练的有效能力。其次在微调过程中我们应该选择性地去微调少量顶部的网络层而不是整个 MobileNet 模型，因为在卷积神经网络中，低层的网络层一般捕获到的是通用的图片特征，这个能力可以泛化应用到几乎所有类型的图片，但是越往顶部的网络层，越来越聚焦于捕获训练时所用到的训练数据的特征，而微调的目标正是是让这个模型更加适用于所用的专门的数据集，也就是本次进行的猫狗图片。

（3）MobileNetV2 模型一共有 154 层结构，我们将模型的前 100 层的参数进行冻结，对顶部的 54 层网络结构中的参数与我们自定义的分类器的若干层一起进行训练，我们通过打印模型的 summary 可以看到，此时共有 54 个可以训练的变量，这些变量中共有可训练的参数 1862721 个。

	base_model.trainable = True
	fine_tune_at = 100
	for layer in base_model.layers[:fine_tune_at]:
	    layer.trainable = False
	print("%d trainable variables "%len(model.trainable_variables))
	model.summary()

结果如下：

	56 trainable variables 
	
	Model: "model_2"
	_________________________________________________________________
	 Layer (type)                Output Shape              Param #   
	=================================================================
	 input_8 (InputLayer)        [(None, 160, 160, 3)]     0         
	                                                                 
	 sequential_2 (Sequential)   (None, 160, 160, 3)       0         
	                                                                 
	 tf.math.truediv_3 (TFOpLamb  (None, 160, 160, 3)      0         
	 da)                                                             
	                                                                 
	 tf.math.subtract_3 (TFOpLam  (None, 160, 160, 3)      0         
	 bda)                                                            
	                                                                 
	 mobilenetv2_1.00_160 (Funct  (None, 5, 5, 1280)       2257984   
	 ional)                                                          
	                                                                 
	 global_average_pooling2d_3   (None, 1280)             0         
	 (GlobalAveragePooling2D)                                        
	                                                                 
	 dropout_2 (Dropout)         (None, 1280)              0         
	                                                                 
	 dense_2 (Dense)             (None, 1)                 1281      
	                                                                 
	=================================================================
	Total params: 2,259,265
	Trainable params: 1,862,721
	Non-trainable params: 396,544
	
（4）之前我们对基础模型训练了 10 个 epoch ，现在我们进行微调过程中再对模型训练 10 个 epoch ，且我们将从上面训练结束的 epoch 开始恢复并进行现在的训练过程。

（5）这里我们改用 RMSprop 优化器，且因为此时我们是要对整体模型进行微调，所以设置的学习率比之前降低 10 倍，否则如果较大会很快产生过拟合。


	fine_tune_epochs = 10
	total_epochs =  initial_epochs + fine_tune_epochs
	model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	              optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr/10),
	              metrics=['accuracy'])
	history_fine = model.fit(train_dataset,  epochs=total_epochs,  initial_epoch=history.epoch[-1],  validation_data=validation_dataset)	
训练结果输出：

	Epoch 10/20
	63/63 [==============================] - 39s 561ms/step - loss: 0.1073 - accuracy: 0.9570 - val_loss: 0.1017 - val_accuracy: 0.9592
	Epoch 11/20
	63/63 [==============================] - 34s 538ms/step - loss: 0.0688 - accuracy: 0.9725 - val_loss: 0.0448 - val_accuracy: 0.9827
	...
	Epoch 19/20
	63/63 [==============================] - 34s 537ms/step - loss: 0.0244 - accuracy: 0.9900 - val_loss: 0.0709 - val_accuracy: 0.9777
	Epoch 20/20
	63/63 [==============================] - 33s 528ms/step - loss: 0.0220 - accuracy: 0.9905 - val_loss: 0.0566 - val_accuracy: 0.9851
	
（6）使用测试数据对模型进行评估。通过此次预训练操作，我们与之前模型对比训练过程的验证准确率和使用测试数据的准确率，发现可以将准确率都有所提高。

	model.evaluate(test_dataset)

结果输出：

	loss: 0.0544 - accuracy: 0.9792

### 5. 预测

我们随机挑选一个 batch 进行预测，并将图片与预测标签进行显示，结果表明预测全都正确。

	image_batch, label_batch = test_dataset.as_numpy_iterator().next()
	predictions = model.predict_on_batch(image_batch).flatten()
	predictions = tf.nn.sigmoid(predictions)
	predictions = tf.where(predictions < 0.5, 0, 1)
	
	plt.figure(figsize=(10, 10))
	for i in range(9):
	    ax = plt.subplot(3, 3, i + 1)
	    plt.imshow(image_batch[i].astype("uint8"))
	    plt.title(class_names[predictions[i]])
	    plt.axis("off")
	   
  《猫狗预测图片》