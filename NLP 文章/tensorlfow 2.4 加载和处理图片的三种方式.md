***本文正在参加[「金石计划 . 瓜分6万现金大奖」](https://juejin.cn/post/7162096952883019783 "https://juejin.cn/post/7162096952883019783")***

# 前言

本文通过使用 cpu 版本的 tensorflow 2.4 ，介绍三种方式进行加载和预处理图片数据。

这里我们要确保 tensorflow 在 2.4 版本以上 ，python 在 3.8 版本以上，因为版本太低有些内置函数无法使用，然后要提前安装好 pillow 和 tensorflow_datasets ，方便进行后续的数据加载和处理工作。

由于本文不对模型进行质量保证，只介绍数据的加载、处理过程，所以只将模型简单训练即可。
	
# 本文大纲

- 使用内置函数读取并处理磁盘数据
- 自定义方式读取和处理磁盘数据
- 从网络上下载数据

# 数据准备

首先我们先准备本文的图片数据，这里我们直接使用 tensorflow 的内置函数，从网络上面下载了一份花朵照片数据集，也可以直接用下面的链接使用迅雷下载。

 	https://storage.googleapis.com/download.tensorflow.org/example\_images/flower\_photos.tgz 
 
数据目录里面包含 5 个子目录，每个子目录对应一个类，分别是雏菊、蒲公英、玫瑰、向日葵、郁金香，图片总共有 3670 张。

	
	import pathlib
	import numpy as np
	import os
	import PIL
	import PIL.Image
	import tensorflow as tf
	import tensorflow_datasets as tfds
	dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
	data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)
	data_dir = pathlib.Path(data_dir)
	image_count = len(list(data_dir.glob('*/*.jpg')))

# 使用内置函数读取并处理磁盘数据

（1）使用 KERAS 内置的函数 image\_dataset\_from\_directory 从本地进行数据的加载，首先定义 batch\_size 为 32 ，每张图片维度大小为 (64,64,3) ，也就是长、宽有 64 个像素点，这里的长、宽像素点数量可以自行修改，需要注意的是数字越大图片越清晰但是后续的计算量会越大，数字越小图片越模糊但是后续的计算量越小。每个像素点是一个 3 维的 RGB 颜色向量。而每个图片对应的标签是一个花朵类别的字符串。我们使用 image\_dataset\_from\_directory 选用了数据中 80% （2936 张）的图片进行训练，20% （734 张）的图片来进行模型效果的验证。 我们将这 5 种图片类别定义为 daisy、 dandelion、roses、 sunflowers、 tulips 保存于 class\_names 。

	batch_size = 32
	height = 64
	width = 64
	train_datas = tf.keras.preprocessing.image_dataset_from_directory( data_dir, validation_split=0.2, subset="training", seed=0,
	                                                                 image_size=(height, width), batch_size=batch_size)
	val_datas = tf.keras.preprocessing.image_dataset_from_directory(  data_dir, validation_split=0.2, subset="validation", seed=0,
	                                                                image_size=(height, width), batch_size=batch_size)
	class_names = train_datas.class_names

(2) 常规的训练流程是我们从磁盘加载好一份数据训练一模型，再去加载下一份数据去训练模型，然后重复这个过程，但是有时候数据集的准备处理非常耗时，使得我们在每次训练前都需要花费大量的时间准备待训练的数据，而此时 CPU 只能等待数据，造成了计算资源和时间的浪费。

（3）在从磁盘加载图片完成后，Dataset.cache() 会将这些图像保留在内存中，这样可以快速的进行数据的获取，如果数据量太大也可以建立缓存。

（4）我们使用 prefetch() 方法，使得我们可以让  Dataset 在训练时预先准备好若干个条数据样本，每次在模型训练的时候都能直接拿来数据进行计算，避免了等待耗时，提高了训练效率。

	AUTOTUNE = tf.data.AUTOTUNE
	train_datas = train_datas.cache().prefetch(buffer_size=AUTOTUNE)
	val_datas = val_datas.cache().prefetch(buffer_size=AUTOTUNE)

（5）这里主要是完成模型的搭建、编译和训练:

* 第一层使用缩放函数 Rescaling 来将 RGB 值压缩，因为每个像素点上代表颜色的 RGB 的三个值范围都是在 0-255 内，所以我们要对这些值进行归一化操作，经过此操作 RGB 三个值的范围被压缩到 0-1 之间，这样可以加速模型的收敛
* 第二、三、四层都是使用了卷积函数，卷积核大小为 3 ，输出一个 32 维的卷积结果向量，并使用 relu 激活函数进行非线性变换，并在卷积之后加入了最大池化层
* 第五层完成了将每张照片的卷积结果向量从三维重新拼接压缩成一维
* 第六层是一个输出为 128 的全连接层，并使用 relu 激活函数进行非线性变换
* 第七层室一个输出为 5 的全连接层，也就是输出层，输出该图片分别属于这 5 种类别的概率分布
* 优化器选择 Adam
* 损失函数选择 SparseCategoricalCrossentropy
* 评估指标选择 Accuracy



		model = tf.keras.Sequential([   tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
		                                tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(),
		                                tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(),
		                                tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(),
		                                tf.keras.layers.Flatten(),
		                                tf.keras.layers.Dense(128, activation='relu'),
		                                tf.keras.layers.Dense(5) ])
		model.compile( optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
		model.fit( train_datas, validation_data=val_datas, epochs=5 )

输出结果：

	Epoch 1/5
	92/92 [==============================] - 10s 101ms/step - loss: 1.5019 - accuracy: 0.3167 - val_loss: 1.1529 - val_accuracy: 0.5177
	Epoch 2/5
	92/92 [==============================] - 6s 67ms/step - loss: 1.1289 - accuracy: 0.5244 - val_loss: 1.0833 - val_accuracy: 0.5736
	...
	Epoch 5/5
	92/92 [==============================] - 6s 65ms/step - loss: 0.8412 - accuracy: 0.6795 - val_loss: 1.0528 - val_accuracy: 0.6172

# 自定义方式读取和处理磁盘数据

（1）上面的过程都是内置的工具包直接将数据进行处理，虽然比较方便，但是可能不够灵活，而在这里我们可以自己手动操作，按照自己的想法将数据进行处理。

（2）从硬盘中读取指定目录中的所有的花朵图片的绝对路径，也就是读取出来的只是图片的绝对路径字符串，如在我的计算机上第一张图片的绝对路径是 

	C:\Users\QJFY-VR\.keras\datasets\flower_photos\roses\24781114_bc83aa811e_n.jpg 
	
然后先将这些数据进行打乱，取 20% 为验证集，取 80% 为训练集。

	datas = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
	datas = datas.shuffle(image_count, reshuffle_each_iteration=False)
	val_size = int(image_count * 0.2)
	train_datas = datas.skip(val_size)
	val_datas = datas.take(val_size)

（3）对训练集和测试集中的每条数据都进行处理，获得最终的图片内容和对应的图片标签： 

* 每张图片的标签，都是通过对每张图片的绝对路径中提取出来的，使用 \\ 分隔符将绝对路径分割成列表，然后取倒数第二个字符串就是其类别标签，并将其转换成 one-hot 向量 
* 每张图片的内容都是通过加载绝对路径，将加载出来的图片内容像素进行指定 height、width 的大小调整进行变化的

		def get_label(file_path):
		    parts = tf.strings.split(file_path, os.path.sep)
		    return tf.argmax(parts[-2] == class_names)
		def decode_img(img):
		    return tf.image.resize(tf.io.decode_jpeg(img, channels=3), [height, width])
		def process_path(file_abs_path):
		    label = get_label(file_abs_path)
		    img = decode_img(tf.io.read_file(file_abs_path))
		    return img, label
		train_datas = train_datas.map(process_path, num_parallel_calls=AUTOTUNE)
		val_datas = val_datas.map(process_path, num_parallel_calls=AUTOTUNE)

（4）将获得的测试集和训练集通过 cache() 保存于内存中，并同样使用 prefetch() 提前加载要使用的数据，使用 shuffle() 将数据进行打散，使用 batch() 每次获取 batch_size 个样本。

（5）使用训练数据训练 5 个 epoch ，并使用验证集进行指标评估 。由于 model 已经被上面的数据进行过训练，所以这里训练过程中从一开始就能看出来 val_accuracy较高。

	def configure_for_performance(ds):
	    ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
	    ds = ds.shuffle(buffer_size=1000).batch(batch_size)
	    return ds
	
	train_datas = configure_for_performance(train_datas)
	val_datas = configure_for_performance(val_datas)
	model.fit( train_datas, validation_data=val_datas, epochs=5 )
结果输出：

	Epoch 1/5
	92/92 [==============================] - 11s 118ms/step - loss: 0.1068 - accuracy: 0.9680 - val_loss: 0.1332 - val_accuracy: 0.9537
	Epoch 2/5
	92/92 [==============================] - 10s 113ms/step - loss: 0.0893 - accuracy: 0.9721 - val_loss: 0.0996 - val_accuracy: 0.9673
	...
	Epoch 5/5
	92/92 [==============================] - 10s 112ms/step - loss: 0.0328 - accuracy: 0.9939 - val_loss: 0.1553 - val_accuracy: 0.9550

# 从网络上下载数据

上面的两个方式都是从本地读取磁盘数据，除此之外我们还可以通过网络来获取数据并进行处理，tfds 中为我们准备了很多种类的数据，包括音频、文本、图片、视频、翻译等数据，通过内置函数 tfds.load 从网络上即可下载指定的数据，这里我们从网络上下载了 tf_flowers 数据，其实就是我们上面用到的磁盘中的花朵磁盘数据数据。

	(train_datas, val_datas, test_datas), metadata = tfds.load(  'tf_flowers', split=['train[:70%]', 'train[70%:90%]', 'train[90%:]'], with_info=True, as_supervised=True)
	train_datas = configure_for_performance(train_datas)
	val_datas = configure_for_performance(val_datas)
	test_datas = configure_for_performance(test_datas)

加载出来数据之后，后面处理的方式可以自行选择，和上面的两种大同小异。