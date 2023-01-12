***本文正在参加[「金石计划 . 瓜分6万现金大奖」](https://juejin.cn/post/7162096952883019783 "https://juejin.cn/post/7162096952883019783")***

# 前言
本文使用 cpu 版本的 TensorFlow 2.4 ，分别搭建单层 Bi-LSTM 模型和多层 Bi-LSTM 模型完成文本分类任务。

确保使用 numpy == 1.19.0 左右的版本，否则在调用 TextVectorization 的时候可能会报 NotImplementedError 。
# 本文大纲
1. 获取数据
2. 处理数据
3. 单层 Bi-LSTM 模型
4. 多层 Bi-LSTM 模型

# 实现过程

### 1. 获取数据

（1）我们本文用到的数据是电影的影评数据，每个样本包含了一个对电影的评论文本和一个情感标签，1 表示积极评论，0 表示负面评论，也就是说这是一份二分类的数据。

（2）我们通过 TensorFlow 内置的函数，可以从网络上直接下载 imdb_reviews 数据到本地的磁盘，并取出训练数据和测试数据。

（3）通过使用 tf.data.Dataset 相关的处理函数，我们将训练数据和测试数据分别进行混洗，并且设置每个 batch 大小都是 64 ，每个样本都是 (text, label) 的形式。如下我们取了任意一个 batch 中的前两个影评文本和情感标签。


	import numpy as np
	import tensorflow_datasets as tfds
	import tensorflow as tf
	import matplotlib.pyplot as plt
	tfds.disable_progress_bar()
	
	BUFFER_SIZE = 10000
	BATCH_SIZE = 64
	dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
	train_dataset, test_dataset = dataset['train'], dataset['test']
	train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
	test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
	for example, label in train_dataset.take(1):
	    print('text: ', example.numpy()[:2])
	    print('label: ', label.numpy()[:2])
	    
 部分样本显示：
 
     text: [ 
         b"First of all, I have to say I have worked for blockbuster and have seen quite a few movies to the point its tough for me to find something I haven't seen. Taking this into account, I want everyone to know that this movie was by far the worst film ever made, it made me pine for Gigli, My Boss's Daughter, and any other piece of junk you've ever seen. BeLyt must be out of his mind, I've only found one person who liked it and even they couldn't tell me what the movie was about. If you are able to decipher this movie and are able to tell me what it was about you have to either be the writer or a fortune teller because there's any other way a person could figure this crap out.<br /><br />FOR THE LOVE OF G-D STAY AWAY!"
         b"Just got out and cannot believe what a brilliant documentary this is. Rarely do you walk out of a movie theater in such awe and amazement. Lately movies have become so over hyped that the thrill of discovering something truly special and unique rarely happens. Amores Perros did this to me when it first came out and this movie is doing to me now. I didn't know a thing about this before going into it and what a surprise. If you hear the concept you might get the feeling that this is one of those touchy movies about an amazing triumph covered with over the top music and trying to have us fully convinced of what a great story it is telling but then not letting us in. Fortunetly this is not that movie. The people tell the story! This does such a good job of capturing every moment of their involvement while we enter their world and feel every second with them. There is so much beyond the climb that makes everything they go through so much more tense. Touching the Void was also a great doc about mountain climbing and showing the intensity in an engaging way but this film is much more of a human story. I just saw it today but I will go and say that this is one of the best documentaries I have ever seen."
     ]
	label:  [0 1]
	
	
### 2. 处理数据

（1）想要在模型中训练这些数据，必须将这些文本中的 token 都转换成机器可以识别的整数，最简单的方法就是使用 TextVectorization 来制作一个编码器 encoder，这里只将出现次数最多的 1000 个 token 当做词表，另外规定每个影评处理之后只能保留最长 200 的长度，如果超过则会被截断，如果不足则用填充字符对应的整数 0 补齐。

（2）这里展现出来了某个样本的经过整数映射止之后的结果，可以看到影评对应的整数数组长度为 200 。

	MAX_SEQ_LENGTH = 200
	VOCAB_SIZE = 1000
	encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_SEQ_LENGTH)
	encoder.adapt(train_dataset.map(lambda text, label: text))
	vocab = np.array(encoder.get_vocabulary())
	encoded_example = encoder(example)[:1].numpy()
	print(encoded_example)
	print(label[:1])
	
随机选取一个样本进行证书映射结果：

	[[ 86   5  32  10  26   6 130  10  26 926  16   1   3  26 108 176   4 164
	   93   6   2 215  30   1  16  70   6 160 140  10 731 108 647  11  78   1
	   10 178 305   6 118  12  11  18  14  33 234   2 240  20 122  91   9  91
	   70   1  16   1  56   1 580   3  99  81 408   5   1 825 122 108   1 217
	   28  46   5  25 349 195  61 249  29 409  37 405   9   3  54  35 404 360
	   70  49   2  18  14  43  45  23  24 491   6   1  11  18   3  24 491   6
	  360  70  49   9  14  43  23  26   6 352  28   2 762  42   4   1   1  80
	  213  99  81  97   4 409  96 811  11 638   1  13  16   2 116   5   1 766
	  242   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
	    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
	    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
	    0   0]]
	tf.Tensor([0], shape=(1,), dtype=int64)
### 3. 单层 Bi-LSTM 模型




（1） 第一层是我们刚才定义好的 encoder ，将输入的文本进行整数的映射。

（2）第二层是 Embedding 层，我们这里设置了每个词的词嵌入维度为 32 维。

（3）第三层是 Bi-LSTM 层，这里我们设置了每个 LSTM 单元的输出维度为 16 维。

（4）第四层是一个输出 8 维向量的全连接层，并且使用的 relu 激活函数。

（5）第五层是 Dropout ，设置神经元丢弃率为 0.5 ，主要是为了防止过拟合。

（6）第六层是一个输出 1 维向量的全连接层，也就是输出层，表示的是该样本的 logit 。

	model = tf.keras.Sequential([
	    encoder,
	    tf.keras.layers.Embedding( input_dim=len(encoder.get_vocabulary()), output_dim=32, mask_zero=True),
	    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
	    tf.keras.layers.Dense(8, activation='relu'),
	    tf.keras.layers.Dropout(0.5),
	    tf.keras.layers.Dense(1)
	])


（7）在没有经过训练的模型上对文本进行预测，如果输出小于 0 则为消极评论，如果大于 0 则为积极评论，我们可以看出这条评论本来应该是积极评论，但是却输出的 logit 却是负数，即错误预测成了消极的。

	sample_text = ('The movie was cool. The animation and the graphics were out of this world. I would recommend this movie.')
	model.predict(np.array([sample_text]))
	
预测结果为：

	array([[-0.01437075]], dtype=float32)
	
（8）我们使用 BinaryCrossentropy 作为损失函数，需要注意的是如果模型输出结果给到 BinaryCrossentropy 的是一个 logit 值（值域范围 [-∞, +∞] ），则应该设置 from_logits=True 。如果模型输出结果给到 BinaryCrossentropy 的是一个概率值 probability （值域范围 [0, 1] ），则应该设置为 from_logits=False 。

（9）我们使用 Adam 作为优化器，并且设置学习率为 1e-3 。

（10）我们使用准确率 accuracy 作为评估指标。

（11）使用训练数据训练 10 个 epoch，同时每经过一个 epoch 使用验证数据对模型进行评估。

	model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	              optimizer=tf.keras.optimizers.Adam(1e-3),
	              metrics=['accuracy'])
	history = model.fit(train_dataset, epochs=10,  validation_data=test_dataset, validation_steps=30)
	
训练过程如下：

	Epoch 1/10
	391/391 [==============================] - 30s 65ms/step - loss: 0.6461 - accuracy: 0.5090 - val_loss: 0.4443 - val_accuracy: 0.8245
	Epoch 2/10
	391/391 [==============================] - 23s 58ms/step - loss: 0.4594 - accuracy: 0.6596 - val_loss: 0.3843 - val_accuracy: 0.8396
	...
	Epoch 10/10
	391/391 [==============================] - 22s 57ms/step - loss: 0.3450 - accuracy: 0.8681 - val_loss: 0.3920 - val_accuracy: 0.8417

（12）训练结束之后使用测试数据对模型进行测试，准确率可以达到 0.8319 。如果经过超参数的调整和足够的训练时间，效果会更好。

	model.evaluate(test_dataset)

结果为：

	391/391 [==============================] - 6s 15ms/step - loss: 0.3964 - accuracy: 0.8319
	
（13）使用训练好的模型对影评进行分类预测，可以看出可以正确得识别文本的情感取向。因为负数表示的就是影评为负面情绪的。

	sample_text = ('The movie was not cool. The animation and the graphics were bad. I would not recommend this movie.')
	model.predict(np.array([sample_text]))

结果为：

	array([[-1.6402857]], dtype=float32)
### 4. 多层 Bi-LSTM 模型

（1）我们上面只是搭建了一层的 Bi-LSTM ，这里我们搭建了两层的 Bi-LSTM 模型，也就是在第二层 Bidirectional 之后又加了一层 Bidirectional ，这样可以使我们的模型更加有效。我们使用的损失函数、优化器、评估指标都和上面一样。

	model = tf.keras.Sequential([
	    encoder,
	    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 32, mask_zero=True),
	    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,  return_sequences=True)),
	    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
	    tf.keras.layers.Dense(8, activation='relu'),
	    tf.keras.layers.Dropout(0.5),
	    tf.keras.layers.Dense(1)
	])
	model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'])
	history = model.fit(train_dataset, epochs=10, validation_data=test_dataset,  validation_steps=30)
	
训练过程如下：

	Epoch 1/10
	391/391 [==============================] - 59s 124ms/step - loss: 0.6170 - accuracy: 0.5770 - val_loss: 0.3931 - val_accuracy: 0.8135
	Epoch 2/10
	391/391 [==============================] - 45s 114ms/step - loss: 0.4264 - accuracy: 0.7544 - val_loss: 0.3737 - val_accuracy: 0.8380
		
	...
	Epoch 10/10
	391/391 [==============================] - 45s 114ms/step - loss: 0.3138 - accuracy: 0.8849 - val_loss: 0.4069 - val_accuracy: 0.8323	
（2）训练结束之后使用测试数据对模型进行测试，准确率可以达到 0.8217 。如果经过超参数的调整和足够的训练时间，效果会更好。

	model.evaluate(test_dataset)
	
结果为：

	391/391 [==============================] - 14s 35ms/step - loss: 0.4021 - accuracy: 0.8217
	
（3）使用训练好的模型对影评进行分类预测，可以看出可以正确得识别文本的情感取向。因为正数表示的就是影评为积极情绪的。

	sample_text = ('The movie was good. The animation and the graphics were very good. you should love movie.')
	model.predict(np.array([sample_text]))

结果为：

	array([[3.571126]], dtype=float32)