# 前言


本文使用 cpu 版本的 tensorflow-2.1 完成对 heart 数据的分类预测。


# 本文大纲

1. 获取并处理 Heart 数据
2. 使用 Dataset 制作输入数据并进行模型训练
3. 使用字典制作输入数据并进行模型训练

# 实现过程

### 1. 获取并处理 Heart 数据


（1）本文使用的数据集是由克利夫兰诊所心脏病基金会提供的一份小型数据集，里面仅有 303 行数据，每行数据都表示一个患者的各项指标情况，本文的任务就是搭建一个小型的深度学习模型，预测每位患者是否会得心脏病，这是一个二分类问题。

（2）本文的数据在 tensorflow 中已经处理好，我们直接使用内置函数从网络上下载到本地磁盘即可。

（3）为了方便数据的处理我们将所有的列的内容都处理成数值，其中 df['thal']  中有不同的字符串，我们将其各个字符串转换成对应的离散值进行表示。
	
	import pandas as pd
	import tensorflow as tf
	csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')
	df = pd.read_csv(csv_file)
	df['thal'] = pd.Categorical(df['thal'])
	df['thal'] = df.thal.cat.codes
	
（4）df['target'] 这一列主要是每行数据的标签，所以我们将这一列单独提取出来。

	labels = df.pop('target')
	df.head(3)
	
数据前三行展示如下：



	index age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal
	0	63	1	1	145	233	1	2	150	0	2.3	3	0	2
	1	67	1	4	160	286	0	2	108	1	1.5	2	3	3
	2	67	1	4	120	229	0	2	129	1	2.6	2	2	4
### 2. 使用 Dataset 制作输入数据并进行模型训练



（1）第一层是一个输出 32 个维度向量的全连接层，并使用了 relu 激活函数进行非线性变化。

（2）第二层是一个输出 32 个维度向量的全连接层，并使用了 relu 激活函数进行非线性变化。

（3）第三层是一个输出 1 个维度向量的全连接层，并使用了 sigmoid 激活函数进行非线性变化，这里其实就是输出层，输出 0 或者 1  。

（4） 模型的优化器选择 adam，模型的损失值选择 binary_crossentropy，模型的评估指标选择 accuracy 。

（5）我们将数据和标签做成训练数据，每个 batch 设置为 16 ，使用训练数据将模型训练 20 个 epoch 。

	def get_model():
	    model = tf.keras.Sequential([
	        tf.keras.layers.Dense(32, activation='relu'),
	        tf.keras.layers.Dense(32, activation='relu'),
	        tf.keras.layers.Dense(1, activation='sigmoid') ])
	    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	    return model
	model = get_model()
	train_datas = tf.data.Dataset.from_tensor_slices((df.values, labels.values))
	train_datas = train_datas.shuffle(len(df)).batch(16)
	model.fit(train_datas, epochs=20)
	
结果如下所示：

	Epoch 1/20
	19/19 [==============================] - 0s 15ms/step - loss: 2.4584 - accuracy: 0.5446
	Epoch 2/20
	19/19 [==============================] - 0s 2ms/step - loss: 0.8050 - accuracy: 0.6766
	...
	Epoch 19/20
	19/19 [==============================] - 0s 1ms/step - loss: 0.4237 - accuracy: 0.7921
	Epoch 20/20
	19/19 [==============================] - 0s 1ms/step - loss: 0.4523 - accuracy: 0.7789
### 3. 使用字典制作输入数据并进行模型训练

（1）上面的方法是将每一行样本数据传入模型进行训练的，我们还可以将字典作为输入传入到模型中进行训练。

（2）先创建了一个字典输入 inputs ，每列列名对应着一个张量。

（3）然后将输入转换为一个大小为 [None, 13] 的张量，其实就是将数据从列形式转换成了行形式，便于后面在模型训练时候的数据使用。

（4）然后将 inputs 传入到一个中间层，它是一个可以输出一个 10 维向量的全连接层，激活函数为 relu  。

（5）最后是一个可以输出一个 1 维向量的全连接层，激活函数为 sigmoid ，输出的便是该患者是否患心脏病的预测标签。

（6）模型的优化器选择为 adam ，损失函数为  binary_crossentropy ，评估指标为 accuracy 。

（7）最后将 df 数据转换成了字典当做训练数据，labels 当做训练标签进行模型的训练即可，总共训练 20 个 epoch ，每个 batch 的大小设置为 16 。

	inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
	mid_layer = tf.stack(list(inputs.values()), axis=-1)
	mid_layer = tf.keras.layers.Dense(20, activation='relu')(mid_layer)
	outputs = tf.keras.layers.Dense(1, activation='sigmoid')(mid_layer)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	train_datas = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), labels.values)).batch(16)
	model.fit(train_datas, epochs=20)
	
结果如下所示：

	Epoch 1/20
	19/19 [==============================] - 0s 12ms/step - loss: 43.7227 - accuracy: 0.2739
	Epoch 2/20
	19/19 [==============================] - 0s 2ms/step - loss: 20.6211 - accuracy: 0.2739
	...
	Epoch 19/20
	19/19 [==============================] - 0s 2ms/step - loss: 0.6439 - accuracy: 0.7261
	Epoch 20/20
	19/19 [==============================] - 0s 2ms/step - loss: 0.6234 - accuracy: 0.7294