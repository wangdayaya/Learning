# 前言

本文使用 cpu 版本的 tensorflow 2.4 来完成 Stack Overflow 的文本分类，为了顺利进行要保证安装了 2.8 版本以上的 tensorflow_text 。


# 本文大纲

1. 获取 Stack Overflow  数据
2. 两种向量化方式处理数据
3. 训练和评估两种模型
4. 完整模型的搭建和评估
5. 模型预测

# 实现思路

### 1. 获取 Stack Overflow  数据

（1）我们用到的数据是 Stack Overflow 中收集到的关于编程问题的数据集，每个数据样本都是由一个问题和一个标签标签组成，问题是对不同编程语言的技术提问，标签是对应的编程语言标签，分别是 CSharp、Java、JavaScript 或 Python ，表示该问题所属的技术范畴，在数据中分别用 0、1、2、3 四个数字来表示。

（2）我们的数据需要使用 tensorflow 内置函数，从网络上进行下载，下载好的数据分为两个文件夹 train 和 test 表示存放了训练数据和测试数据，而 train 和 test 都一样拥有 CSharp、Java、JavaScript 和 Python 四个文件夹，里面包含了四类编程语言对应的样本，如 train 目录如下所示，test 目录类似。

	train/ 
		...csharp/ 
			......1.txt  
			.......... 
		 ...java/ 
			 ......1.txt  
			 .......... 
		 ...javascript/ 
			 ......1.txt  
			 ..........
		 ...python/ 
			  ......1.txt  
			  ..........

（3）在模型训练过程中要将数据分三份，分别是训练集、验证集和测试集，将训练集中的 20% 当做验证集，也就是说我们最后训练集中有 6400 个样本，验证集中有 1600 个样本，测试集中有 8000 个样本。

（4）为了增加分类的难度，问题中出现的单词 Python、CSharp、JavaScript 或 Java 都被替换为 blank 字符串。

	import collections
	import pathlib
	import tensorflow as tf
	from tensorflow.keras import layers
	from tensorflow.keras import losses
	from tensorflow.keras import utils
	from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
	import tensorflow_datasets as tfds
	import tensorflow_text as tf_text
	
	url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
	batch_size = 32
	seed = 1
	
	dataset_dir = utils.get_file( origin=url, untar=True, cache_dir='stack_overflow', cache_subdir='')
	dataset_dir = pathlib.Path(dataset_dir).parent
	train_dir = dataset_dir/'train'
	test_dir = dataset_dir/'test'
	
	train_datas = utils.text_dataset_from_directory( train_dir, batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)
	val_datas = utils.text_dataset_from_directory( train_dir, batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
	test_datas = utils.text_dataset_from_directory( test_dir, batch_size=batch_size)

### 2. 数据处理
（1）我们在使用数据之前，要进行预处理、分词、向量化。预处理指预处理文本数据，通常就是去掉文本中用不到的标点符号、HTML 元素或者停用词等，精简样本数据内容 。分词就是按空格将一个句子拆分为若干个单词。向量化是指将每个词都转化成对应的整数，以便将它们输入到神经网络中进行运算。

（2）上面说的三个操作我们可以使用 TextVectorization 来一次性搞定，需要注意的是使用 TextVectorization 会将文本默认地转换为小写字母并移除标点符号。在进行分词的时候如果没有明确要求默认会将文本按空格分割。向量化的默认模式为 'int' ，这会为每个单词生成一个对应的整数。还可以选择使用模式 'binary' 来构建词袋模型，也就是用一个长度为 VOCAB\_SIZE 的向量表示一个样本，只有该样本中出现的单词对应的索引位置为 1 ，其他位置都为 0 。

（3）我们现在要将训练数据中的标签去掉，只保留问题描述。然后用 binary\_vectorize\_layer 生成训练数据中单词和对应词袋中的索引，用 int\_vectorize\_layer 生成训练数据中单词和对应的整数。

（4）我们分别使用两种不同的向量化工具生成两类训练集、验证集和测试集。

	VOCAB_SIZE = 10000
	MAX_SEQUENCE_LENGTH = 250
	binary_vectorize_layer = TextVectorization( max_tokens=VOCAB_SIZE, output_mode='binary')
	int_vectorize_layer = TextVectorization( max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=MAX_SEQUENCE_LENGTH)
	train_text = train_datas.map(lambda text, labels: text)
	binary_vectorize_layer.adapt(train_text)
	int_vectorize_layer.adapt(train_text)
	
	def binary_vectorize(text, label):
	    text = tf.expand_dims(text, -1)
	    return binary_vectorize_layer(text), label
	
	def int_vectorize(text, label):
	    text = tf.expand_dims(text, -1)
	    return int_vectorize_layer(text), label
	
	binary_train_datas = train_datas.map(binary_vectorize)
	binary_val_datas = val_datas.map(binary_vectorize)
	binary_test_datas = test_datas.map(binary_vectorize)
	
	int_train_datas = train_datas.map(int_vectorize)
	int_val_datas = val_datas.map(int_vectorize)
	int_test_datas = test_datas.map(int_vectorize)
	
（5）为了保证在加载数据的时候不会出现 I/O 不会阻塞，我们在从磁盘加载完数据之后，使用 cache 会将数据保存在内存中，确保在训练模型过程中数据的获取不会成为训练速度的瓶颈。如果说要保存的数据量太大，可以使用 cache 创建磁盘缓存提高数据的读取效率。另外我们还使用 prefetch 在训练过程中可以并行执行数据的预获取。

（6）我们对两类向量化数据，都进行了同样的 cache 和 prefetch 操作。

	AUTOTUNE = tf.data.AUTOTUNE
	def configure_dataset(dataset):
	    return dataset.cache().prefetch(buffer_size=AUTOTUNE)
	
	binary_train_datas = configure_dataset(binary_train_datas)
	binary_val_datas = configure_dataset(binary_val_datas)
	binary_test_datas = configure_dataset(binary_test_datas)
	
	int_train_datas = configure_dataset(int_train_datas)
	int_val_datas = configure_dataset(int_val_datas)
	int_test_datas = configure_dataset(int_test_datas)	
### 3. 训练和评估两种模型
（1）我们对进行了 binary\_vectorize 操作的训练数据创建一个 binary\_model 模型，模型只是一个非常简单的结构，只有一层输入为 4 个维度向量的全连接操作，其实就是输出该问题样本分别属于四种类别的概率分布。模型的损失函数选择 SparseCategoricalCrossentropy ，模型的优化器选择 adam ，模型的评估指标选择为 accuracy 。

	binary_model = tf.keras.Sequential([layers.Dense(4)])
	binary_model.compile(   loss=losses.SparseCategoricalCrossentropy(from_logits=True),
	                        optimizer='adam',
	                        metrics=['accuracy'])
	binary_model.fit( binary_train_datas, validation_data=binary_val_datas, epochs=20)
	
训练过程如下：

	Epoch 1/20
	200/200 [==============================] - 3s 12ms/step - loss: 1.1223 - accuracy: 0.6555 - val_loss: 0.9304 - val_accuracy: 0.7650
	Epoch 2/20
	200/200 [==============================] - 3s 12ms/step - loss: 0.7780 - accuracy: 0.8248 - val_loss: 0.7681 - val_accuracy: 0.7856
	...
	Epoch 19/20
	200/200 [==============================] - 3s 13ms/step - loss: 0.1570 - accuracy: 0.9812 - val_loss: 0.4923 - val_accuracy: 0.8087
	Epoch 20/20
	200/200 [==============================] - 2s 12ms/step - loss: 0.1481 - accuracy: 0.9836 - val_loss: 0.4929 - val_accuracy: 0.8081
	
（2）我们对进行了 int\_vectorize 操作的训练数据创建一个 int\_model 模型， 第一层是对所有的单词都进行词嵌入操作，将每个单词转换成 64 维向量，这里因为 0 要进行 padding 操作，所以在进行词嵌入的时候要额外加 1 。第二层使用卷积函数，输出 64 维的向量，卷积核大小为 5 ，步长为 2，并使用了 padding 操作进行补齐，激活函数使用的是 relu 。第三层是一个最大池化层。第四层是一个输出 4 个维度的向量的全连接层，其实就是输出该问题样本分别属于四种类别的概率分布。

	def create_model():
	    model = tf.keras.Sequential([
	      layers.Embedding(VOCAB_SIZE + 1, 64, mask_zero=True),
	      layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
	      layers.GlobalMaxPooling1D(),
	      layers.Dense(4) ])
	    return model
	
	int_model = create_model()
	int_model.compile(
	    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
	    optimizer='adam',
	    metrics=['accuracy'])
	int_model.fit(int_train_datas, validation_data=int_val_datas, epochs=20)

训练过程如下：

	Epoch 1/20
	200/200 [==============================] - 5s 24ms/step - loss: 1.1247 - accuracy: 0.5069 - val_loss: 0.7829 - val_accuracy: 0.6756
	Epoch 2/20
	200/200 [==============================] - 5s 24ms/step - loss: 0.6034 - accuracy: 0.7708 - val_loss: 0.5820 - val_accuracy: 0.7681
	...
	Epoch 19/20
	200/200 [==============================] - 5s 24ms/step - loss: 5.8555e-04 - accuracy: 1.0000 - val_loss: 0.7737 - val_accuracy: 0.8019
	Epoch 20/20
	200/200 [==============================] - 5s 24ms/step - loss: 4.9328e-04 - accuracy: 1.0000 - val_loss: 0.7853 - val_accuracy: 0.8019
	
	
（3）我们使用各自的测试数据分别对两个模型进行测试。

	_, binary_accuracy = binary_model.evaluate(binary_test_datas)
	_, int_accuracy = int_model.evaluate(int_test_datas)
	
	print("Binary model accuracy: {:2.2%}".format(binary_accuracy))
	print("Int model accuracy: {:2.2%}".format(int_accuracy))

测试结果为：

	Binary model accuracy: 81.29%
	Int model accuracy: 80.67%
### 4. 完整模型的搭建和评估
（1）为了快速搭建模型，我们还可以将  binary\_vectorize\_layer 与  binary\_model 进行拼接，最后再接入 sigmoid 激活函数，这样模型就有了能够处理原始文本的能力。

（2）因为模型的功能完整，所以我们只需要将原始的测试文本数据传入即可。

	whole_model = tf.keras.Sequential(
	    [binary_vectorize_layer, 
	     binary_model,
	     layers.Activation('sigmoid')])
	
	whole_model.compile(
	    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
	    optimizer='adam',
	    metrics=['accuracy'])
	
	_, accuracy = whole_model.evaluate(test_datas)
	print("Accuracy: {:2.2%}".format(accuracy))
	
测试结果为：

	Accuracy: 81.29%
### 5. 模型预测

我们分别使用两个样本来用模型进行预测，可以看出预测结果准确。

	def get_sample(predicteds):
	    int_labels = tf.math.argmax(predicteds, axis=1)
	    predicted_labels = tf.gather(train_datas.class_names, int_labels)
	    return predicted_labels
	    
	inputs = [
	    "how do I extract keys from a dict into a list?",  
	    "debug public static void main(string[] args) {...}",  
	]
	predicted_scores = whole_model.predict(inputs)
	predicted_labels = get_sample(predicted_scores)
	for t, label in zip(inputs, predicted_labels):
	    print("Question: ", t)
	    print("Predicted label: ", label.numpy())    
	    
预测结果：

	Question:  how do I extract keys from a dict into a list?
	Predicted label:  b'python'
	Question:  debug public static void main(string[] args) {...}
	Predicted label:  b'java'  