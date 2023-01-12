# 前言

本文主要使用 cpu 版本的 tensorflow 2.4 版本完成文本的 word embedding 训练，并且以此为基础完成影评文本分类任务。

# 本文大纲

1. 三种文本向量化方法
2. 获取数据
3. 处理数据
4. 搭建、训练模型
5. 导出训练好的词嵌入向量


# 具体介绍
### 1. 三种文本向量化方法

通常在深度学习模型中我们的输入都是以向量形式存在的，所以我们处理数据过程的重要一项任务就是将文本中的 token （一个 token 可以是英文单词、一个汉字、一个中文词语等，需要自己规定）转换成对应的向量，本文会给出三种常见文本向量化的策略。

（1）One-Hot Encodings 。其实很好理解，假如我们的数据是“我是人”，因为有 3 个不同的汉字，我会给每个汉字一个对应的索引，然后我会创建一个长度为 3 的向量，假如我给每个汉字赋予的索引为“我->0”“是->1”“人->2”，那么每个字对应的 One-Hot Encodings 为 [1,0,0]、[0,1,0]、[0,0,1] 。那么“我是人”的这个句子的向量表示就可以将这三个向量拼接起来即可。这种方法的优点明显，方便理解和实现，但是缺点也很明显，效率非常低。One-Hot Encodings 所产生的的向量都是稀疏的。假如词汇表中有 1000 个单词，要对每个单词进行向量化编码，其中几乎 99% 的位置都为零。

（2）encode each word with a unique num 。我们可以使用唯一的数字对每个单词进行编码。还是上面的例子，我们给每个字分配一个对应的整数，假如分配结果为 “我->1”“是->2”“人->3”，我就能将句子“我是人”这句话就可以编码为一个稠密向量，如 [1,2,3]。此时的向量是一个稠密向量（所有位置都有有意义的整数填充）。但是这种方法有个缺点，编码的数字是可以人为任意设置，它不能捕获汉字之间的任何语义关系，也无法从数字上看出对应的近义词之间的关系。

（3）Word Embeddings 。词嵌入是一种将单词编码为有效稠密向量的方法，其中相似的单词具有相似相近的向量编码。词嵌入是浮点类型的稠密向量，向量的长度需要人为指定。我们不必像上面两种方法手动去设置编码中的向量值，而是将他们都作为可训练的参数，通过给模型喂大量的数据，不断的训练来捕获单词之间的细粒度语义关系，常见的词向量维度可以设置从 8 维到 1024 维范围中的任意整数。理论上维度越高词嵌入的语义越丰富但是训练成本越高。如我们上面的例子，我们设置词嵌入维度为 4 ，最后通过训练得到的词嵌入可能是 “我->[-3.2, 1.5, -4,6, 3.4]”“是-> [0.2, 0.6, -0.6, 1.5]”“人->[3.4, 5.3, -7.2, 1.5]”。


### 2. 获取数据

（1）本次我们要用到的是数据是 Large Movie Review Dataset ，我们需要使用 tensorflow 的内置函数从网络上下载到本地磁盘，为了简化数据，我们将训练数据目录中的 unsup 子目录都删除，最后取出 20000 个训练样本作为训练集，取出 5000 个训练样本作为验证集。

	import io
	import os
	import re
	import shutil
	import string
	import tensorflow as tf
	from tensorflow.keras import Sequential
	from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
	from tensorflow.keras.layers import TextVectorization
	
	batch_size = 512
	seed = 1
	url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
	
	dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,  untar=True, cache_dir='.', cache_subdir='')
	dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
	train_dir = os.path.join(dataset_dir, 'train')
	remove_dir = os.path.join(train_dir, 'unsup')
	shutil.rmtree(remove_dir)
	
	train_datas = tf.keras.utils.text_dataset_from_directory( 'aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)
	val_datas = tf.keras.utils.text_dataset_from_directory( 'aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)

（2）这里展示出 2 条样本，每个样本都有一个标签和一个文本描述，标签 1 表示评论是 positive , 标签 0 表示评论是: negative 。

	1 b'The first time I saw this film, I was in shock for days afterwards. Its painstaking and absorbing treatment of the subject holds the attention, helped by good acting and some really intriguing music. The ending, quite simply, had me gasping. First rate!'
	0 b"This is quite possibly the worst movie of all time. It stars Shaquille O'Neil and is about a rapping genie. Apparently someone out there thought that this was a good idea and got suckered into dishing out cash to produce this wonderful masterpiece. The movie gets 1 out of 10."
### 3. 处理数据
（1）为了保证在加载数据的时候不会出现 I/O 不会阻塞，我们在从磁盘加载完数据之后，使用 cache 会将数据保存在内存中，确保在训练模型过程中数据的获取不会成为训练速度的瓶颈。如果说要保存的数据量太大，可以使用 cache 创建磁盘缓存提高数据的读取效率。另外我们还使用 prefetch 在训练过程中可以并行执行数据的预获取。

	AUTOTUNE = tf.data.AUTOTUNE
	train_datas = train_datas.cache().prefetch(buffer_size=AUTOTUNE)
	val_datas = val_datas.cache().prefetch(buffer_size=AUTOTUNE)
	
（2）将训练数据中的标签去掉，只保留文本描述，然后使用 TextVectorization 对数据进行预处理，先转换层小写英文，然后再将无用的字符剔除，并且我们规定了每个文本的最大长度为 100 个单词，超过的文本部分会被丢弃。最后将训练数据中的词都放入一个最大为 10000 的词汇表中，其中有一个特殊的表示 OOV 的 [UNK] ，也就说来自训练数据中的词只有 9999 个，使用 vectorize_layer 为每个单词进行 int 向量化，其实就是在文章开头提到的第二种向量化策略。

	def handle(input_data):
	    lowercase = tf.strings.lower(input_data)
	    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
	    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')
	
	vocab_size = 10000
	sequence_length = 100
	vectorize_layer = TextVectorization(standardize=handle,
	                                    max_tokens=vocab_size,
	                                    output_mode='int',
	                                    output_sequence_length=sequence_length)
	
	text_datas = train_datas.map(lambda x, y: x)
	vectorize_layer.adapt(text_datas)
	
	
### 4. 搭建、训练模型

我们此次搭建的模型是一个“Continuous bag of words" 风格的模型。

（1）第一层是已经上面初始化好的 vectorize_layer ，它可以将文本经过预处理，然后将分割出来的单词都赋予对应的整数。 

（2）第二层是一个嵌入层，我们定义了词嵌入维度为 32，也就是为每一个词对应的整数都转换为一个 32 维的向量来进行表示，这些向量的值是可以在模型训练时进行学习的权重参数。通过此层输出的维度为：（batch_size, sequence_length, embedding_dim）。 

（3）第三层是一个 GlobalAveragePooling1D 操作，因为每个样本的维度为 (sequence_length, embedding_dim) ，该操作可以按照对 sequence_length 维度求平均值来为每个样本返回一个固定长度的输出向量，最后输出的维度为：（batch_size, embedding_dim）。 

（4）第四层是一个输出 32 维向量的全连接层操作，并且使用 relu 激活函数进行非线性变化。 

（5）最后一层是一个输出 1 维向量的全连接层操作，表示该样本的属于 positive 的概率。

（6）优化器选择 Adam ，损失函数为 BinaryCrossentropy ，评估指标为 accuracy

	embedding_dim=32
	model = Sequential([
	  vectorize_layer,
	  Embedding(vocab_size, embedding_dim, name="embedding"),
	  GlobalAveragePooling1D(),
	  Dense(32, activation='relu'),
	  Dense(1)
	])
	model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  metrics=['accuracy'])
	model.fit(train_datas, validation_data=val_datas, epochs=20, callbacks=[tensorboard_callback])

训练过程打印：

	Epoch 1/20
	40/40 [==============================] - 3s 52ms/step - loss: 0.6898 - accuracy: 0.4985 - val_loss: 0.6835 - val_accuracy: 0.5060
	Epoch 2/20
	40/40 [==============================] - 2s 50ms/step - loss: 0.6654 - accuracy: 0.4992 - val_loss: 0.6435 - val_accuracy: 0.5228
	...
	Epoch 19/20
	40/40 [==============================] - 2s 49ms/step - loss: 0.1409 - accuracy: 0.9482 - val_loss: 0.4532 - val_accuracy: 0.8210
	Epoch 20/20
	40/40 [==============================] - 2s 48ms/step - loss: 0.1327 - accuracy: 0.9528 - val_loss: 0.4681 - val_accuracy: 0.8216	

### 5. 导出训练好的词嵌入向量

这里我们取出已经训练好的词嵌入，然后打印出前三个单词以及词向量，因为索引 0 的词是空字符，所以直接跳过了，只显示了两个单词的内容。我们可以将所有训练好的词嵌入向量都写入本地磁盘的文件，供以后使用。

	weights = model.get_layer('embedding').get_weights()[0]
	vocab = vectorize_layer.get_vocabulary()
	for i, word in enumerate(vocab[:3]):
	    if i == 0:
	        continue   
	    vecoter = weights[i]
	    print(word,"||", ','.join([str(x) for x in vecoter]))
    
单词和对应词嵌入向量：

	[UNK] || 0.020502748,-0.038312573,-0.036612183,-0.050346173,-0.07899615,-0.03143682,-0.06429587,0.07334388,-0.01887771,-0.08744612,-0.021639654,0.04726765,0.042426057,0.2240213,0.022607388,-0.08052631,0.023943739,0.05245169,-0.017815227,0.053340062,-0.033523336,0.057832733,-0.007486237,-0.16336738,0.022891225,0.12611994,-0.11084395,-0.0076115266,-0.03733231,-0.010371257,-0.045643456,-0.05392711
	the || -0.029460065,-0.0021714368,-0.010394105,-0.03353872,-0.097529344,-0.05249973,-0.03901586,0.009200298,-0.085409686,-0.09302798,-0.07607663,0.046305165,-0.010357974,0.28357282,0.009442638,-0.036655612,0.063269086,0.06721396,0.063007854,0.03185595,-0.014642656,0.089468665,-0.014918188,-0.15671577,0.043026615,0.17086154,-0.0461816,0.021180542,-0.045269016,-0.101499856,-0.03948177,0.028299723    