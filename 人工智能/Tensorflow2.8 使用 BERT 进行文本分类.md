# 前言

本文使用 cpu 版本的 Tensorflow 2.8 ，通过搭建 BERT 模型完成文本分类任务。

# 大纲

1. python 库准备
2. BERT 是什么？
3. 获取并处理 IMDB 数据
4. 初始 TensorFlow Hub 中的 BERT 处理器和模型
5. 搭建模型
6. 训练模型
7. 测试模型
8. 保存模型
9. 重新加载模型并进行预测


# 1. python 库准备

为了保证能正常运行本文代码，需要保证以下库的版本：

* tensorflow==2.8.4 
* tensorflow-text==2.8.1 
* tf-models-official==2.7.0 
* python==3.8.0

在安装 tf-models-official 的时候可能会报错 ：Microsoft Visual C++ 14.0 or greater is required 。直接进入 https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/ 这里进行下载新的Microsoft C++ 生成工具，然后安装重启电脑即可。

# 2. BERT 是什么？

BERT 和其他 Transformer 编码器架构模型都在 NLP 的各种任务上取得了巨大的成功。它们都是使用了多层的注意力机制，可以有效地对文本进行双向的深层次语义编码表示。BERT 模型已经在大型文本语料库上进行了充足的预训练，我们在使用的时候只需要针对特定任务进行微调即可。


# 3. 获取并处理 IMDB 数据

（1）使用 tensorflow 的内置函数，从网络上将 Large Movie Review Dataset 数据下载到本地，没有特别指定的话一般位置在当前同级目录下。此数据集是一个电影评论数据集，其中包含来自 Internet 电影数据库的 50000 条电影评论的文本，每个文本都对应一个标签标记其为积极或者消极的。

（2）我们将数据中无用的 unsup 文件夹都删掉，这样后面处理数据会更加方便。

	import os
	import shutil
	import tensorflow as tf
	import tensorflow_hub as hub
	import tensorflow_text as text
	from official.nlp import optimization
	import matplotlib.pyplot as plt
	tf.get_logger().setLevel('ERROR')
	url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
	dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url, untar=True, cache_dir='.', cache_subdir='')
	dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
	train_dir = os.path.join(dataset_dir, 'train')
	remove_dir = os.path.join(train_dir, 'unsup')
	shutil.rmtree(remove_dir)

（3）我们可以直接使用内置函数 text\_dataset\_from\_directory 直接从硬盘读取数据生成 tf.data.Dataset 。

（4）IMDB 数据集已经被分为了训练集和测试集，但是还缺少验证集，所以让我们需要从训练集中取出 20% 来创建一个验证集。最终训练集 20000 个样本，验证集 5000 个样本，测试集 25000 个样本。每个样本都是 (text,label) 对。

（5）为了保证在加载数据的时候不会出现 I/O 不会阻塞，我们在从磁盘加载完数据之后，使用 cache 会将数据保存在内存中，确保在训练模型过程中数据的获取不会成为训练速度的瓶颈。如果说要保存的数据量太大，可以使用 cache 创建磁盘缓存提高数据的读取效率。另外我们还使用 prefetch 在训练过程中可以并行执行数据的预获取。

	AUTOTUNE = tf.data.AUTOTUNE
	batch_size = 64
	seed = 110
	train_datas = tf.keras.utils.text_dataset_from_directory( 'aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)
	class_names = train_datas.class_names
	train_datas = train_datas.cache().prefetch(buffer_size=AUTOTUNE)
	val_datas = tf.keras.utils.text_dataset_from_directory(  'aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
	val_datas = val_datas.cache().prefetch(buffer_size=AUTOTUNE)
	test_datas = tf.keras.utils.text_dataset_from_directory( 'aclImdb/test', batch_size=batch_size)
	test_datas = test_datas.cache().prefetch(buffer_size=AUTOTUNE)

（6）随机取出两个处理好的样本进行展示：

	for text_batch, label_batch in train_datas.take(1):
	    for i in range(2):
	        print(f'Review: {text_batch.numpy()[i][:100]}...')
	        label = label_batch.numpy()[i]
	        print(f'Label : {label} ({class_names[label]})')

结果输出：

	Review: b"This 30 minute documentary Bu\xc3\xb1uel made in the early 1930's about one of Spain's poorest regions is,"...
	Label : 0 (neg)
	Review: b'I\'ve tried to watch this show several times, but for a show called "That \'70s Show," I don\'t find mu'...
	Label : 0 (neg)
    

# 4. 初识 TensorFlow Hub 中的 BERT 处理器和模型

（1）由于正规的从 TensorFlow Hub 下载模型需要“科学上网”，所以我们可以到这个镜像网站（https://hub.tensorflow.google.cn/google/collections/bert/1）中找适合我们的模型，这里提供了很多版本的 BERT 模型，为了方便我们快速学习，我们选用了比较小的 Small BERT ，及其对应的数据输入处理器。一般下载到本地的路径为 C:\Users\（用户名）\AppData\Local\Temp\tfhub\_modules\ 下面。

（2）preprocess 可以将文本转化成 BERT 所需要的输入，这样就免去了自己写 Python 代码来预处理文本来适应 BERT 模型的输入。这里会对文本处理产生对应的三个张量 input\_word\_ids、input\_type_ids、input\_mask ：

*  input\_word\_ids：一个 [batch\_size, 128] 的 int32 张量，每个张量包含了每句话中每个 token 对应的整数映射，并且包含了 START、END、PAD 对应的整数符号。如例子所见 how are you 对应的 input\_word\_ids 向量维度为 128 ， 101 对应 START ，102 对应 END ，中间的数字对应文本中的三个单词，其余的 0 对应 PAD 。 
*  input\_mask：一个 [batch\_size, 128] 的 int32 张量，PAD 之前的位置，也就是 START、END、以及 token 对应的整数的位置都是用 1 表示，填充 PAD 之后的位置都用 0 表示。如例子所见 how are you 对应的 input\_mask 向量维度都为 128 ，前 5 个位置都是 1 ，后面全是 0 。 
*  input\_type\_ids：一个 [batch\_size, 128] 的 int32 张量，如果输入是分段的，那么第一个输入段包括 START 和 END 的对应位置的都为 0 。如果存在第二段则包括 END 在内的输入都用 1 进行表示， 如果存在第三段则用 2 进行表示，也就是每一段都有一个不同的数字进行表示，剩下 PAD 填充的位置仍然用 0 表示。如例子所见 how are you 对应的 input\_type\_ids 向量维度为 128 ，前 5 个位置都是 0 ，因为没有第二段，所以后面都是 PAD 仍然用 0 表示。

（3）同样我们也使用了 small\_bert 接收 preprocess 处理之后的结果，这时我们可以产生四个对应的张量 pooled\_output、sequence\_output、default、encoder\_outputs ，这里我们主要用到前两个：

* pooled\_output：一个 [batch\_size, 512] 的 float32 张量，每个张量都是 512 维，表示将每个输入序列都编码为一个 512 维的表示向量。 
* sequence\_output：一个 [batch\_size, 128，512] 的 float32 张量，每个张量都是 [128, 512] 维，表示每个输入序列的每个 token 的编码结果输出是 512 维的表示。

处理器和模型获取：

	preprocess_url  = 'https://hub.tensorflow.google.cn/tensorflow/bert_en_uncased_preprocess/3'
	preprocess = hub.KerasLayer(preprocess_url)
	bert_url  = 'https://hub.tensorflow.google.cn/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/2'
	bert_model = hub.KerasLayer(bert_url)

处理器例子展示：

	text_test = ['how are you']
	preprocess_result = preprocess(text_test)
	
	print(f'keys           : {list(preprocess_result.keys())}')
	print(f'shape          : {preprocess_result["input_word_ids"].shape}')
	print(f'input_word_ids : {preprocess_result["input_word_ids"]}')
	print(f'input_mask     : {preprocess_result["input_mask"]}')
	print(f'input_type_ids : {preprocess_result["input_type_ids"]}')
	
输出：

	keys           : ['input_word_ids', 'input_type_ids', 'input_mask']
	shape          : (1, 128)
	input_word_ids : [[ 101 2129 2024 2017  102    0    0    0    0    0    0    0    0    0
	     0    0    0    0    0    0    0    0    0    0    0    0    0    0
	     0    0    0    0    0    0    0    0    0    0    0    0    0    0
	     0    0    0    0    0    0    0    0    0    0    0    0    0    0
	     0    0    0    0    0    0    0    0    0    0    0    0    0    0
	     0    0    0    0    0    0    0    0    0    0    0    0    0    0
	     0    0    0    0    0    0    0    0    0    0    0    0    0    0
	     0    0    0    0    0    0    0    0    0    0    0    0    0    0
	     0    0    0    0    0    0    0    0    0    0    0    0    0    0
	     0    0]]
	input_mask     : [[1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
	input_type_ids : [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
	  
模型例子展示：

	bert_results = bert_model(preprocess_result)
	
	print(f'Loaded BERT             : {bert_url}')
	print(f'Keys                    : {list(bert_results.keys())}')
	print(f'Pooled Outputs Shape    :{bert_results["pooled_output"].shape}')
	print(f'Sequence Outputs Values :{bert_results["pooled_output"].dtype}')
	print(f'Sequence Outputs Shape  :{bert_results["sequence_output"].shape}')
	print(f'Sequence Outputs Values :{bert_results["sequence_output"].dtype}')

输出：

	Loaded BERT             : https://hub.tensorflow.google.cn/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/2
	Keys                    : ['pooled_output', 'sequence_output', 'default', 'encoder_outputs']
	Pooled Outputs Shape    :(1, 512)
	Sequence Outputs Values :<dtype: 'float32'>
	Sequence Outputs Shape  :(1, 128, 512)
	Sequence Outputs Values :<dtype: 'float32'>	
	
# 5. 搭建模型

（1）第一层是输入层，用来接收用户输入的文本。

（2）第二层是我们上面已经介绍过得数据处理层，直接用从 TensorFlow Hub 下载的 bert\_en\_uncased\_preprocess 处理器即可。

（3）第三层是我们的 BERT 层，这里也是用我们上面介绍过得模型，直接使用从 TensorFlow Hub 下载的 bert\_en\_uncased\_L-8\_H-512\_A-8 模型即可。

（4）第四层是一个 Dropout 层，用来将 BERT 输出进行随机丢弃，避免过拟合。

（5）第五层一个输出 1 维向量的全连接层，其实就是输出该样本的分类 logit 。

	def create_model():
	    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
	    preprocessing_layer = hub.KerasLayer(preprocess, name='preprocessing')
	    encoder_inputs = preprocessing_layer(text_input)
	    encoder = hub.KerasLayer(bert_url, trainable=True, name='BERT_encoder')
	    outputs = encoder(encoder_inputs)
	    net = outputs['pooled_output']
	    net = tf.keras.layers.Dropout(0.1)(net)
	    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
	    return tf.keras.Model(text_input, net)
	model = create_model()
	
# 6. 训练模型

（1）由于这是一个二元分类问题，并且模型最终输出的是概率，因此我们选择 BinaryCrossentropy 作为损失函数。使用 BinaryAccuracy 作为我们的评估指标，在进行预测的时候模型输出概率大于 threshold 的预测为 1 也就是积极情绪的，小于等于 threshold 的预测为 0 ，也就是消极的，threshold 默认是 0.5 。

（2）为了进行微调，我们使用 BERT 最初训练时用的的优化器：Adam 。该优化器最大程度减少预测损失，并通过权重衰减进行正则化，所以它也被称为 AdamW 。

（3）我们使用与 BERT 预训练相同的学习率（也就是我们的 init\_lr 变量），训练刚开始时，采用较小的学习率，随着迭代次数增加学习率线性增大，当迭代步达到 num\_warmup\_steps 时，学习率设置为为初始设定的学习率 init\_lr ，然后学习率随着迭代次数逐步衰减。BERT 论文中将用于微调的初始学习率设置较小,如:5e-5，3e-5，2e-5 。

（4）为什么使用 adamw 优化器 ?由于刚开始训练时,模型的权重是随机初始化的，此时若选择一个较大的学习率,可能带来模型优化的不稳定(振荡)，选择 AdamW 优化器，可以使得开始训练的若干 epoches 或者 steps 内学习率较小,在预热的小学习率下，模型可以慢慢趋于稳定,等模型相对稳定后再选择预先设置的学习率进行训练（此后的学习率是衰减的），有助于使模型收敛速度变快，效果更佳。

	print(f'Training model with {bert_url}')
	epochs = 5
	steps_per_epoch = tf.data.experimental.cardinality(train_datas).numpy()
	num_train_steps = steps_per_epoch * epochs
	num_warmup_steps = int(0.1*num_train_steps)
	optimizer = optimization.create_optimizer(init_lr=3e-5,  num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, optimizer_type='adamw')
	model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=tf.metrics.BinaryAccuracy())
	history = model.fit(x=train_datas, validation_data=val_datas, epochs=epochs)
	
训练过程，可以看出相当耗时，这也是使用 BERT 的一个明显缺点：

	Training model with https://hub.tensorflow.google.cn/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/2
	Epoch 1/5
	313/313 [==============================] - 3433s 11s/step - loss: 0.4705 - binary_accuracy: 0.7515 - val_loss: 0.3789 - val_binary_accuracy: 0.8124
	Epoch 2/5
	313/313 [==============================] - 3328s 11s/step - loss: 0.3043 - binary_accuracy: 0.8653 - val_loss: 0.3734 - val_binary_accuracy: 0.8450
	Epoch 3/5
	313/313 [==============================] - 3293s 11s/step - loss: 0.2301 - binary_accuracy: 0.9024 - val_loss: 0.4295 - val_binary_accuracy: 0.8532
	Epoch 4/5
	313/313 [==============================] - 3289s 11s/step - loss: 0.1697 - binary_accuracy: 0.9344 - val_loss: 0.4831 - val_binary_accuracy: 0.8492
	Epoch 5/5
	313/313 [==============================] - 3411s 11s/step - loss: 0.1308 - binary_accuracy: 0.9497 - val_loss: 0.4631 - val_binary_accuracy: 0.8538

# 7. 测试模型
使用测试数据对训练好的模型进行评估，可以看到准确率达到了  0.8630 ，如果给予充足的调参和训练时间，效果会更好。

	model.evaluate(test_datas)
	
输出：

	391/391 [==============================] - 1153s 3s/step - loss: 0.4290 - binary_accuracy: 0.8630	
# 8. 保存模型

将训练好的模型保存到本地，以后可以随时读取模型进行预测工作。

	dataset_name = 'imdb'
	saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))
	model.save(saved_model_path, include_optimizer=False)

# 9. 重新加载模型并进行预测

我们将使用上面已经存在的模型 model 和刚才重新加载的模型 reloaded\_model 进行预测，将一个积极情绪样本和一个消极情绪样本输入模型，发现能够预测正确（接近），而且两个模型的结果是一样的。

	def print_my_examples(inputs, results):
	    result_for_printing =  [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}' for i in range(len(inputs))]
	    print(*result_for_printing, sep='\n')
	    
	examples = ['The movie was great!', 'The movie was terrible...']
	
	reloaded_model = tf.saved_model.load(saved_model_path)
	reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
	original_results = tf.sigmoid(model(tf.constant(examples)))
	
	print('Results from reloaded_model:')
	print_my_examples(examples, reloaded_results)
	print('Results from model:')
	print_my_examples(examples, original_results)
	
结果输出：

	Results from reloaded_model:
	input: The movie was great!           : score: 0.994967
	input: The movie was terrible...      : score: 0.002266
	Results from model:
	input: The movie was great!           : score: 0.994967
	input: The movie was terrible...      : score: 0.002266	