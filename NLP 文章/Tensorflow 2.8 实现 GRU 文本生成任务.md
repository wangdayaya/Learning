# 前言

本文使用 cpu 的 tensorflow 2.8 来完成 GRU 文本生成任务。

如果想要了解文本生成的相关概念，可以参考我之前写的文章：https://juejin.cn/post/6973567782113771551

# 大纲

1. 获取数据
2. 处理数据
3. 搭建并训练模型
4. 生成文本逻辑
5. 预测
6. 保存和读取模型

# 实现

### 1. 获取数据

（1）我们使用到的数据是莎士比亚的作品，我们使用 TensorFlow 的内置函数从网络下载到本地的磁盘，我们展现了部分内容，可以看到里面都是一段一段对话形式的台词。

（2）通过使用集合找出数据中总共出现了 65 个不同的字符。

	import tensorflow as tf
	import numpy as np
	import os
	import time
	path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
	text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
	vocab = sorted(set(text))
	print(text[:100])
	print(f'{len(vocab)} unique characters')
	
结果输出：

	First Citizen:
	Before we proceed any further, hear me speak.
	
	All:
	Speak, speak.
	
	First Citizen:
	You
	65 unique characters	
### 2. 处理数据

（1）在使用数据的时候我们需要将所有的字符都映射成对应的数字， StringLookup 这个内置函数刚好可以实现这个功能，使用这个函数之前要将文本都切分成字符。另外我们还可以使用 StringLookup 这个内置函数完成从数字到字符的映射转换。我们自定义了函数 text\_from\_ids 可以实现将字符的序列还原回原始的文本。

（2）我们将莎士比亚数据中的文本使用 ids\_from\_chars 全部转换为整数序列，然后使用 from\_tensor\_slices 创建 Dataset 对象。

（3）我们将数据都切分层每个 batch 大小为 seq\_length+1 的长度，这样是为了后面创建（input，target）这一样本形式的。每个样本 sample 的 input 序列选取文本中的前 seq\_length 个字符 sample[:seq\_length] 为输入。对于每个 input ，相应的 target 也包含相同长度的文本，只是整体向右移动了一个字符，选取结果为 sample[1:seq\_length+1]。例如 seq\_length 是 4，我们的序列是“Hello”,那么 input 序列为“hell”，目标序列为“ello”。

（4）我们展示了一个样本，可以看到 input 和 label 的形成遵循上面的规则，其目的就是要让 RNN 的每个时间步上都有对应的输入字符和对应的目标字母，输入字符是当前的字符，目标字符肯定就是后面一个相邻的字符。

	ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
	chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
	def text_from_ids(ids):
	    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
	all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
	ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
	seq_length = 64
	sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
	def split_input_target(sequence):
	    input_text = sequence[:-1]
	    target_text = sequence[1:]
	    return input_text, target_text
	dataset = sequences.map(split_input_target)
	
	for input_example, target_example in dataset.take(1):
	    print("Input :", text_from_ids(input_example).numpy())
	    print("Label:", text_from_ids(target_example).numpy())
	    
结果输出：

	Input : b'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAl'
	Label: b'irst Citizen:\nBefore we proceed any further, hear me speak.\n\nAll' 
	
（5）我们将所有处理好的样本先进行混洗，保证样本的随机性，然后将样本都进行分批，每个 batch 设置大小为 64 ，设置每个词嵌入维度为 128 ，设置 GRU 的输入为 128 维。   

	BATCH_SIZE = 64
	BUFFER_SIZE = 10000
	vocab_size = len(ids_from_chars.get_vocabulary())
	embedding_dim = 128
	gru_units = 128
	dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
### 3. 搭建并训练模型

（1）第一层是词嵌入层，主要是将用户输入的序列中的每个证书转换为模型需要的多维输入。

（2）第二层是 GRU 层，主要是接收每个时间步的输入，并且将前后状态进行计算和保存，让 GRU 可以记住文本序列规律。

（3）第三层是全连接层，主要是输出一个字典大小维度的向量，表示的是每个字符对应的概率分布。

（4）这里有一些细节需要处理，如果 states 是空，那么就直接随机初始化 gru 的初始状态，另外如果需要返回 states 结果，那么就将全连接层的输出和 states 一起返回。

	class MyModel(tf.keras.Model):
	    def __init__(self, vocab_size, embedding_dim, gru_units):
	        super().__init__(self)
	        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
	        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=True,  return_state=True)
	        self.dense = tf.keras.layers.Dense(vocab_size)
	
	    def call(self, inputs, states=None, return_state=False, training=False):
	        x = inputs
	        x = self.embedding(x, training=training)
	        if states is None:
	            states = self.gru.get_initial_state(x)
	        x, states = self.gru(x, initial_state=states, training=training)
	        x = self.dense(x, training=training)
	
	        if return_state:
	            return x, states
	        else:
	            return x
	model = MyModel( vocab_size=vocab_size, embedding_dim=embedding_dim,  gru_units=gru_units)
	
（5）我们随机选取一个样本，输入到还没有训练的模型中，然后进行文本生成预测，可以看出目前的输出毫无规。
	
	for one_input, one_target in dataset.take(1):
	    one_predictions = model(one_input)
	    print(one_predictions.shape, "--> (batch_size, sequence_length, vocab_size)")
	sampled_indices = tf.random.categorical(one_predictions[0], num_samples=1)
	sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
	print("Input:\n", text_from_ids(one_input[0]).numpy())
	print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())
	
结果输出：

	(64, 64, 66) --> (batch_size, sequence_length, vocab_size)
	Input:
	 b'\nBut let thy spiders, that suck up thy venom,\nAnd heavy-gaited t'
	Next Char Predictions:
	 b'ubH-I\nBxZReX!n\n$VBgkBqQxQEVaQ!-Siw uHoTaX!YT;vFYX,r:aLh h$fNRlEN'	
	 
 （6）这里主要是选择损失函数和优化器，我们选取 SparseCategoricalCrossentropy 来作为损失函数，选取 Adam 作为优化器。

（7）我这里还定义了一个回调函数，在每次 epoch 结束的时候，我们保存一次模型，总共执行 20 个 epoch 。

	loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
	model.compile(optimizer='adam', loss=loss)
	checkpoint_dir = './my_training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_prefix, save_weights_only=True)
	EPOCHS=20
	history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

结果输出：

	Epoch 1/20
	268/268 [==============================] - 12s 39ms/step - loss: 2.7113
	Epoch 2/20
	268/268 [==============================] - 11s 39ms/step - loss: 2.1106
	...
	Epoch 19/20
	268/268 [==============================] - 11s 40ms/step - loss: 1.4723
	Epoch 20/20
	268/268 [==============================] - 11s 38ms/step - loss: 1.4668
	
### 4. 生成文本逻辑

（1）这里为我们主要是定义了一个类，可以使用已经训练好的模型进行文本生成的任务，在初始化的时候我们需要将字符到数字的映射 chars\_from\_ids，以及数字到字符的映射 ids\_from\_chars 都进行传入。

（2）这里需要注意的是我们新增了一个 prediction\_mask ，最后将其与模型输出的 predicted\_logits 进行相加，其实就是将 [UNK] 对应概率降到无限小，这样就不会在采样的时候采集 [UNK] 。

（3）在进行预测时候我们只要把每个序列上的最后一个时间步的输出拿到即可，这其实就是所有字符对应的概率分布，我们只需要通过 categorical 函数进行随机采样，概率越大的字符被采集到的可能性越大。

	class OneStep(tf.keras.Model):
	    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
	        super().__init__()
	        self.temperature = temperature
	        self.model = model
	        self.chars_from_ids = chars_from_ids
	        self.ids_from_chars = ids_from_chars
	
	        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
	        sparse_mask = tf.SparseTensor( values=[-float('inf')]*len(skip_ids), indices=skip_ids, dense_shape=[len(ids_from_chars.get_vocabulary())])
	        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
	
	    @tf.function
	    def generate_one_step(self, inputs, states=None):
	        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
	        input_ids = self.ids_from_chars(input_chars).to_tensor()
	        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
	        predicted_logits = predicted_logits[:, -1, :]
	        predicted_logits = predicted_logits/self.temperature
	        predicted_logits = predicted_logits + self.prediction_mask
	        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
	        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
	        predicted_chars = self.chars_from_ids(predicted_ids)
	        return predicted_chars, states
	    
	one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
	
	
### 5. 预测

（1）我们可以对一个样本进行文本生成预测，也可以对批量的样本进行文本预测工作。下面分别使用例子进行了效果展示。

（2）我们可以发现，在不仔细检查的情况下，模型生成的文本在格式上和原作是类似的，而且也形成了“单词”和“句子”，尽管有的根本压根就不符合语法，想要增强效果的最简单方法就是增大模型的（尤其是 GRU）的神经元个数，或者增加训练的 epoch 次数。

	states = None
	next_char = tf.constant(['First Citizen:'])
	result = [next_char]
	for n in range(300):
	    next_char, states = one_step_model.generate_one_step(next_char, states=states)
	    result.append(next_char)
	result = tf.strings.join(result)
	print(result[0].numpy().decode('utf-8'))
	
结果输出：

	First Citizen: I kome flower as murtelys bease her sovereign!
	
	DUKE VINCENTIO:
	More life, I say your pioused in joid thune:
	I am crebles holy for lien'd; she will. If helps an Gaod questilford
	And reive my hearted
	At you be so but to-deaks' BAPtickly Romeo, myself then saddens my wiflious wine creple.
	Now if you 	
	
进行批量预测：

	states = None
	next_char = tf.constant(['First Citizen:', 'Second Citizen:', 'Third Citizen:'])
	result = [next_char]
	
	for n in range(300):
	    next_char, states = one_step_model.generate_one_step(next_char, states=states)
	    result.append(next_char)
	
	result = tf.strings.join(result)
	end = time.time()
	print(result)	
	
结果：

	tf.Tensor(
	[b"First Citizen: stors, not not-became mother, you reachtrall eight.\n\nBUCKINGHAM:\nI net\nShmo'ens from him thy haplay. So ready,\nCantent'd should now to thy keep upon thy king.\nWhat shall play you just-my mountake\nPanch his lord, ey? Of thou!\n\nDUKE VINCENTIO:\nThus vilided,\nSome side of this? I though he is heart the"
	 b"Second Citizen:\nThen I'll were her thee exceacies even you laggined.\n\nHENRY BOLINGBROKE:\nMet like her safe.\n\nGLOUCESTER:\nSoet a spired\nThat withal?\n\nJULIET,\nA rable senul' thmest thou wilt the saper and a Came; or like a face shout thy worsh is tortument we shyaven?\nLet it take your at swails,\nAnd will cosoprorate"
	 b'Third Citizen:\nDishall your wife, is thus?\n\nQUEEN ELIZABETH:\nNo morrot\nAny bring it bedies did be got have it,\nPervart put two food the gums: and my monst her,\nYou complike your noble lies. An must against man\nDreaming times on you.\nIt were you. I was charm on the contires in breath\nAs turning: gay, sir, Margaret'], shape=(3,), dtype=string)	
### 6. 保存和读取模型

我们对模型的权重进行保存，方便下次调用。

	tf.saved_model.save(one_step_model, 'one_step')
	one_step_reloaded = tf.saved_model.load('one_step')
	
使用加载的模型进行文本生成预测。

	states = None
	next_char = tf.constant(['First Citizen:'])
	result = [next_char]
	
	for n in range(300):
	    next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
	    result.append(next_char)
	
	print(tf.strings.join(result)[0].numpy().decode("utf-8"))	
结果输出：

	First Citizen:
	Let me shet
	Of mortal prince! BJuiting late and fublings.
	Art like could not, thou quiclay of all that changes
	Whose himit offent and montagueing: therefore, and their ledion:
	Proceed thank you; and never night.
	
	GRUMIO:
	Nell hath us to the friend'st though, sighness?
	
	GLOUCESSE:
	How'd hang
	A littl	