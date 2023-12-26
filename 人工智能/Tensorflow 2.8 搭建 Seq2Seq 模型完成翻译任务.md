***本文正在参加[「金石计划 . 瓜分6万现金大奖」](https://juejin.cn/post/7162096952883019783 "https://juejin.cn/post/7162096952883019783")***

# 前言


本文使用 cpu 版本的 TensorFlow 2.8 版本完成西班牙文到英文的翻译任务，我们假定读者已经熟悉了 Seq2Seq 的模型，如果还不了解可以移步看我之前写的文章，或者看相关论文：

* 《Effective Approaches to Attention-based Neural Machine Translation》 (Luong et al., 2015)
*    https://juejin.cn/post/6973930281728213006

建议安装好相关的工具包：

	tensorflow-text==2.10
	einops

# 大纲

1. 获取数据
2. 处理数据
3. 搭建 Encoder 
4. 搭建 Attention
5. 搭建 Decoder 
6. 搭建完整的 Translator 模型
7. 编译、训练
8. 推理
9. 保存和加载模型


# 实现

### 1. 获取数据

（1）本文使用了一份西班牙文转英文的数据，每一行都是一个样本，每个样本有一个西班牙文和对应的英文翻译，两者中间由一个水平制表符进行分割。

（2）我们可以使用 TensorFlow 的内置函数来从网络上下载本文所用到的数据到本地，一般会下载到本地的 C:\Users\《用户名》.keras\datasets\spa-eng.zip 路径下面。

（3）我们首使用 utf-8 的编码格式读取磁盘中的文件到内存，然后将每一行的样本用水平制表符切割，将西班牙文作为我们的输入，将英文作为我们的输出目标。

（4）随机选取 80% 的数据作为我们的训练集，剩下的 20% 的数据当做验证集。

（5）我们随机选取了一个样本的输入和目标进行显示，可以看到在模型的 Encoder 部分使用的是西班牙文，在模型的 Decoder 部分使用的是英文。

	import pathlib
	import numpy as np
	import typing
	from typing import Any, Tuple
	import einops
	import matplotlib.pyplot as plt
	import matplotlib.ticker as ticker
	import tensorflow as tf
	import tensorflow_text as tf_text
	path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True)
	path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'
	BUFFER_SIZE = len(context_raw)
	BATCH_SIZE = 128
	def load_data(path):
	    text = path.read_text(encoding='utf-8')
	    lines = text.splitlines()
	    pairs = [line.split('\t') for line in lines]
	    context = np.array([context for target, context in pairs])
	    target = np.array([target for target, context in pairs])
	    return target, context
	target_raw, context_raw = load_data(path_to_file)
	is_train = np.random.uniform(size=(len(target_raw),)) < 0.8
	train_raw = ( tf.data.Dataset.from_tensor_slices((context_raw[is_train], target_raw[is_train])).shuffle(BUFFER_SIZE).batch(BATCH_SIZE))
	val_raw = ( tf.data.Dataset.from_tensor_slices((context_raw[~is_train], target_raw[~is_train])).shuffle(BUFFER_SIZE).batch(BATCH_SIZE))
	for example_context, example_target in train_raw.take(1):
	    for a,b in zip(example_context[:1], example_target[:1]):
	        print(a)
	        print(b)
	    break

样本结果打印：

	tf.Tensor(b'A los gatos no les gusta estar mojados.', shape=(), dtype=string)
	tf.Tensor(b'Cats dislike being wet.', shape=(), dtype=string)

### 2. 处理数据

（1）因为使用到不同的语言不同，可能涉及到不同的编码问题，所以我们要使用 TensorFlow 内置的函数，将输入和目标的所有文本都进行编码的标准化，统一使用 utf-8 ，并且将文本中除了字母、空格、以及若干个指定的标点符号之外的字符都进行剔除，并且在输入和目标文本的开始和末端加入 [START] 和 [END] 来表示句子的开始和结束。

（2）因为输入和目标需要维护不同的词典，所以我们对 token 进行整数映射的时候要维护一个输入词典和一个目标词典，并且两个词典都要有 token -> int 和 int -> token 的映射。

	def standardize(text):
	    text = tf_text.normalize_utf8(text, 'NFKD')
	    text = tf.strings.lower(text)
	    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
	    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
	    text = tf.strings.strip(text)
	    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
	    return text
	max_vocab_size = 6000
	context_processor = tf.keras.layers.TextVectorization(standardize=standardize, max_tokens=max_vocab_size, ragged=True)
	context_processor.adapt(train_raw.map(lambda context, target: context))
	target_processor = tf.keras.layers.TextVectorization( standardize=standardize,  max_tokens=max_vocab_size, ragged=True)
	target_processor.adapt(train_raw.map(lambda context, target: target))
	print(context_processor.get_vocabulary()[:10])
	print(target_processor.get_vocabulary()[:10])
	
两个词典中的部分 token 展示：

	['', '[UNK]', '[START]', '[END]', '.', 'que', 'de', 'el', 'a', 'no']
	['', '[UNK]', '[START]', '[END]', '.', 'the', 'i', 'to', 'you', 'tom']	
（3）这里主要是随机选取一个样本，我们将西班牙文中的 token 进行整数映射，然后再转回西班牙文，很明显我们看到，在句子的开头和末尾加入了 [START] 、[END] ，而且如果在词典中不存在的 token 直接表示为了 [UNK] 。英文的转换过程也是如此。

	def context_target_example(s, example, processor):
	    print(s, ":")
	    tokens = processor(example)
	    print(tokens[:1, :])
	    vocab = np.array(processor.get_vocabulary())
	    result = vocab[tokens[0].numpy()]
	    print(' '.join(result))
	    
	context_target_example('context', example_context ,context_processor )
	context_target_example('target', example_target ,target_processor )
	
结果打印：

	context :
	<tf.RaggedTensor [[2, 8, 26, 646, 9, 204, 63, 117, 1, 4, 3]]>
	[START] a los gatos no les gusta estar [UNK] . [END]
	target :
	<tf.RaggedTensor [[2, 677, 2399, 286, 1329, 4, 3]]>
	[START] cats dislike being wet . [END]	
	
（4）由于在 Seq2Seq 模型在 Decoder 部分，需要将目标进行预测，所以我们要对目标数据进行处理，使得每个时间步都有输入和输出，输入当然就是其本身，而输入就是相邻的下一个 token 。我们在处理目标数据的时候，要将 target 数据本身当做 Decoder 输入，然后整体将 target 右移一位当做 Decoder 输出。

（5）我们随机抽取了一个样本，这里展现了模型的 Encoder 输入，Decoder 输入以及 Docoder 输出。我们可以看出在使用了 to\_tensor 函数之后发生了填充操作，该 batch 中所有的 context 的长度都会变成该 batch 中出现的最长序列的那个 context 的长度，需要填充的位置都用 0 表示。同样的道理该 batch 中的 target\_in、target\_out 的长度也都会发生同样的填充变化。换句话说不同的 batch 中 Encoder 都不相等，不同的 batch 中 Decoder 的长度都不相等，唯一能保证相等的是同一个 batch 中的 Decoder 端的 target\_in 和 target\_out 长度相等。	

	def process_text(context, target):
	    context = context_processor(context).to_tensor()
	    target = target_processor(target)
	    target_in = target[:,:-1].to_tensor()
	    target_out = target[:,1:].to_tensor()
	    return (context, target_in), target_out
	
	train_datas = train_raw.map(process_text, tf.data.AUTOTUNE)
	val_datas = val_raw.map(process_text, tf.data.AUTOTUNE)
	for (one_context, one_target_in), one_target_out in train_datas.take(1):
	    print("one_context:")
	    print(one_context[:2, :].numpy()) 
	    print("one_target_in:")
	    print(one_target_in[:2, :].numpy()) 
	    print("one_target_out:")
	    print(one_target_out[:2, :].numpy())

文本处理后转化为 token 的样例：
    
	one_context:
	[[   2   13  627  616   14   20  610    9  605  134   12    3    0    0
	     0    0    0    0]
	 [   2   18   75  894    6   23  595    5   18 4656    4    3    0    0
	     0    0    0    0]]
	one_target_in:
	[[  2  58 128  15   5 698  34  22 989  20   8  38  43  11]
	 [  2  85 275  42  23  14  10 177  16   1  23   4   0   0]]
	one_target_out:
	[[ 58 128  15   5 698  34  22 989  20   8  38  43  11   3]
	 [ 85 275  42  23  14  10 177  16   1  23   4   3   0   0]]
     
### 3. 搭建 Encoder 

（1）第一层是文本处理器，要使用 context\_processor 将文本进行预处理，并将 token 映射成整数 id

（2）第二层是嵌入层，要将每个整数 id 都映射成一个 128 维的向量。

（3）第三层是双向 GRU 层，主要是捕获输入的西班牙文的文本特征，并且在序列的每个时间步都输出一个 128 维的向量

（4）我们用上面用过的例子 one\_context 输入到模型的 Encoder 中，可以看到输入的大小为 [batch\_size, source\_seq\_length] ，也就是说该 batch 中有 batch\_size 个样本，每个输入样本长度为 source\_seq\_length ，输出的大小为 [batch\_size, source\_seq\_length, units]，也就是说该 batch 中有 batch\_size 个样本，每个输入样本的长度为 source\_seq\_length ，序列中每个时间步的输出结果是 units 维。

	UNITS = 128
	class Encoder(tf.keras.layers.Layer):
	    def __init__(self, text_processor, units):
	        super(Encoder, self).__init__()
	        self.text_processor = text_processor
	        self.vocab_size = text_processor.vocabulary_size()
	        self.units = units
	        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
	        self.rnn = tf.keras.layers.Bidirectional( merge_mode='sum', layer=tf.keras.layers.GRU(units, return_sequences=True, recurrent_initializer='glorot_uniform'))
	
	    def call(self, x):
	        x = self.embedding(x)
	        x = self.rnn(x)
	        return x
	
	    def convert_input(self, texts):
	        texts = tf.convert_to_tensor(texts)
	        if len(texts.shape) == 0:
	            texts = tf.convert_to_tensor(texts)[tf.newaxis]
	        context = self.text_processor(texts).to_tensor()
	        context = self(context)
	        return context
	
	encoder = Encoder(context_processor, UNITS)
	example_context_output = encoder(one_context)
	print(f'Context tokens, shape (batch_size, source_seq_length)       : {one_context.shape}')
	print(f'Encoder output, shape (batch_size, source_seq_length, units): {example_context_output.shape}')

结果输出：

	Context tokens, shape (batch_size, source_seq_length)       : (128, 18)
	Encoder output, shape (batch_size, source_seq_length, units): (128, 18, 128)

### 4. 搭建 Attention

（1）Attention 层允许 Decoder 访问 Encoder 提取的输入文本的特征信息，Attention 层会以 Decoder 输出为 query ，以 Encoder 输出为 key 和 value ，计算 Decoder 输出与 Encoder 输出的不同位置的有关重要程度，并将其加到 Decoder 的输出中。

（2）我这里选用了上面的例子 one\_target\_in ，假如它只经过了词嵌入，然后输出词嵌入的结果，我们目前认为这就是经过 Decoder 解码的输出 example\_target\_embed ，我们计算这个 example\_target\_embed 和之前计算出来的对应的 Encoder 的输出 example\_context\_output 的注意力结果向量，我们可以发现 Decoder 的输入是 [batch\_size, target\_seq\_length, units] ，表示 Decoder 的输入的 batch 有 batch\_size 个样本，每个样本长度为 target\_seq\_length，Decoder 的每个时间步上输出的维度为 units 。 Attention 的结果和 Decoder 的向量维度是一样的，这保证了 Attention 结果可以和 Decoder 输出结果可以相加。Attention 的权重大小是 [batch\_size, target\_seq\_length, source\_seq\_length] ，表示该 batch 中有 batch\_size 个样本，每个样本的 Attention 的大小是 [target\_seq\_length, source\_seq\_length] ，表示计算出的每个样本的 Decoder 输出和 Encoder 输出的所有一一对应的位置的有关重要程度。

	class CrossAttention(tf.keras.layers.Layer):
	    def __init__(self, units, **kwargs):
	        super().__init__()
	        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
	        self.layernorm = tf.keras.layers.LayerNormalization()
	        self.add = tf.keras.layers.Add()
	
	    def call(self, x, context):
	        attn_output, attn_scores = self.mha( query=x,  value=context, return_attention_scores=True)
	        attn_scores = tf.reduce_mean(attn_scores, axis=1)
	        self.last_attention_weights = attn_scores
	        x = self.add([x, attn_output])
	        x = self.layernorm(x)
	        return x
	    
	attention_layer = CrossAttention(UNITS)
	embed = tf.keras.layers.Embedding(target_processor.vocabulary_size(),  output_dim=UNITS, mask_zero=True)
	example_target_embed = embed(one_target_in)
	result = attention_layer(example_target_embed, example_context_output)
	
	print(f'Context sequence, shape (batch_size, source_seq_length, units)             : {example_context_output.shape}')
	print(f'Target sequence, shape (batch_size, target_seq_length, units)              : {example_target_embed.shape}')
	print(f'Attention result, shape (batch_size, target_seq_length, units)             : {result.shape}')
	print(f'Attention weights, shape (batch_size, target_seq_length, source_seq_length): {attention_layer.last_attention_weights.shape}')

结果打印：

	Context sequence, shape (batch_size, source_seq_length, units)             : (128, 18, 128)
	Target sequence, shape (batch_size, target_seq_length, units)              : (128, 14, 128)
	Attention result, shape (batch_size, target_seq_length, units)             : (128, 14, 128)
	Attention weights, shape (batch_size, target_seq_length, source_seq_length): (128, 14, 18)

### 5. 搭建 Decoder 

（1）第一层是文本处理器，使用 target_processor 将文本进行预处理，并将 token 映射成整数 id 。

（2）第二层是嵌入层，要将每个整数 id 都映射成一个 128 维的向量。

（3）第三层是一个单向 GRU 层，因为这里是要从左到右进行的解码工作，所以只能是一个从左到右的单向 GRU 层，主要是捕获输入的英文的文本特征，并且在序列的每个时间步都输出一个 128 维的向量。

（4）第四层是一个 Attention 层，这里主要是计算第三层的输出和 Encoder 的输出的注意力结果，并将其和第三层的输出进行相加。

（5）第五层是一个全连接层，Decoder 的每个输出位置都有一个词典大小的向量，表示每个位置预测下一个单词的概率分布。

（6）我们使用上面的例子产生的 Encoder 输出 example\_context\_output 和 Decoder 输入 one\_target\_in ，经过 Decoder 中间的计算过程，我们可以发现最终的输出结果大小是 [batch\_size, target\_seq\_length, target\_vocabulary\_size] 。表示输出有 batch\_size 个样本结果，每个样本的序列长度为 target\_seq\_length ，序列的每个位置上有一个词典大小为 target\_vocabulary\_size 的概率分布。

（7）在训练时，模型会预测每个位置的目标单词，每个位置的输出预测结果都是没有交互、独立的，所以 Decoder 使用单向 GRU 来处理目标序列。但是使用模型进行推理时，每个位置生成一个单词，并还要将此预测的单词继续反馈到模型的下一个位置中当作一部分输入，进行下一个位置的预测。

	class Decoder(tf.keras.layers.Layer):
	    @classmethod
	    def add_method(cls, fun):
	        setattr(cls, fun.__name__, fun)
	        return fun
	
	    def __init__(self, text_processor, units):
	        super(Decoder, self).__init__()
	        self.text_processor = text_processor
	        self.vocab_size = text_processor.vocabulary_size()
	        self.word_to_id = tf.keras.layers.StringLookup( vocabulary=text_processor.get_vocabulary(),  mask_token='', oov_token='[UNK]')
	        self.id_to_word = tf.keras.layers.StringLookup( vocabulary=text_processor.get_vocabulary(),  mask_token='', oov_token='[UNK]', invert=True)
	        self.start_token = self.word_to_id('[START]')
	        self.end_token = self.word_to_id('[END]')
	
	        self.units = units
	        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
	        self.rnn = tf.keras.layers.GRU(units, return_sequences=True, return_state=True,  recurrent_initializer='glorot_uniform')
	        self.attention = CrossAttention(units)
	        self.output_layer = tf.keras.layers.Dense(self.vocab_size)
	        
	    def call(self, context, x, state=None, return_state=False):  
	        x = self.embedding(x)
	        x, state = self.rnn(x, initial_state=state)
	        x = self.attention(x, context)
	        self.last_attention_weights = self.attention.last_attention_weights
	        logits = self.output_layer(x)
	        if return_state:
	            return logits, state
	        else:
	            return logits
	
	decoder = Decoder(target_processor, UNITS)
	logits = decoder(example_context_output, one_target_in)
	
	print(f'encoder output shape: (batch_size, source_seq_length, units) {example_context_output.shape}')
	print(f'input target tokens shape: (batch_size, target_seq_length) {one_target_in.shape}')
	print(f'logits shape shape: (batch_size, target_seq_length, target_vocabulary_size) {logits.shape}')

结果打印：

	encoder output shape: (batch_size, source_seq_length, units) (128, 18, 128)
	input target tokens shape: (batch_size, target_seq_length) (128, 14)
	logits shape shape: (batch_size, target_seq_length, target_vocabulary_size) (128, 14, 6000)

（8）Decoder 在推理需要的其他需要的函数。

	@Decoder.add_method
	def get_initial_state(self, context):
	    batch_size = tf.shape(context)[0]
	    start_tokens = tf.fill([batch_size, 1], self.start_token)
	    done = tf.zeros([batch_size, 1], dtype=tf.bool)
	    embedded = self.embedding(start_tokens)
	    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]
	
	@Decoder.add_method
	def tokens_to_text(self, tokens):
	    words = self.id_to_word(tokens)
	    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
	    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
	    result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
	    return result
	
	@Decoder.add_method
	def get_next_token(self, context, next_token, done, state, temperature = 0.0):
	    logits, state = self( context, next_token, state = state, return_state=True) 
	    if temperature == 0.0:
	        next_token = tf.argmax(logits, axis=-1)
	    else:
	        logits = logits[:, -1, :]/temperature
	        next_token = tf.random.categorical(logits, num_samples=1)
	    done = done | (next_token == self.end_token)
	    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)
	    return next_token, done, state

### 6. 搭建完整的 Translator 模型

（1）这里创建翻译类 Translator ，相当于一个完成 Seq2Seq 的模型，包含了一个 Encoder 和一个 Decoder 。

（2）使用之前用到的测试样本，我们发现 Translator 返回的结果就是 Decoder 部分返回的结果。

	class Translator(tf.keras.Model):
	    @classmethod
	    def add_method(cls, fun):
	        setattr(cls, fun.__name__, fun)
	        return fun
	
	    def __init__(self, units,  context_processor, target_processor):
	        super().__init__()
	        encoder = Encoder(context_processor, units)
	        decoder = Decoder(target_processor, units)
	        self.encoder = encoder
	        self.decoder = decoder
	
	    def call(self, inputs):
	        context, x = inputs
	        context = self.encoder(context)
	        logits = self.decoder(context, x)
	        try:
	            del logits._keras_mask
	        except AttributeError:
	            pass
	
	        return logits
	
	model = Translator(UNITS, context_processor, target_processor)
	logits = model((one_context, one_target_in))
	
	print(f'Context tokens, shape: (batch_size, source_seq_length) {one_context.shape}')
	print(f'Target tokens, shape: (batch_size, target_seq_length) {one_target_in.shape}')
	print(f'logits, shape: (batch_size, target_seq_length, target_vocabulary_size) {logits.shape}')

结果打印：

	Context tokens, shape: (batch_size, source_seq_length) (128, 18)
	Target tokens, shape: (batch_size, target_seq_length) (128, 14)
	logits, shape: (batch_size, target_seq_length, target_vocabulary_size) (128, 14, 6000)


### 7. 编译、训练

（1）我们在这里定义了模型使用了掩码的损失函数的计算方法和准确率的计算方法，我们还选择了 Adam 作为我们的优化器。

（2）在训练过程中我们使用训练数据训练 100 个 epoch ，每个 epoch 训练 100 个 batch ，并且在每个 epoch 训练结束后使用验证集进行评估，在第 24 个 epoch 的时候发生了 EarlyStopping 。

	def masked_loss(y_true, y_pred):
	    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=True, reduction='none')
	    loss = loss_fn(y_true, y_pred)
	    mask = tf.cast(y_true != 0, loss.dtype)
	    loss *= mask
	    return tf.reduce_sum(loss)/tf.reduce_sum(mask)
	
	def masked_acc(y_true, y_pred):
	    y_pred = tf.argmax(y_pred, axis=-1)
	    y_pred = tf.cast(y_pred, y_true.dtype)
	    match = tf.cast(y_true == y_pred, tf.float32)
	    mask = tf.cast(y_true != 0, tf.float32)
	    return tf.reduce_sum(match)/tf.reduce_sum(mask)
	
	model.compile(optimizer='adam', loss=masked_loss,   metrics=[masked_acc, masked_loss])
	history = model.fit( train_datas.repeat(),  epochs=100, steps_per_epoch = 100,  validation_data=val_datas, \
	                    validation_steps = 20, callbacks=[ tf.keras.callbacks.EarlyStopping(patience=3)])

训练过程：

	Epoch 1/100
	100/100 [==============================] - 43s 348ms/step - loss: 5.5715 - masked_acc: 0.2088 - masked_loss: 5.5715 - val_loss: 4.4883 - val_masked_acc: 0.3005 - val_masked_loss: 4.4883
	Epoch 2/100
	100/100 [==============================] - 31s 306ms/step - loss: 4.0577 - masked_acc: 0.3545 - masked_loss: 4.0577 - val_loss: 3.6292 - val_masked_acc: 0.4000 - val_masked_loss: 3.6292
	......
	Epoch 24/100
	100/100 [==============================] - 32s 324ms/step - loss: 0.9762 - masked_acc: 0.7857 - masked_loss: 0.9762 - val_loss: 1.2726 - val_masked_acc: 0.7473 - val_masked_loss: 1.2726

（3）损失函数随着 epoch 的变化过程。

	plt.plot(history.history['loss'], label='loss')
	plt.plot(history.history['val_loss'], label='val_loss')
	plt.ylim([0, max(plt.ylim())])
	plt.xlabel('Epoch')
	plt.ylabel('loss')
	plt.legend()


![翻译任务损失函数变化.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b1ae19720b234208bfab0f56402ec719~tplv-k3u1fbpfcp-watermark.image?)

（4）准确率随着 epoch 的变化过程。

	plt.plot(history.history['masked_acc'], label='accuracy')
	plt.plot(history.history['val_masked_acc'], label='val_accuracy')
	plt.ylim([0, max(plt.ylim())])
	plt.xlabel('Epoch')
	plt.ylabel('acc')
	plt.legend()


![翻译任务准确率变化.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c3638fb50b324d409902574ede0c9ba3~tplv-k3u1fbpfcp-watermark.image?)
### 8. 推理

（1）当模型训练完成之后我们可以用训练好的模型进行翻译工作，这里我们定义了一个翻译函数，让模型的翻译的结果的最长长度为 50 。

	@Translator.add_method
	def translate(self, texts, *, max_length=500, temperature=tf.constant(0.0)):
	    context = self.encoder.convert_input(texts)
	    batch_size = tf.shape(context)[0]
	    next_token, done, state = self.decoder.get_initial_state(context)
	    tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)
	    for t in tf.range(max_length):
	        next_token, done, state = self.decoder.get_next_token(  context, next_token, done, state, temperature)
	        tokens = tokens.write(t, next_token)
	        if tf.reduce_all(done):
	            break
	    tokens = tokens.stack()
	    tokens = einops.rearrange(tokens, 't batch 1 -> batch t')
	    text = self.decoder.tokens_to_text(tokens)
	    return text

（2）我们使用了三个西班牙文的样本来进行翻译测试，并且我们给出了翻译的耗时，可以看出翻译基本准确。

	%%time
	inputs = [ 'Hace mucho frio aqui.',    # "It's really cold here."
	           'Esta es mi vida.',         # "This is my life."
	           'Su cuarto es un desastre.' # "His room is a mess"
	         ]
	for t in inputs:
	    print(model.translate([t])[0].numpy().decode())
	    
结果打印：

	its very cold here . 
	this is my life . 
	his room is a disaster . 
	CPU times: total: 577 ms
	Wall time: 578 ms    

### 9. 保存和加载模型

（1）我们要将训练好的模型进行保存，在使用的时候可以进行加载使用。

	class Export(tf.Module):
	    def __init__(self, model):
	        self.model = model
	    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
	    def translate(self, inputs):
	        return self.model.translate(inputs)
	export = Export(model)
	tf.saved_model.save(export, 'dynamic_translator',  signatures={'serving_default': export.translate})

（2）使用加载的模型进行推理

	%%time
	reloaded = tf.saved_model.load('dynamic_translator')
	result = reloaded.translate(tf.constant(inputs))
	print(result[0].numpy().decode())
	print(result[1].numpy().decode())
	print(result[2].numpy().decode())
	
结果打印：

	its very cold here .  
	this is my life .  
	his room is a disaster . 
	CPU times: total: 42.5 s
	Wall time: 42.8 s	