# 前言

本文使用 cpu 版本的 tensorflow 2.4 ，在 shakespeare 数据的基础上使用 Skip-Gram 算法训练词嵌入。

# 相关概念

Word2Vec 不是一个单纯的算法，而是对最初的神经概率语言模型 NNLM 做出的改进和优化，可用于从大型数据集中学习单词的词嵌入。通过word2vec 学习到的单词词嵌入已经被成功地运用于下游的各种 NLP 任务中。

Word2Vec 是轻量级的神经网络，其模型仅仅包括输入层、隐藏层和输出层，模型框架根据输入输出的不同，主要包括 CBOW 和 Skip-Gram 模型：

* CBOW ：根据周围的上下文词预测中间的目标词。周围的上下文词由当中间的目标词的前面和后面的若干个单词组成，这种体系结构中单词的顺序并不重要。
*  Skip-Gram ：在同一个句子中预测当前单词前后一定范围内的若干个目标单词。

# 本文大纲

1. 使用例子介绍 Skip-Gram 操作
2. 获取、处理数据
3. 搭建、训练模型
4. 查看 Word2Vec 向量

# 实现过程

### 1. 使用例子介绍 Skip-Gram 操作

（1）使用例子说明负采样过程 我们首先用一个句子 "我是中国人"来说明相关的操作流程。

（2）先要对句子中的 token 进行拆分，保存每个字到整数的映射关系，以及每个整数到字的映射关系。

（3）然后用整数对整个句子进行向量化，也就是用整数表示对应的字，从而形成一个包含了整数的的向量，需要注意的是这里要特意保留 0 作为填充占位符。

（4）sequence 模块提供可以简化 word2vec 数据准备的功能，我们使用 skipgram 函数，在 example\_sequence 中以每个单词为中心，与前后窗口大小为 window\_size 范围内的词生成 Skip-Gram 整数对集合。具体结果可以结合例子的输出理解。


	import io
	import re
	import string
	import tqdm
	import numpy as np
	import tensorflow as tf
	from tensorflow.keras import layers
	
	SEED = 2
	AUTOTUNE = tf.data.AUTOTUNE
	window_size = 2
	sentence = "我是一个伟大的中国人"
	tokens = list(sentence)
	vocab, index = {}, 1
	vocab['<pad>'] = 0
	for token in tokens:
	    if token not in vocab:
	        vocab[token] = index
	        index += 1
	vocab_size = len(vocab)
	inverse_vocab = {index: token for token, index in vocab.items()}
	example_sequence = [vocab[word] for word in tokens]
	positive_skip, _ = tf.keras.preprocessing.sequence.skipgrams( example_sequence,  vocabulary_size = vocab_size, window_size = window_size, negative_samples = 0)
	positive_skip.sort()
	for t, c in positive_skip:
	    print(f"({t}, {c}): ({inverse_vocab[t]}, {inverse_vocab[c]})")


所有正样本输出：

	(1, 2): (我, 是)
	(1, 3): (我, 一)
	(2, 1): (是, 我)
	(2, 3): (是, 一)
	(2, 4): (是, 个)
	(3, 1): (一, 我)
	(3, 2): (一, 是)
	(3, 4): (一, 个)
	(3, 5): (一, 伟)
	(4, 2): (个, 是)
	(4, 3): (个, 一)
	(4, 5): (个, 伟)
	(4, 6): (个, 大)
	(5, 3): (伟, 一)
	(5, 4): (伟, 个)
	(5, 6): (伟, 大)
	(5, 7): (伟, 的)
	(6, 4): (大, 个)
	(6, 5): (大, 伟)
	(6, 7): (大, 的)
	(6, 8): (大, 中)
	(7, 5): (的, 伟)
	(7, 6): (的, 大)
	(7, 8): (的, 中)
	(7, 9): (的, 国)
	(8, 6): (中, 大)
	(8, 7): (中, 的)
	(8, 9): (中, 国)
	(8, 10): (中, 人)
	(9, 7): (国, 的)
	(9, 8): (国, 中)
	(9, 10): (国, 人)
	(10, 8): (人, 中)
	(10, 9): (人, 国)
	
（5）skipgram 函数通过在给定的 window_size 上窗口上进行滑动来返回所有正样本对，但是我们在进行模型训练的时候还需要负样本，要生成 skip-gram 负样本，就需要从词汇表中对单词进行随机采样。使用 log\_uniform\_candidate\_sampler 函数对窗口中给定 target 进行 num\_ns 个负采样。我们可以在一个目标词 target 上调用负采样函数，并将上下文 context 出现的词作为 true\_classes ，以避免在负采样时被采样到。

（6）在较小的数据集中一般将 num_ns 设置为 [5， 20] 范围内的整数，而在较大的数据集一般设置为 [2， 5] 范围内整数。	

	target_word, context_word = positive_skip[0]
	num_ns = 3
	context_class = tf.expand_dims( tf.constant([context_word], dtype="int64"), 1)
	negative_sampling, _, _ = tf.random.log_uniform_candidate_sampler( true_classes=context_class,   num_true=1,  num_sampled=num_ns,  unique=True,  range_max=vocab_size, seed=SEED,  name="negative_sampling"  )
	
（7）我们选用了一个正样本 (我, 是) 来为其生成对应的负采样样本，目标词为“我”，该样本的标签类别定义为“是”，使用函数 log\_uniform\_candidate\_sampler 就会以“我”为目标，再去在词典中随机采样一个不存在于 true\_classes 的字作为负采样的标签类别， 如下我们生成了三个样本类别，可以分别组成（我，一）、（我，个）、（我，我）三个负样本。

（8）对于一个给定的正样本 skip-gram 对，每个样本对都是 (target\_word, context\_word) 的形式，我们现在又生成了 3 个负采样，将 1 个正样本 和 3 负样本组合到一个张量中。对于正样本标签标记为 1 和负样本标签标记为 0 。

	squeezed_context_class = tf.squeeze(context_class, 1)
	context = tf.concat([squeezed_context_class, negative_sampling], 0)
	label = tf.constant([1] + [0]*num_ns, dtype="int64")
	target = target_word
	print(f"target_index    : {target}")
	print(f"target_word     : {inverse_vocab[target_word]}")
	print(f"context_indices : {context}")
	print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
	print(f"label           : {label}")
	
结果为：

	target_index    : 1
	target_word     : 我
	context_indices : [2 3 4 1]
	context_words   : ['是', '一', '个', '我']
	label           : [1 0 0 0]
	
### 2. 获取、处理数据
（1）这里我们使用 tensorflow 的内置函数从网络上下载 shakespeare 文本数据，里面保存的都是莎士比亚的作品。

（2）我们使用内置的 TextVectorization 函数对数据进行预处理，并且将出现的所有的词都映射层对应的整数，并且保证每个样本的长度不超过 10

（3）将所有的数据都转化成对应的整数表示，并且设置每个 batcc_size 为 1024 。

	path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
	text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))
	def custom_standardization(input_data):
	    lowercase = tf.strings.lower(input_data)
	    return tf.strings.regex_replace(lowercase,  '[%s]' % re.escape(string.punctuation), '')
	vocab_size = 4096
	sequence_length = 10
	vectorize_layer = layers.TextVectorization(
	                        standardize=custom_standardization,
	                        max_tokens=vocab_size,
	                        output_mode='int',
	                        output_sequence_length=sequence_length)
	vectorize_layer.adapt(text_ds.batch(1024))
	inverse_vocab = vectorize_layer.get_vocabulary()
	text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
	sequences = list(text_vector_ds.as_numpy_iterator())

截取部分进行打印：

	for seq in sequences[:5]:
	    print(f"{seq} => {[inverse_vocab[i] for i in seq]}")
	    
 结果为：
 
	 [ 89 270   0   0   0   0   0   0   0   0] => ['first', 'citizen', '', '', '', '', '', '', '', '']
	[138  36 982 144 673 125  16 106   0   0] => ['before', 'we', 'proceed', 'any', 'further', 'hear', 'me', 'speak', '', '']
	[34  0  0  0  0  0  0  0  0  0] => ['all', '', '', '', '', '', '', '', '', '']
	[106 106   0   0   0   0   0   0   0   0] => ['speak', 'speak', '', '', '', '', '', '', '', '']
	[ 89 270   0   0   0   0   0   0   0   0] => ['first', 'citizen', '', '', '', '', '', '', '', '']
	
（4）我们将上面所使用到的步骤都串联起来，可以组织形成生成训练数据的函数，里面包括了正采样和负采样操作。另外可以使用 make\_sampling\_table 函数生成基于词频的采样概率表，对应到词典中第 i 个最常见的词的概率，为平衡期起见，对于越经常出现的词，采样到的概率越低。

（5）这里调用 generate\_training\_data 函数可以生成训练数据，target 的维度为 (64901,) ，contexts 和 labels 的维度为 (64901, 5) 。

（6）要对大量的训练样本执行高效的批处理，可以使用 Dataset 相关的 API ，使用 shuffle 可以从缓存的 BUFFER_SIZE 大小的样本集中随机选择一个，使用 batch 表示我们每个 batch 的大小设置为 BATCH\_SIZE ，使用 cache 为了保证在加载数据的时候不会出现 I/O 不会阻塞，我们在从磁盘加载完数据之后，使用 cache 会将数据保存在内存中，确保在训练模型过程中数据的获取不会成为训练速度的瓶颈。如果说要保存的数据量太大，可以使用 cache 创建磁盘缓存提高数据的读取效率。我们使用 prefetch 在训练过程中可以并行执行数据的预获取。

（7）最终每个样本最终的形态为 ((target\_word, context\_word),label) 。


	def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
	    targets, contexts, labels = [], [], []
	    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
	    for sequence in tqdm.tqdm(sequences):
	        positive_skip, _ = tf.keras.preprocessing.sequence.skipgrams(
	                              sequence,
	                              vocabulary_size=vocab_size,
	                              sampling_table=sampling_table,
	                              window_size=window_size,
	                              negative_samples=0)
	        for target_word, context_word in positive_skip:
	            context_class = tf.expand_dims(
	                              tf.constant([context_word], dtype="int64"), 1)
	            negative_sampling, _, _ = tf.random.log_uniform_candidate_sampler(
	                              true_classes=context_class,
	                              num_true=1,
	                              num_sampled=num_ns,
	                              unique=True,
	                              range_max=vocab_size,
	                              seed=seed,
	                              name="negative_sampling")
	            context = tf.concat([tf.squeeze(context_class,1), negative_sampling], 0)
	            label = tf.constant([1] + [0]*num_ns, dtype="int64")
	            targets.append(target_word)
	            contexts.append(context)
	            labels.append(label)
	    return targets, contexts, labels
	    
	targets, contexts, labels = generate_training_data( sequences=sequences, window_size=2,  num_ns=4, vocab_size=vocab_size, seed=SEED)
	targets = np.array(targets)
	contexts = np.array(contexts)
	labels = np.array(labels)
	BATCH_SIZE = 1024
	BUFFER_SIZE = 10000
	dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
	dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
	dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)	
### 3. 搭建、训练模型



（1）第一层是 target Embedding 层，我们为目标单词初始化词嵌入，设置词嵌入向量的维度为 128 ，也就是说这一层的参数总共有 (vocab\_size * embedding\_dim) 个，输入长度为 1 。

（2）第二层是 context Embedding 层，我们为上下文单词初始化词嵌入，我们仍然设置词嵌入向量的维度为 128 ，这一层的参数也有 (vocab\_size * embedding\_dim) 个，输入长度为 num_ns+1  。

（3）第三层是点积计算层，用于计算训练对中 target 和 context 嵌入的点积。

（4）我们选择 Adam 优化器来进行优化，选用 CategoricalCrossentropy 作为损失函数，选用 accuracy 作为评估指标，使用训练数据来完成 20 个 eopch 的训练。

	class Word2Vec(tf.keras.Model):
	    def __init__(self, vocab_size, embedding_dim):
	        super(Word2Vec, self).__init__()
	        self.target_embedding = layers.Embedding(vocab_size,
	                                          embedding_dim,
	                                          input_length=1,
	                                          name="w2v_embedding")
	        self.context_embedding = layers.Embedding(vocab_size,
	                                       embedding_dim,
	                                       input_length=num_ns+1)
	    def call(self, pair):
	        target, context = pair
	        if len(target.shape) == 2:
	            target = tf.squeeze(target, axis=1)
	        word_emb = self.target_embedding(target)
	        context_emb = self.context_embedding(context)
	        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
	        return dots
	    
	embedding_dim = 128
	word2vec = Word2Vec(vocab_size, embedding_dim)
	word2vec.compile(optimizer='adam',  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
	word2vec.fit(dataset, epochs=20)
	
	
过程如下：

	Epoch 1/20
	63/63 [==============================] - 1s 14ms/step - loss: 1.6082 - accuracy: 0.2321
	Epoch 2/20
	63/63 [==============================] - 1s 14ms/step - loss: 1.5888 - accuracy: 0.5527
	...
	Epoch 19/20
	63/63 [==============================] - 1s 13ms/step - loss: 0.5041 - accuracy: 0.8852
	Epoch 20/20
	63/63 [==============================] - 1s 13ms/step - loss: 0.4737 - accuracy: 0.8945	
### 4. 查看 Word2Vec 向量

我们已经训练好所有的词向量，可以查看前三个单词对应的词嵌入，不过因为第一个是一个填充字符，我们直接跳过了，所以只显示了两个单词的结果。

	weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
	vocab = vectorize_layer.get_vocabulary()
	for index, word in enumerate(vocab[:3]):
	    if index == 0:
	        continue  
	    vec = weights[index]
	    print(word, "||",' '.join([str(x) for x in vec]) + "")

输出：

	[UNK] || -0.033048704 -0.13244359 0.011660721 0.04466736 0.016815167 -0.0021747486 -0.22271504 -0.19703679 -0.23452276 0.11212586 -0.016061027 0.17981936 0.07774545 0.024562761 -0.17993309 -0.18202212 -0.13211365 -0.0836222 0.14589612 0.10907205 0.14628777 -0.10057361 -0.20254703 -0.012516517 -0.026788604 0.10540704 0.10908849 0.2110478 0.09297589 -0.20392798 0.3033481 -0.06899316 -0.11218286 0.08671802 -0.032792106 0.015512758 -0.11241121 0.03193802 -0.07420188 0.058226038 0.09341678 0.0020246594 0.11772731 0.22016191 -0.019723132 -0.124759704 0.15371098 -0.032143503 -0.16924457 0.07010268 -0.27322608 -0.04762394 0.1720905 -0.27821517 -0.021202642 0.022981782 0.017429957 -0.018919267 0.0821674 0.14892177 0.032966584 0.016503694 -0.024588188 -0.15450846 0.25163063 -0.09960359 -0.08205034 -0.059559997 -0.2328465 -0.017229442 -0.11387295 0.027335169 -0.21991524 -0.25220546 -0.057238836 0.062819794 -0.07596143 0.1036019 -0.11330178 0.041029476 -0.0036062107 -0.09850497 0.026396573 0.040283844 0.09707356 -0.108100675 0.14983237 0.094585866 -0.11460251 0.159306 -0.18871744 -0.0021350821 0.21181738 -0.11000824 0.026631303 0.0043079373 -0.10093511 -0.057986196 -0.13534115 -0.05459506 0.067853846 -0.09538108 -0.1882101 0.15350497 -0.1521072 -0.01917603 -0.2464314 0.07098584 -0.085702434 -0.083315894 0.01850418 -0.019426668 0.215964 -0.04208141 0.18032664 -0.067350626 0.29129744 0.07231988 0.2200896 0.04984232 -0.2129336 -0.005486685 0.0047443025 -0.06323578 0.10223014 -0.14854044 -0.09165846 0.14745502
	the || 0.012899147 -0.11042492 -0.2028403 0.20705906 -0.14402795 -0.012134922 -0.008227022 -0.19896115 -0.18482314 -0.31812677 -0.050654292 0.063769065 0.013379926 -0.04029531 -0.19954327 0.020137483 -0.035291195 -0.03429038 0.07547649 0.04313068 -0.05675204 0.34193155 -0.13978302 0.033053987 -0.0038114514 8.5749794e-05 0.15582523 0.11737131 0.1599838 -0.14866571 -0.19313708 -0.0936122 0.12842192 -0.037015382 -0.05241146 -0.00085232017 -0.04838702 -0.17497984 0.13466156 0.17985387 0.032516308 0.028982501 -0.08578549 0.014078035 0.11176433 -0.08876962 -0.12424359 -0.00049041177 -0.07127252 0.13457641 -0.17463619 0.038635027 -0.23191011 -0.13592774 -0.01954393 -0.28888118 0.0130044455 0.10935221 -0.10274326 0.16326426 0.24069212 -0.068884164 -0.042140033 -0.08411051 0.14803806 -0.08204498 0.13407354 -0.08042538 0.032217037 -0.2666482 -0.17485079 0.37256253 -0.02551431 -0.25904474 -0.002844959 0.1672513 0.035283662 -0.11897226 0.14446032 0.08866355 -0.024791516 -0.22040974 0.0137709975 -0.16484109 0.18097405 0.07075867 0.13830985 0.025787655 0.017255543 -0.0387513 0.07857641 0.20455246 -0.02442122 -0.18393797 -0.0361829 -0.12946953 -0.15860991 -0.10650375 -0.251683 -0.1709236 0.12092594 0.20731401 0.035180748 -0.09422942 0.1373039 0.121121824 -0.09530268 -0.15685256 -0.14398256 -0.068801016 0.0666081 0.13958378 0.0868633 -0.036316663 0.10832365 -0.21385072 0.15025891 0.2161903 0.2097545 -0.0487211 -0.18837014 -0.16750671 0.032201447 0.03347862 0.09050423 -0.20007794 0.11616628 0.005944925