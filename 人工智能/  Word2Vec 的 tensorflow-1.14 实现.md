	
	import tensorflow as tf
	import numpy as np
	
	tf.reset_default_graph()
	sentences = ["i like dog", "i like cat", "i like animal",
	              "dog love meat", "dog love milk", "dog love fish",
	              "dog like apple", "i like milk", "i love meat",
	            "cat like apple", "cat like milk", "cat love meat"]
	# 这里为了简单，就是将 sentences 中所有的句子中的单词都连到一起放进列表中
	words_sequence = " ".join(sentences).split()
	# 字典
	words = list(set(words_sequence))
	# 单词和索引的映射关系
	word2idx = {v:k for k,v in enumerate(words)}
	idx2word = {k:v for k,v in enumerate(words)}
	
	# 每批样本大小
	batch_size = 20
	# 词向量维度，为了能画出效果，只能选 3 维的，实际情况可以选择 50-300 以内的数字
	embedding_dim = 3
	# 负采样本大小，不能超过 batch_size
	num_sampled = 10
	# 词典大小
	V = len(words)
	# 生成随机输入和标签样本
	def random_batch():
	    skip_garms = []
	    for i in range(2, len(words_sequence)-2):
	        target = word2idx[words_sequence[i]]  # 目标单词
	        context = [word2idx[words_sequence[i-2]], word2idx[words_sequence[i-1]], word2idx[words_sequence[i+1]], word2idx[words_sequence[i+2]]] # 上下文窗口为 2 的所有单词
	        for w in context:
	            skip_garms.append([target, w])  # 组合成 【目标单词索引，任意一个上下文单词索引】的形式
	
	    random_inputs, random_labels =[], []   # 输出样本和标签样本
	    random_index = np.random.choice(range(len(skip_garms)), batch_size, replace=False)  # 从 skip_garms 中随机找 batch_size 个样本当作输入样本，每个样本都是 【目标单词索引，任意一个上下文单词索引】的形式
	    for i in random_index:
	        random_inputs.append(skip_garms[i][0])   # 中心词
	        random_labels.append([skip_garms[i][1]])   # 标签单词，这里把标签包装成 [batch_size ,1] 大小是为了进行后面的 nce_loss 计算
	    return random_inputs, random_labels
	# 初始化词向量
	embedding = tf.Variable(tf.random_uniform([V, embedding_dim], -1, 1))  # [V, dim]
	# 输入
	inputs = tf.placeholder(tf.int32, shape=[batch_size])
	# 对输入进行词嵌入
	inputs_embedding = tf.nn.embedding_lookup(embedding, inputs)   # [batch_size, dim]
	# 标签，这里把标签包装成 [batch_size ,1] 大小是为了进行后面的 nce_loss 计算
	labels = tf.placeholder(tf.int32, shape=[batch_size, 1])   # [batch_size, 1]
	# 负采样矩阵和词向量一样大
	nce_weights =tf.Variable(tf.random_uniform([V, embedding_dim], -1, 1))
	# 负采样偏置参数
	nce_biases = tf.Variable(tf.zeros([V]))
	# 计算损失函数并进行优化
	cost = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, labels, inputs_embedding, num_sampled, V))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
	# 初始化 tf
	with tf.Session() as sess:
	    init = tf.global_variables_initializer()
	    sess.run(init)
	    # 训练样本
	    for epoch in range(10000):
	        inputs_batch, labels_batch = random_batch()
	        _, loss = sess.run([optimizer, cost], feed_dict={inputs: inputs_batch, labels: labels_batch})
	        if (epoch+1)%1000 == 0:
	            print('epoch ','%04d'%(epoch+1), 'loss = ','%06f'%(loss))
	    # 拿到训练好的词向量进行三维展示
	    trained_embeddings = embedding.eval()
	    plt.rcParams['figure.figsize'] = (18.0, 14.0)
	    fig = plt.figure() 
	    ax = fig.gca(projection='3d')
	    for i, label in enumerate(words):
	        x, y, z = trained_embeddings[i]
	        ax.scatter(x, y, z, c='r')
	        ax.text(x, y, z, label)
	    plt.show()
	    plt.savefig(fname="w2v.png")
	    
结果打印

	epoch  1000 loss =  3.496442
	epoch  2000 loss =  3.246312
	epoch  3000 loss =  3.166934
	epoch  4000 loss =  3.191701
	epoch  5000 loss =  3.043439
	epoch  6000 loss =  2.950867
	epoch  7000 loss =  2.951037
	epoch  8000 loss =  2.861731
	epoch  9000 loss =  2.944158
	epoch  10000 loss =  2.967491

![结果](/Users/wys/Desktop/技术文档/w2v.png)