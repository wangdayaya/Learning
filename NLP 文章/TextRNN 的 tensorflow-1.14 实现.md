RNN 原理自己找，这里只给出简单例子的实现代码，实现对下一个单词的预测
	
	import tensorflow as tf
	import numpy as np
	
	tf.reset_default_graph()
	sentences = ['i love damao','i like mengjun','we love all']
	words = list(set(" ".join(sentences).split()))
	word2idx = {v:k for k,v in enumerate(words)}
	idx2word = {k:v for k,v in enumerate(words)}
	
	V = len(words)   # 词典大小
	step = 2   # 时间序列长度
	hidden = 5   # 隐层大小
	dim = 50   # 词向量维度

	# 制作输入和标签
	def make_batch(sentences):
	    input_batch = []
	    target_batch = []
	    for sentence in sentences:
	        words = sentence.split()
	        input = [word2idx[word] for word in words[:-1]]
	        target = word2idx[words[-1]]
	        
	        input_batch.append(input)
	        target_batch.append(np.eye(V)[target])   # 这里将标签改为 one-hot 编码，之后计算交叉熵的时候会用到
	    return input_batch, target_batch

	# 初始化词向量
	embedding = tf.get_variable(shape=[V, dim], initializer=tf.random_normal_initializer(), name="embedding")
	
	X = tf.placeholder(tf.int32, [None, step])
	XX = tf.nn.embedding_lookup(embedding,  X)
	Y = tf.placeholder(tf.int32, [None, V])
	
	# 定义 cell
	cell = tf.nn.rnn_cell.BasicRNNCell(hidden)
	# 计算各个时间点的输出和隐层输出的结果
	outputs, hiddens = tf.nn.dynamic_rnn(cell, XX, dtype=tf.float32)     # outputs: [batch_size, step, hidden] hiddens: [batch_size, hidden]
	
	# 这里将所有时间点的状态向量都作为了后续分类器的输入（也可以只将最后时间节点的状态向量作为后续分类器的输入）
	W = tf.Variable(tf.random_normal([step*hidden, V]))
	b = tf.Variable(tf.random_normal([V]))
	L = tf.matmul(tf.reshape(outputs,[-1, step*hidden]), W) + b
	
	# 计算损失并进行优化
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=L))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
	
	# 预测
	prediction = tf.argmax(L, 1)
	
	# 初始化 tf
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	
	# 喂训练数据
	input_batch, target_batch = make_batch(sentences)
	for epoch in range(5000):
	    _, loss = sess.run([optimizer, cost], feed_dict={X:input_batch, Y:target_batch})
	    if (epoch+1)%1000 == 0:
	        print("epoch: ", '%04d'%(epoch+1), 'cost= ', '%04f'%(loss))
	             
	# 预测数据
	predict = sess.run([prediction], feed_dict={X: input_batch})
	print([sentence.split()[:2] for sentence in sentences], '->', [idx2word[n] for n in predict[0]])
	 
	 
结果打印：

	epoch:  1000 cost=  0.008979
	epoch:  2000 cost=  0.002754
	epoch:  3000 cost=  0.001283
	epoch:  4000 cost=  0.000697
	epoch:  5000 cost=  0.000406
	[['i', 'love'], ['i', 'like'], ['we', 'love']] -> ['damao', 'mengjun', 'all'] 