LSTM 的原理自己找，这里只给出简单的示例代码
	
	# LSTM 的原理自己找，这里只给出简单的示例代码
	
	import tensorflow as tf
	import numpy as np
	tf.reset_default_graph()
	
	# 预测最后一个字母
	words = ['make','need','coal','word','love','hate','live','home','hash','star']
	# 字典集
	chars = [c for c in 'abcdefghijklmnopqrstuvwxyz']
	# 生成字符索引字典
	word2idx = {v:k for k,v in enumerate(chars)}
	idx2word = {k:v for k,v in enumerate(chars)}
	
	V = len(chars) # 字典大小
	step = 3 # 时间步长大小
	hidden = 50 # 隐藏层大小
	dim = 32 # 词向量维度
	
	def make_batch(words):
	    input_batch, target_batch = [], []
	    for word in words:
	        input = [word2idx[c] for c in word[:-1]] # 除最后一个字符的所有字符当作输入
	        target = word2idx[word[-1]] # 最后一个字符当作标签
	        input_batch.append(input)
	        target_batch.append(np.eye(V)[target]) # 这里将标签转换为 one-hot ，后面计算 softmax_cross_entropy_with_logits_v2 的时候会用到
	    return input_batch, target_batch
	
	# 初始化词向量
	embedding  = tf.get_variable("embedding", shape=[V, dim], initializer=tf.random_normal_initializer)
	X = tf.placeholder(tf.int32, [None, step])
	# 将输入进行词嵌入转换
	XX = tf.nn.embedding_lookup(embedding, X)
	Y = tf.placeholder(tf.int32, [None, V])
	
	# 定义 LSTM cell
	cell = tf.nn.rnn_cell.BasicLSTMCell(hidden)
	# 隐层计算结果
	outputs, states = tf.nn.dynamic_rnn(cell, XX, dtype=tf.float32)   # output:  [batch_size, step, hidden]  states: (c=[batch_size, hidden], h=[batch_size, hidden])
	
	# 隐层连接分类器的权重和偏置参数
	W = tf.Variable(tf.random_normal([hidden, V]))
	b = tf.Variable(tf.random_normal([V]))
	
	# 这里只用到了最后输出的 c 向量 states[0] （也可以用所有时间点的输出特征向量）
	feature = tf.matmul(states[0], W) + b   # [batch_size, n_class]
	# 计算损失并进行迭代优化
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=feature, labels=Y))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
	
	# 预测
	prediction = tf.argmax(feature, 1) 
	
	# 初始化 tf
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	
	# 生产输入和标签
	input_batch, target_batch = make_batch(words)
	# 训练模型
	for epoch in range(1000):
	    _, loss = sess.run([optimizer, cost], feed_dict={X:input_batch, Y:target_batch})
	    if (epoch+1)%100 == 0:
	        print('epoch: ', '%04d'%(epoch+1), 'cost=', '%04f'%(loss))
	
	# 预测结果
	predict = sess.run([prediction], feed_dict={X:input_batch})
	print([words[i][:-1]+' '+idx2word[c] for i,c in enumerate(predict[0])])  
	
	
结果打印：

	epoch:  0100 cost= 0.003784
	epoch:  0200 cost= 0.001891
	epoch:  0300 cost= 0.001122
	epoch:  0400 cost= 0.000739
	epoch:  0500 cost= 0.000522
	epoch:  0600 cost= 0.000388
	epoch:  0700 cost= 0.000300
	epoch:  0800 cost= 0.000238
	epoch:  0900 cost= 0.000193
	epoch:  1000 cost= 0.000160
	
	['mak e', 'nee d', 'coa l', 'wor d', 'lov e', 'hat e', 'liv e', 'hom e', 'has h', 'sta r'] 