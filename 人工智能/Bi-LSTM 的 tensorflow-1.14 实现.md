原理自己上网找，这里只用简单的示例来展示原理，输入一句话，用 BI-LSTM 预测下一个词。
	
	import tensorflow as tf
	import numpy as np
	
	sentences = 'Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis nostrud exercitation'
	# UNK 单词作为占位符号
	words = ['UNK'] + list(set([word for word in sentences.split()]))
	# 生成单词和索引的映射关系
	word2idx = {v:k for k,v in enumerate(words)}
	idx2word = {k:v for k,v in enumerate(words)}
	
	V = len(words)  # 字典大小
	step = len(sentence.split())  # 这里将 sentence 的长度作为了输入句子的长度
	hidden = 50   # 隐层大小
	dim = 50  # 词向量维度
	def make_batch(sentence):
	    input_batch, target_batch = [],[]
	    words = sentence.split()
	    for i,word in enumerate(words[:-1]):
	        input = [word2idx[word] for word in words[:i+1]]  # 将不同长度的句子转换成索引列表，当作输入的句子
	        input += [0] * (step - len(input))  # 将长度不足 step 的句子用 UNK 补齐
	        target = word2idx[words[i+1]]  # 用 input 句子的下一个单词索引作为标签
	        input_batch.append(input)
	        target_batch.append(np.eye(V)[target])  # 将 target 转换为 one-hot 编码，之后的 softmax_cross_entropy_with_logits_v2 会用到
	    return input_batch, target_batch
	    
	tf.reset_default_graph()
	
	# 初始化 词向量
	embedding = tf.get_variable(name="embedding", shape=[V, dim], initializer=tf.random_normal_initializer)
	X = tf.placeholder(tf.int32, [None, step])
	# 词嵌入
	XX = tf.nn.embedding_lookup(embedding, X)
	Y = tf.placeholder(tf.int32, [None, V])
	
	# 前向 LSTM
	lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden)
	# 反向 LSTM
	lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden)
	
	# 隐层计算
	# outputs : (fw=[batch_size, step, hidden], bw=[batch_size, step, hidden])
	# states : (fw=(c=[batch_size, hidden], h=[batch_size, hidden]), bw=(c=[batch_size, hidden], h=[batch_size, hidden]))
	outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, XX, dtype=tf.float32)
	
	# 将最后输出的正反方向的向量进行拼接
	outputs = tf.concat([outputs[0], outputs[1]], 2) # [batch_size, step, 2*hidden]
	outputs = tf.transpose(outputs, [1,0,2])[-1]    # [batch_size, 2*hidden]
	
	# 连接隐层和分类器的权重参数和偏置参数
	W = tf.Variable(tf.random_normal([2*hidden, V]))
	b = tf.Variable(tf.random_normal([V]))
	
	# 结果概率
	logits = tf.matmul(outputs, W) + b  # [batch_size, V]
	
	# 计算损失并进行优化
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
	
	# 预测
	prediction = tf.argmax(logits, 1)
	
	
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	
	# 生产输入和标签
	input_batch, target_batch = make_batch(sentence)
	# 训练模型
	for epoch in range(3000):
	    _, loss = sess.run([optimizer, cost], feed_dict={X:input_batch, Y:target_batch})
	    if (epoch+1)%500 == 0:
	        print("epoch=",'%04d'%(epoch+1)," loss=", '%04f'%(loss))
	        
	# 预测
	predict = sess.run([prediction], feed_dict={X:input_batch})
	for i,idxs in enumerate(input_batch):
	    print(" ".join([idx2word[idx] for idx in idxs if idx!=0]) ,' 预测单词:',idx2word[predict[0][i]], ' 真实单词:', idx2word[np.argmax(target_batch[i])])       
	    
结果打印：

	epoch= 0500  loss= 0.001467
	epoch= 1000  loss= 0.000409
	epoch= 1500  loss= 0.000199
	epoch= 2000  loss= 0.000116
	epoch= 2500  loss= 0.000074
	epoch= 3000  loss= 0.000050
	Lorem  预测单词: ipsum  真实单词: ipsum
	Lorem ipsum  预测单词: dolor  真实单词: dolor
	Lorem ipsum dolor  预测单词: sit  真实单词: sit
	Lorem ipsum dolor sit  预测单词: amet  真实单词: amet
	Lorem ipsum dolor sit amet  预测单词: consectetur  真实单词: consectetur
	Lorem ipsum dolor sit amet consectetur  预测单词: adipisicing  真实单词: adipisicing
	Lorem ipsum dolor sit amet consectetur adipisicing  预测单词: elit  真实单词: elit
	Lorem ipsum dolor sit amet consectetur adipisicing elit  预测单词: sed  真实单词: sed
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed  预测单词: do  真实单词: do
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do  预测单词: eiusmod  真实单词: eiusmod
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod  预测单词: tempor  真实单词: tempor
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor  预测单词: incididunt  真实单词: incididunt
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt  预测单词: ut  真实单词: ut
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut  预测单词: labore  真实单词: labore
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore  预测单词: et  真实单词: et
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et  预测单词: dolore  真实单词: dolore
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et dolore  预测单词: magna  真实单词: magna
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et dolore magna  预测单词: aliqua  真实单词: aliqua
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua  预测单词: Ut  真实单词: Ut
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut  预测单词: enim  真实单词: enim
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim  预测单词: ad  真实单词: ad
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad  预测单词: minim  真实单词: minim
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim  预测单词: veniam  真实单词: veniam
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam  预测单词: quis  真实单词: quis
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis  预测单词: nostrud  真实单词: nostrud
	Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis nostrud  预测单词: exercitation  真实单词: exercitation