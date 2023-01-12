这里只是用简单例子演示关于 self-attention 的逻辑，判断一句话的情感是正面或者是负面，具体原理自己百度即可。
	
	import tensorflow as tf
	import numpy as np
	tf.reset_default_graph()
	
	# 词向量维度
	dim = 2
	# 隐层大小
	hidden = 5
	# 时间步大小
	step = 3
	# 情感类别 正面或者负面
	N = 2
	
	sentences = ["i love mengjun","i like peipei","she likes damao","she hates wangda","wangda is good","mengjun is bad"]
	labels = [1,1,1,0,1,0]
	
	words = list(set(" ".join(sentences).split()))
	# 词典大小
	V = len(words)
	# 单词和索引互相映射
	word2idx = {v:k for k,v in enumerate(words)}
	idx2word = {k:v for k,v in enumerate(words)}
	
	# 处理输入数据
	input_batch = []
	for sentence in sentences:
	    input_batch.append([word2idx[word] for word in sentence.split()])
	
	# 处理输出目标数据
	target_batch = []
	for label in labels:
	    target_batch.append(np.eye(N)[label]) # 这里要进行独热编码，后面计算损失会用到
	    
	# 初始化词向量
	embedding = tf.Variable(tf.random_normal([V, dim]))
	# 输出分类时使用到的向量矩阵
	out = tf.Variable(tf.random_normal([hidden * 2, N]))
	
	X = tf.placeholder(tf.int32, [None, step])
	# 对输入进行词嵌入
	X_embedding = tf.nn.embedding_lookup(embedding, X)
	Y = tf.placeholder(tf.int32, [None, N])
	
	# 定义正向和反向的 lstm 
	lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden)
	lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden)
	
	# 经过双向 lstm 的计算得到结果 
	# output : ([batch_size, step, hidden],[batch_size, step, hidden])  
	# final_state : (fw:(c:[batch_size, hidden], h:[batch_size, hidden]), bw:(c:[batch_size, hidden], h:[batch_size, hidden]))
	output, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X_embedding, dtype=tf.float32)
	# 将 output 根据 hidden 维度拼接起来，[batch_size, step, hidden*2]
	output = tf.concat([output[0], output[1]], 2)
	
	# 将 final_state 的反方向的 c 和 h 根据 hidden 维度拼接起来， [batch_size, hidden*2]
	final_hidden_state = tf.concat([final_state[1][0], final_state[1][1]], 1)
	# 增加第三个维度，方便计算 [batch_size, hidden*2, 1]
	final_hidden_state = tf.expand_dims(final_hidden_state, 2)
	
	# 计算每个时间步的输出与最后输出状态的相似度 
	# [batch_size, step, hidden*2] * [batch_size, hidden*2, 1] = squeeze([batch_size, step, 1]) = [batch_size, step]
	attn_weights = tf.squeeze(tf.matmul(output, final_hidden_state), 2)
	# 在时间步维度上进行 softmax 得到权重向量
	soft_attn_weights = tf.nn.softmax(attn_weights, 1)
	
	# 各时间步输出和对应的权重想成得到上下文矩阵 [batch_size, hidden*2, step] * [batch_size, step, 1] = [batch_size, hidden*2, 1]
	context = tf.matmul(tf.transpose(output, [0, 2, 1]), tf.expand_dims(soft_attn_weights, 2))
	# squeeze([batch_size, hidden*2, 1]) = [batch_size, hidden*2]
	context = tf.squeeze(context, 2)
	
	# 输出概率矩阵 [batch_size, hidden*2] * [hidden*2, N] = [batch_size, N]
	model = tf.matmul(context, out)
	# 计算损失并优化
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model ,labels=Y))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
	# 预测
	hypothesis = tf.nn.softmax(model)
	prediction = tf.argmax(hypothesis, 1)
	
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	for epoch in range(5000):
	    _, loss = sess.run([optimizer, cost], feed_dict={X:input_batch, Y:target_batch})
	    if (epoch+1) % 1000 == 0:
	        print('epoch ','%06d'%(epoch+1), ' loss ', '%08f'%loss)
	        
	test_text = [[word2idx[word] for word in 'she hates wangda'.split()]]
	predict = sess.run([prediction], feed_dict={X: test_text})
	print('she hates wangda', '-->', predict[0][0])
	    
结果打印：

	epoch  001000  loss  0.001645
	epoch  002000  loss  0.000279
	epoch  003000  loss  0.000106
	epoch  004000  loss  0.000052
	epoch  005000  loss  0.000029
	she hates wangda --> 0  