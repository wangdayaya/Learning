原文链接拜上镇贴：https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

NNLM 的用法思路：输入句子序列，预测下一个单词（英文）或者字（中文）的一种语言模型。


操作步骤如图有如下（这里用英文句子为例，中文类似）:

![avatar](https://imgconvert.csdnimg.cn/aHR0cDovL3d3dy5jaGVuamlhbnF1LmNvbS9tZWRpYS91cGltZy9pbmRleF8yMDE5MDkyNzE3NDgxNV83NDMucG5n?x-oss-process=image/format,png)

1. 输入层：分词，单词索引互相映射，将目标单词当作标签，上文的前 n-1 个单词作为输入

2. 投影层：在投影层中，在 V*m 的词向量中进行 look_up ，其中 V 是词典的大小，而 m 是词向量维度。C 中每一行作为词的分布式表示，每一个单词都经过表 C 的转化变成一个词向量。n-1 个词向量首尾相接的拼起来，转化为 (n-1)\*m 的空间向量矩阵。

3.  隐藏层：根据公式进行计算 hidden = tanh(d+X*H) ，此层为 tanh 的非线性全连接层，H 和 d 可以理解为隐藏层的权重参数和偏置参数。

4. 输出层：根据公式进行计算 y=b+X\*W+ hidden\*U ，此层为 softmax 输出层，根据概率大小可以得到相应的预测单词，U 是输出层参数，b 是输出层偏置。W 是词向量直接连到输出通路的参数，但是矩阵并不是必要的，可以为 0 。


	
		
		import tensorflow as tf
		import numpy as np
		
		sentences = ['i like damao','i hate meng','i love maga']
		words = " ".join(sentences).split()
		words = list(set(words))
		word2idx = {v:k for k,v in enumerate(words)}
		idx2word = {k:v for k,v in enumerate(words)}
		
		n_step = 2  # 窗口大小
		dim = 50  # 词向量维度
		n_hidden = 10  # 隐藏层大小
		V = len(words)  # 词库大小
		
		# 生成输入和标签
		def make_batch(sentences):
		    input_batch = []
		    target_batch = []
		    for sentence in sentences:
		        words = sentence.split()
		        input = [word2idx[word] for word in words[:-1]]  # 将除了最后一个字的所有单词都加入 input
		        target = word2idx[words[-1]]  # 将要预测的单词加入 target
		        
		        input_batch.append(input)
		        target_batch.append(np.eye(V)[target])  # 这里是对 target 做热独编码，在下面计算 softmax_cross_entropy_with_logits_v2 会用到
		    return input_batch, target_batch
		    
		tf.reset_default_graph()
		
		X = tf.placeholder(tf.int64, [None, n_step])
		Y = tf.placeholder(tf.int64, [None, V])
		
		embedding = tf.get_variable(name="embedding", shape=[V, dim], initializer= tf.random_normal_initializer())  # 随机生成词向量
		XX = tf.nn.embedding_lookup(embedding, X)  # 词嵌入
		
		# 根据 y=b+X*W+tanh(d+X*H)*U 写代码
		input = tf.reshape(XX, shape=[-1, n_step*dim])
		H = tf.Variable(tf.random_normal([n_step*dim, n_hidden]))
		d = tf.Variable(tf.random_normal([n_hidden]))
		U = tf.Variable(tf.random_normal([n_hidden, V]))
		b = tf.Variable(tf.random_normal([V]))
		W = tf.Variable(tf.random_normal([n_step*dim, V]))
		A = tf.nn.tanh(tf.matmul(input, H) + d)
		B = tf.matmul(input, W) + tf.matmul(A, U) + b 
		
		# 计算损失并进行优化
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=B, labels=Y))
		optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
		
		# 预测结果
		prediction = tf.argmax(B, 1)
		
		# tf 初始化
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		
		# 开始训练
		input_batch, target_batch = make_batch(sentences)
		for epoch in range(5000):
		    _, loss = sess.run([optimizer, cost], feed_dict={X:input_batch, Y:target_batch})
		    if (epoch+1)%1000==0:
		        print('epoch:','%04d'%(epoch+1), 'cost =', '{:.6f}'.format(loss))
		
		# 使用训练样本进行简单的预测
		predict = sess.run([prediction], feed_dict={X:input_batch})
		print(predict)
		print([sen.split()[:2] for sen in sentences], '->', [idx2word[i] for i in predict[0]])
		        
	打印结果如下：

		epoch: 1000 cost = 0.001325
		epoch: 2000 cost = 0.000391
		epoch: 3000 cost = 0.000178
		epoch: 4000 cost = 0.000094
		epoch: 5000 cost = 0.000053
		[['i', 'like'], ['i', 'hate'], ['i', 'love']] -> ['damao', 'meng', 'maga']