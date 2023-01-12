原理自己找，这里就是用示例简单实现 S2S 的原理，主要用来进行单词的翻译。
	

	import tensorflow as tf
	import numpy as np
	tf.reset_default_graph()
	
	# 构建字符字典、字符和索引之间的映射关系
	chars =  [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
	char2idx = {v:k for k,v in enumerate(chars)}
	idx2char = {k:v for k,v in enumerate(chars)}
	
	datas = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
	
	
	
	class Model:
	    def __init__(self):
	        self.step = 5  # 这里把输入的最大字符长度作为 step
	        self.hidden = 256  # 隐层大小
	        self.dim = 32  # 词向量维度
	        self.V = len(chars)  # 词典大小
	        # dropout 
	        self.dropout = tf.placeholder(tf.float32)
	        #初始化词向量
	        self.embedding = tf.get_variable(name="embedding", shape=[self.V, self.dim], initializer=tf.random_normal_initializer())
	        # encoder 的输入进行词嵌入
	        self.enc_input = tf.placeholder(tf.int32, [None, None])
	        # decoder 的输入进行词嵌入
	        self.dec_input = tf.placeholder(tf.int32, [None, None]) # [batch_size, step+1]
	        # decoder 的目标输出
	        self.targets = tf.placeholder(tf.int32, [None, None])  # [batch_size, step+1]
	        # 构造图
	        with tf.variable_scope('net') as scope:
	            self.buile_net()
	
	    def buile_net(self):
	        # encoder 词嵌入输入
	        enc_input_embedding = tf.nn.embedding_lookup(self.embedding, self.enc_input)   # [batch_size, step, dim]
	        # decoder 词嵌入输入
	        dec_input_embedding = tf.nn.embedding_lookup(self.embedding, self.dec_input)    # [batch_size, step+1, dim]
	
	        # encoder 过程
	        with tf.variable_scope('encode'):
	            enc_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden) 
	            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=self.dropout) # 进行 dropout
	            _, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input_embedding, dtype=tf.float32) #  enc_states : [batch_size, hidden]
	
	        # decoder 过程
	        with tf.variable_scope('decode'):
	            dec_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden)
	            dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=self.dropout)  # 进行 dropout
	            outputs, _ = tf.nn.dynamic_rnn(dec_cell, dec_input_embedding, initial_state=enc_states, dtype=tf.float32) # outputs : [batch_size, step+1, hidden]
	
	        # 全连接层输出每个样本的每个步骤的结果概率矩阵
	        logits = tf.layers.dense(outputs, self.V, activation=None)  # logits : [batch_size, step+1, V]
	        # 计算损失并优化
	        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.targets))
	        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
	        # 预测
	        self.prediction = tf.argmax(logits, 2)
	
	    def make_batch(self, datas):
	        input_batch, output_batch, target_batch = [], [], []
	        for data in datas:
	            for i in range(2):
	                data[i] =  data[i] + 'P' * (self.step - len(data[i]))  # 将 datas 中的每个单词都用 P 补齐成 step 长度
	
	            input = [char2idx[c] for c in data[0]]
	            output = [char2idx[c] for c in 'S'+data[1]]      # decoder 的输入需要开始符号
	            target = [char2idx[c] for c in data[1]+'E']       # decoder 的输出需要结束符号
	
	            input_batch.append(input)
	            output_batch.append(output)
	            target_batch.append(target)
	        return input_batch, output_batch, target_batch
	
	# 初始化 Model
	m = Model()  
	# 初始化 tf
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	# 生产数据
	input_batch, output_batch, target_batch = m.make_batch(datas)
	# 训练模型
	for epoch in range(5000):
	    _, loss = sess.run([m.optimizer, m.cost], feed_dict={m.enc_input:input_batch, m.dec_input:output_batch, m.targets:target_batch, m.dropout:0.5})
	    if (epoch+1)%1000 == 0:
	        print('epoch: ','%04d'%(epoch+1),' loss:','%6f'%(loss))
	# 翻译
	def translate(word):
	    tmp = [word, 'P'*len(word)]
	    input_batch, output_batch, _ = m.make_batch([tmp])
	    result = sess.run(m.prediction, feed_dict={m.enc_input:input_batch, m.dec_input:output_batch, m.dropout:1.})  # 预测时候不需要 dropout ，否则预测结果可能会不准
	    decoded = [idx2char[i] for i in result[0]]
	    end = decoded.index('E')
	    translated = ''.join(decoded[:end])
	    return translated.replace('P','')
	
	# 翻译示例 
	print('man ->', translate('man'))
	print('mans ->', translate('mans'))
	print('king ->', translate('king'))
	print('black ->', translate('black'))
	print('up ->', translate('up'))
	
结果打印：
		
	epoch:  1000  loss: 0.000143
	epoch:  2000  loss: 0.000105
	epoch:  3000  loss: 0.000022
	epoch:  4000  loss: 0.000029
	epoch:  5000  loss: 0.000005
	man -> women
	mans -> women
	king -> queen
	black -> white
	upp -> down