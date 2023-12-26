## 目标
本文旨在介绍 tensorflow 入门知识点及实战示例，希望各位新手同学能在学习之后熟练 tensorflow 相关基本操作

## 简单的循环神经网络

	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	
	mnist = input_data.read_data_sets('MNIST', one_hot=True)
	batch_size = 64
	n_batches = mnist.train.num_examples // batch_size
	n_classes = 10 # 类别个数
	hidden_size = 128 # 隐层纬度
	steps = 28 # 最大序列
	embedding_size = 28 # 输入维度
	
	x = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.float32, [None, 10])
	
	weights = tf.Variable(tf.random_normal([hidden_size, n_classes], stddev=0.1))
	biases = tf.Variable(tf.zeros([n_classes]))
	
	def RNN(x, w, b):
	    inputs = tf.reshape(x, shape = [-1, steps, embedding_size]) # 将图像拉成一个时序序列
	    cell = tf.contrib.rnn.BasicRNNCell(hidden_size) # 每个 RNN 隐藏输出维度 hidden_size
	    _, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32) # rnn 模型计算获取最有状态向量
	    result = tf.nn.softmax(tf.matmul(final_state, w) + b) # 对结果进行 softmax
	    return result
	
	predict = RNN(x, weights, biases) # 预测结果
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict)) # 损失
	opt = tf.train.AdamOptimizer(0.001).minimize(loss) # 定义优化器
	correct = tf.equal(tf.argmax(y,1), tf. argmax(predict,1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # 准确率
	
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    total_batch = 0
	    last_batch = 0
	    best = 0
	    for epoch in range(100):
	        for _ in range(n_batches):
	            xx, yy = mnist.train.next_batch(batch_size)
	            sess.run(opt, {x:xx, y:yy})
	        acc, l = sess.run([accuracy, loss], {x:mnist.test.images, y:mnist.test.labels})
	        if acc > best:
	            best = acc
	            last_batch = total_batch
	            print('eopch:%d, acc:%f, loss:%f'%(epoch, acc, l))
	        if total_batch - last_batch > 5: # 早停条件
	            print('early stop')
	            break
	        total_batch += 1


## 结果输出

	eopch:0, acc:0.878200, loss:1.589309
	eopch:1, acc:0.907200, loss:1.556449
	eopch:2, acc:0.917600, loss:1.546643
	eopch:4, acc:0.933500, loss:1.528619
	eopch:5, acc:0.950100, loss:1.512312
	eopch:6, acc:0.951100, loss:1.511004
	early stop
	
## 要点一

关于循环神经网络及其变体的基础知识大家可以上网去找，很多学习资料，也可以参考我之前写的文章学习：https://juejin.cn/post/6972340784720773151

## 要点二

相较于上一个单纯的多隐层网络 + Dropout 只有 97.8% 的准确率，使用循环神经网络准确率只有 95% ，而且也没有卷积神经网络的准确率高，说明 CNN 在处理图像方面天生就有优势，循环神经网络适合处理文本数据。



## 本文参考

本文参考：https://blog.csdn.net/qq_19672707/article/details/105616284
