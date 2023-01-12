## 目标
本文旨在介绍 tensorflow 入门知识点及实战示例，希望各位新手同学能在学习之后熟练 tensorflow 相关基本操作

## 简单的卷积神经网络实现

	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	
	mnist = input_data.read_data_sets("MNIST", one_hot=True)
	batch_size = 64
	n_batches = mnist.train.num_examples // batch_size
	
	def weight_variable(shape):
	    return tf.Variable(tf.random_normal(shape, stddev=0.1))
	def biases_variable(shape):
	    return tf.Variable(tf.constant(0.1, shape=shape))
	def conv2d(x, w):
	    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME') # strides 表示步长，padding 表示填充方式
	def max_pool_2x2(x):
	    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # ksize 表示卷积核大小
	
	x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
	y = tf.placeholder(shape=[None, 10], dtype=tf.float32)
	x_ = tf.reshape(x, [-1, 28, 28, 1]) # 因为图片本身就是 28*28*1 的大小，28 是图片像素点的长和宽的个数，1 是每个像素点像素维度
	
	w1 = weight_variable([5,5,1,32]) # 这个权重是靠公式可以算出来的
	b1 = biases_variable([32]) # 保证和 w 的最后一个维度相同即可
	
	h1 = tf.nn.relu(conv2d(x_, w1) + b1) # 使用 relu 激活函数
	p1 = max_pool_2x2(h1) # 最大池化操作
	
	w2 = weight_variable([5,5,32,64])
	b2 = biases_variable([64])
	
	h2 = tf.nn.relu(conv2d(p1, w2) + b2) # 使用 relu 激活函数
	p2 = max_pool_2x2(h2) # 最大池化操作
	
	w_fc1 = weight_variable([7*7*64, 1024]) # 全联接层操作
	b_fc1 = biases_variable([1024]) # 保证和 w_fc1 最后一个维度相同
	
	p2_flat = tf.reshape(p2, [-1, w_fc1.shape[0]])
	h_fc1 = tf.nn.tanh(tf.matmul(p2_flat, w_fc1) + b_fc1)
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	w_fc2 = weight_variable([1024, 10]) # 输出层操作
	b_fc2 = biases_variable([10]) # 保证和 w_fc2 最后一个维度相同
	prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	opt = tf.train.AdamOptimizer(0.001).minimize(loss)
	correct = tf.equal(tf.argmax(y,1), tf.argmax(prediction, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    for epoch in range(10):
	        for _ in range(n_batches):
	            xx, yy = mnist.train.next_batch(batch_size)
	            sess.run(opt, feed_dict={x:xx, y:yy, keep_prob:0.5})
	        acc, l = sess.run([accuracy, loss], feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
	        print(epoch, l, acc)
	        
## 结果输出

	0 1.5783486 0.8818
	1 1.4810778 0.9803
	2 1.4758472 0.9855
	3 1.472993 0.9884
	4 1.4741219 0.9866
	5 1.4728734 0.9882
	6 1.4742823 0.9869
	7 1.4712367 0.9898
	8 1.4690293 0.9922
	9 1.473154 0.988
	10 1.4709185 0.9904

## 要点一

关于卷积神经网络的基础知识大家可以上网去找，很多学习资料，包括卷积核、步长、padding 等内容

## 要点二  

相较于上一篇文章中介绍的单纯的多隐层网络 + Dropout 只有 97.8% 的准确率，使用卷积神经网络准确率可以轻松提升到 99% ，说明卷积神经网络天生适合处理图像数据，但是由于模型的复杂性增加，训练耗费的资源和时间也增多了


## 本文参考

本文参考：https://blog.csdn.net/qq_19672707/article/details/105613732