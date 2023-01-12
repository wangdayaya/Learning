
## 简单分类模型代码

	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	
	mnist = input_data.read_data_sets("MNIST", one_hot=True)
	batch_size = 16
	n_batches = mnist.train.num_examples // batch_size
	
	x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
	y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
	keep_prob = tf.placeholder(tf.float32)
	
	# 第一层隐藏层通过激活函数进行非线性变化
	w1 = tf.Variable(tf.zeros([784, 1024]))
	b1 = tf.Variable(tf.zeros(1024))
	a1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1) # relu 非线性激活函数
	o1 = tf.nn.dropout(a1, keep_prob) # dropout 操作
	
	# 第二层隐藏层通过激活函数进行非线性变化
	w2 = tf.Variable(tf.zeros([1024, 512]))
	b2 = tf.Variable(tf.zeros(512))
	a2 = tf.nn.sigmoid(tf.matmul(o1, w2) + b2)
	o2 = tf.nn.dropout(a2, keep_prob)
	
	# 第三层隐藏层通过激活函数进行非线性变化
	w3 = tf.Variable(tf.zeros([512,128]))
	b3 = tf.Variable(tf.zeros(128))
	a3 = tf.nn.sigmoid(tf.matmul(o2,w3) + b3)
	o3 = tf.nn.dropout(a3, keep_prob)
	
	# 输出层
	w4 = tf.Variable(tf.zeros([128, 10]))
	b4 = tf.Variable(tf.zeros(10))
	prediction = tf.nn.softmax(tf.matmul(o3, w4) + b4)
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	opt = tf.train.AdamOptimizer(0.001).minimize(loss)
	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) 
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	
	import time
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    total_batch = 0
	    last_batch = 0
	    best = 0
	    start = time.time()
	    for epoch in range(100):
	        for _ in range(n_batches):
	            batch_x, batch_y = mnist.train.next_batch(batch_size)
	            sess.run([opt], feed_dict={x:batch_x, y:batch_y, keep_prob:0.5})
	        loss_value, acc = sess.run([loss, accuracy], feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
	        if acc > best:
	            best = acc
	            last_batch  = total_batch
	            print('epoch:%d, loss:%f, acc:%f, time:%f' % (epoch, loss_value, acc, time.time()-start))
	            start = time.time()
	        if total_batch - last_batch > 5: # 早停条件
	            print('when epoch-%d early stop train'%epoch)
	            break
	        total_batch += 1
## 结果输出

	Extracting MNIST/train-images-idx3-ubyte.gz
	Extracting MNIST/train-labels-idx1-ubyte.gz
	Extracting MNIST/t10k-images-idx3-ubyte.gz
	Extracting MNIST/t10k-labels-idx1-ubyte.gz
	epoch:0, loss:1.609388, acc:0.849500, time:11.415676
	epoch:1, loss:1.525251, acc:0.935700, time:11.466139
	epoch:2, loss:1.515377, acc:0.946300, time:10.386464
	epoch:3, loss:1.507261, acc:0.954000, time:10.178594
	epoch:4, loss:1.503498, acc:0.957800, time:11.311379
	epoch:5, loss:1.501618, acc:0.959000, time:10.101135
	epoch:6, loss:1.499541, acc:0.961100, time:10.134475
	epoch:7, loss:1.496089, acc:0.965000, time:10.052625
	epoch:8, loss:1.495209, acc:0.965500, time:10.609939
	epoch:9, loss:1.494871, acc:0.966000, time:10.070237
	epoch:10, loss:1.490888, acc:0.970000, time:10.127296
	epoch:13, loss:1.490968, acc:0.970200, time:30.249309
	epoch:16, loss:1.489859, acc:0.971600, time:30.295541
	epoch:17, loss:1.489045, acc:0.971800, time:10.351570
	epoch:18, loss:1.487513, acc:0.974000, time:10.136432
	epoch:22, loss:1.486135, acc:0.974900, time:40.279734
	epoch:24, loss:1.485551, acc:0.975600, time:20.794270
	epoch:26, loss:1.485324, acc:0.975900, time:21.456657
	epoch:29, loss:1.485043, acc:0.976200, time:32.043005
	epoch:30, loss:1.483336, acc:0.978000, time:10.434125
	when epoch-36 early stop train

## 要点一

本文使用 tf.zeros 对 w1 、w2、w3 、w4 进行初始化，训练效果很棒，如果用 tf.random_normal 进行初始化收敛会比较慢，不信你可以试试。



## 要点二

本文使用的激活函数是 sigmoid ，但是可以换成其他的激活函数 relu 、tanh ，但是需要注意的是不能使用 tf.zeros 对  w1 、w2、w3 、w4 进行初始化，这样在训练过程的反向传播中无法对权重参数进行优化，所以我们要使用 tf.random_normal(shape, stddev=0.1) 对   w1 、w2、w3 、w4 进行初始化，这样收敛速度也比较快。

## 要点三

学习率按照常规设置为 0.001 数量级即可，学习率太大了模型训练效果毫无起色，甚至训练越来越差，学习率太小了模型训练也可能毫无起色，也可能收敛太慢，不信你可以将学习率换成 0.1 和 0.0001 试试。

## 要点四

可以看出使用多层的非线性变化与 Dropout 技术可以很大提高图片识别的准确率，在上一文中的简单分类模型准确率只有 0.926 ，而本文的技术准确率能达到 0.978 。

## 本文参考

本文参考: https://blog.csdn.net/qq_19672707/article/details/105589932


