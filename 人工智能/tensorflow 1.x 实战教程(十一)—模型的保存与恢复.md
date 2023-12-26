## 目标
本文旨在介绍 tensorflow 入门知识点及实战示例，希望各位新手同学能在学习之后熟练 tensorflow 相关基本操作

## 模型保存

	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	
	mnist = input_data.read_data_sets("MNIST", one_hot=True)
	batch_size = 64
	n_batches = mnist.train.num_examples // batch_size
	
	x = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.float32, [None, 10])
	 
	w = tf.Variable(tf.random_normal([784, 10], stddev=0.1))
	b = tf.Variable(tf.zeros([10]))
	predict = tf.nn.softmax(tf.matmul(x, w) + b)
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
	opt = tf.train.AdamOptimizer(0.001).minimize(loss)
	correct = tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	
	saver = tf.train.Saver()
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    total_batch = 0
	    last = 0
	    best = 0
	    for epoch in range(100):
	        for _ in range(n_batches):
	            xx,yy = mnist.train.next_batch(batch_size)
	            sess.run(opt, {x:xx, y:yy})
	        acc, l = sess.run([accuracy, loss], {x:mnist.test.images, y:mnist.test.labels})
	        if acc > best:
	            best = acc
	            last = total_batch
	            saver.save(sess, 'saved_model/model') # 每次只保存最好的结果
	            print(epoch, acc, l)
	        if total_batch - last > 5:
	            print('early stop')
	            break
	        total_batch += 1
	        
## 结果输出

	0 0.9035 1.5953374
	1 0.9147 1.5688152
	2 0.9212 1.5580758
	3 0.9234 1.552525
	4 0.9239 1.5495663
	5 0.9264 1.5462393
	6 0.9271 1.5441632
	7 0.9288 1.5419955
	8 0.9302 1.5403246
	12 0.9308 1.5376735
	14 0.9324 1.5360526
	19 0.9333 1.534032
	25 0.9338 1.5329739
	26 0.934 1.5326717
	early stop
	
## 模型读取

	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    saver.restore(sess, 'saved_model/model')
	    acc, l = sess.run([accuracy, loss], {x:mnist.test.images, y:mnist.test.labels})
	    print(acc, l)
	    
## 结果打印
  
  	0.934 1.5326717

## 要点一 

因为我们只保存效果最好的模型，所以我们在读取模型，使用相同数据进行测试的结果和训练的最后一次是一样的。

## 本文参考

本文参考：https://blog.csdn.net/qq_19672707/article/details/106082917