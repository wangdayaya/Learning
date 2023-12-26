## 简单的 fetch 代码

	import tensorflow as tf
	
	a = tf.constant(1.0)
	b = tf.constant(4.0)
	c = tf.constant(7.0)
	add = tf.add(tf.add(a, b), c)
	mul = tf.multiply(tf.multiply(a, b), c)
	with tf.Session() as sess:
	    print(sess.run(add))
	    print(sess.run(mul))
## 输出结果
    
	12.0
	28.0
## 简单的 feed 代码

	import tensorflow as tf
	
	n = tf.placeholder(tf.int8)
	m = tf.placeholder(tf.int8)
	r = tf.multiply(n,m)
	with tf.Session() as sess:
	    print(sess.run(r, feed_dict={n:3,m:4}))   
## 输出结果

	12    
## 本文参考
本文参考：https://blog.csdn.net/qq_19672707/article/details/105235728  