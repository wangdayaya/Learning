## 简单的常量运算代码

	import tensorflow as tf
	v1 = tf.constant([[5,6]])
	v2 = tf.constant([[2],[4]])
	p1 = tf.matmul(v1, v2)
	p2 = tf.matmul(v2, v1)
	with tf.Session() as sess: # 因为这里没有变量，都是常量，所以直接可以进行运算，输出值
	    print(sess.run(p1))
	    print(sess.run(p2))
## 输出结果

	[[34]]
	[[10 12]
	 [20 24]]    
	
## 本文参考

本文参考：https://blog.csdn.net/qq_19672707/article/details/105113638