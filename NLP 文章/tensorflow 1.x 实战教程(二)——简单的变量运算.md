## 简单的变量运算

	import tensorflow as tf
	x = tf.Variable([9,10])
	y = tf.constant([4,4])
	sub = tf.subtract(x, y)
	add = tf.add(x, y)
	init = tf.global_variables_initializer() # 这里因为 graph 中有变量 x ，所以要有一个操作对 graph 中的变量进行初始化
	with tf.Session() as sess:
	    sess.run(init)
	    print(sess.run([sub, add]))
## 输出结果	

	[array([5, 6], dtype=int32), array([13, 14], dtype=int32)]
	
## 进阶—变量自增

	import tensorflow as tf
	state = tf.Variable(0, name='state')
	add = tf.add(state, 2) # 为 state 加 2
	update = tf.assign(state, add) # 将变化之后的 add 赋值给 state 完成值的更新
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
	    sess.run(init)
	    print(sess.run(state)) # 输出原始的 state 值
	    for _ in range(3):
	        sess.run(update) # update 操作中已经包含了加法和赋值两个操作
	        print(sess.run(state)) # 输出变化之后的 state 值
## 输出结果
	0
	2
	4
	6   
## 本文参考    

本文参考：https://blog.csdn.net/qq_19672707/article/details/105233828