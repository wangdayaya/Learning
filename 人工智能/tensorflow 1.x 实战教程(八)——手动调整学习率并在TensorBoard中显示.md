## 目标
本文旨在介绍 tensorflow 入门知识点及实战示例，希望各位新手同学能在学习之后熟练 tensorflow 相关操作


## 手动调整学习率代码

	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	
	mnist = input_data.read_data_sets("MNIST", one_hot=True)
	batch_size = 64
	n_batches = mnist.train.num_examples // batch_size
	
	def variable_info(var): # 将所有变量都存起来，一会可以在 tensorboard 的图中观察其动态
	    with tf.name_scope('summaries'):
	        mean_value = tf.reduce_mean(var)
	        tf.summary.scalar('mean', mean_value)
	        with tf.name_scope('stddev'):
	            stddev_value = tf.sqrt(tf.reduce_mean(tf.square(var - mean_value)))
	        tf.summary.scalar('stddev', stddev_value)
	        tf.summary.scalar('max', tf.reduce_max(var))
	        tf.summary.scalar('min', tf.reduce_min(var))
	        tf.summary.histogram('histogram',var)
	
	with tf.name_scope("input_layer"):
	    x = tf.placeholder(tf.float32, [None, 784])
	    y = tf.placeholder(tf.float32, [None, 10])
	    keep_prob = tf.placeholder(tf.float32)
	    lr = tf.Variable(0.01,tf.float32) # 这里的学习率设置为变量
	    tf.summary.scalar('learning_rate',lr) # 将 lr 存起来，一会可以在 tensorboard 的图中观察其动态
	    
	with tf.name_scope('network'): # 搭建网络结构
	    with tf.name_scope("weights"):
	        w = tf.Variable(tf.truncated_normal([784,10], stddev=0.1), name='w')
	        variable_info(w)
	    with tf.name_scope('baises'):
	        b = tf.Variable(tf.zeros([10]) + 0.1, name="b")
	        variable_info(b)
	    with tf.name_scope('xw_plus_b'):
	        a = tf.matmul(x,w) + b
	    with tf.name_scope('softmax'):
	        out = tf.nn.softmax(a)
	
	with tf.name_scope("loss_train"): # 计算损失值，定义优化器
	    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
	    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
	    tf.summary.scalar("loss", loss) # 将 loss 存起来，一会可以在 tensorboard 的图中观察其动态
	
	with tf.name_scope("eval"): # 计算准确率，评估模型效果
	    with tf.name_scope("correct"):
	        correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
	    with tf.name_scope("accuracy"):
	        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	        tf.summary.scalar('accuracy',accuracy)
	        
	init = tf.global_variables_initializer()
	merged = tf.summary.merge_all()
	
	with tf.Session() as sess:
	    sess.run(init)
	    writer = tf.summary.FileWriter('tflogs/', sess.graph) # 将结果文件都存入目标目录中，使用 tensorboard 工具可以查看
	    for epoch in range(20):
	        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
	        for batch in range(n_batches):
	            batch_x, batch_y = mnist.train.next_batch(batch_size)
	            summary, _ = sess.run([merged, train_step], feed_dict = {x:batch_x, y:batch_y, keep_prob:0.5})
	            writer.add_summary(summary, epoch * n_batches + batch)
	        loss_value, acc, lr_value = sess.run([loss, accuracy, lr], feed_dict = {x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
	        print("Iter: ", epoch, "Loss: ", loss_value, "Acc: ", acc, "lr: ", lr_value)
	        
## 结果输出

	Iter:  0 Loss:  1.6054871 Acc:  0.895 lr:  0.001
	Iter:  1 Loss:  1.5699576 Acc:  0.9148 lr:  0.00095
	Iter:  2 Loss:  1.5598879 Acc:  0.9195 lr:  0.0009025
	Iter:  3 Loss:  1.5546178 Acc:  0.9215 lr:  0.000857375
	Iter:  4 Loss:  1.5502373 Acc:  0.9254 lr:  0.00081450626
	Iter:  5 Loss:  1.5473799 Acc:  0.9269 lr:  0.0007737809
	Iter:  6 Loss:  1.5452079 Acc:  0.9277 lr:  0.0007350919
	Iter:  7 Loss:  1.5434842 Acc:  0.9294 lr:  0.0006983373
	Iter:  8 Loss:  1.5427189 Acc:  0.9278 lr:  0.0006634204
	Iter:  9 Loss:  1.5417348 Acc:  0.9293 lr:  0.0006302494
	Iter:  10 Loss:  1.540729 Acc:  0.9293 lr:  0.0005987369
	Iter:  11 Loss:  1.5403976 Acc:  0.9298 lr:  0.0005688001
	Iter:  12 Loss:  1.5395288 Acc:  0.9301 lr:  0.0005403601
	Iter:  13 Loss:  1.5395651 Acc:  0.9298 lr:  0.0005133421
	Iter:  14 Loss:  1.5387015 Acc:  0.9307 lr:  0.000487675
	Iter:  15 Loss:  1.5383359 Acc:  0.9308 lr:  0.00046329122
	Iter:  16 Loss:  1.5379355 Acc:  0.931 lr:  0.00044012666
	Iter:  17 Loss:  1.5374689 Acc:  0.9314 lr:  0.00041812033
	Iter:  18 Loss:  1.5376941 Acc:  0.9305 lr:  0.00039721432
	Iter:  19 Loss:  1.5371386 Acc:  0.9308 lr:  0.0003773536   
	
## 要点一

上述过程执行完如果想看保存的数据在图中的变化，可以按照下面的步骤来

* 	pip install tensorboard==1.10.0
* 	然后命令行运行 ./tensorboard --logdir=保存的绝对位置
* 	浏览器中打开 http://localhost:6006/ 就能看到相应的参数变化，很方便

## 要点二

调整学习率有助于训练模型，开始的时候使用较大点的学习率来保证模型效果快速收敛，在收敛到最优点附近时要尽量缩小学习率避免来回震荡。这里使用的就是最常见的一种学习率衰减方式，也就是指数衰减。

## 参考

本文参考：https://blog.csdn.net/qq_19672707/article/details/105596340