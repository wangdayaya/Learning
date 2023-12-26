## 简单的分类模型代码

	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	
	mnist = input_data.read_data_sets("MNIST", one_hot=True) # 读数据
	batch_size = 64
	n_batchs = mnist.train.num_examples // batch_size
	
	x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x') # 因为每张图片数据是 28*28=784 维的
	y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')  # 因为一共有 10 种类别的图片
	
	# softmax((x * w) + b)
	w = tf.Variable(tf.ones(shape=[784, 10]))
	b = tf.Variable(tf.zeros(shape=[10]))
	predict = tf.nn.softmax(tf.matmul(x, w) + b)
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
	opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	init = tf.global_variables_initializer()
	
	correct = tf.equal(tf.argmax(predict,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # 计算准确率
	
	with tf.Session() as sess:
	    sess.run(init)
	    total_batch = 0
	    last_batch = 0
	    best = 0
	    for epoch in range(100):
	        for _ in range(n_batchs):
	            xx,yy = mnist.train.next_batch(batch_size)
	            sess.run(opt, feed_dict={x:xx, y:yy})
	        loss_value, acc = sess.run([loss, accuracy], feed_dict={x:mnist.test.images, y:mnist.test.labels})
	        # 始终打印最好的准确率信息
	        if acc > best:
	            best = acc
	            last_batch  = total_batch
	            print('epoch:%d, loss:%f, acc:%f' % (epoch, loss_value, acc))
	        if total_batch - last_batch > 5: # 训练早停条件
	            print('when epoch-%d early stop train'%epoch)
	            break
	        total_batch += 1
	        
## 输出结果

	Extracting MNIST/train-images-idx3-ubyte.gz
	Extracting MNIST/train-labels-idx1-ubyte.gz
	Extracting MNIST/t10k-images-idx3-ubyte.gz
	Extracting MNIST/t10k-labels-idx1-ubyte.gz
	WARNING:tensorflow:From <ipython-input-18-ec0f1616d772>:16: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
	Instructions for updating:
	
	Future major versions of TensorFlow will allow gradients to flow
	into the labels input on backprop by default.
	
	See tf.nn.softmax_cross_entropy_with_logits_v2.
	
	epoch:0, loss:1.697559, acc:0.819900
	epoch:1, loss:1.627650, acc:0.887500
	epoch:2, loss:1.604011, acc:0.897200
	epoch:3, loss:1.592221, acc:0.902300
	epoch:4, loss:1.585058, acc:0.904700
	epoch:5, loss:1.579867, acc:0.907700
	epoch:6, loss:1.575740, acc:0.909700
	epoch:7, loss:1.572829, acc:0.911200
	epoch:8, loss:1.570307, acc:0.912600
	epoch:9, loss:1.567902, acc:0.913100
	epoch:10, loss:1.565990, acc:0.913900
	epoch:11, loss:1.564570, acc:0.916100
	epoch:13, loss:1.561729, acc:0.917800
	epoch:14, loss:1.560736, acc:0.917900
	epoch:15, loss:1.559514, acc:0.918600
	epoch:17, loss:1.557875, acc:0.919600
	epoch:18, loss:1.557073, acc:0.920100
	epoch:21, loss:1.554998, acc:0.920500
	epoch:22, loss:1.554592, acc:0.920700
	epoch:23, loss:1.553998, acc:0.921500
	epoch:24, loss:1.553378, acc:0.922100
	epoch:28, loss:1.551517, acc:0.922400
	epoch:29, loss:1.551527, acc:0.922700
	epoch:31, loss:1.550692, acc:0.923000
	epoch:32, loss:1.550284, acc:0.923200
	epoch:33, loss:1.550164, acc:0.923300
	epoch:34, loss:1.549571, acc:0.923600
	epoch:35, loss:1.549563, acc:0.923700
	epoch:38, loss:1.548744, acc:0.923800
	epoch:39, loss:1.548406, acc:0.924700
	epoch:41, loss:1.547895, acc:0.924800
	epoch:45, loss:1.547032, acc:0.925300
	epoch:49, loss:1.546252, acc:0.925900
	epoch:51, loss:1.545930, acc:0.926400
	epoch:56, loss:1.545088, acc:0.926700
	epoch:59, loss:1.544781, acc:0.927400
	epoch:65, loss:1.544077, acc:0.927500
	epoch:66, loss:1.543733, acc:0.927800
	epoch:70, loss:1.543496, acc:0.928100
	epoch:76, loss:1.542884, acc:0.928300
	epoch:80, loss:1.542315, acc:0.928600
	when epoch-86 early stop train    
	
## 本文参考

本文参考：https://blog.csdn.net/qq_19672707/article/details/105545952