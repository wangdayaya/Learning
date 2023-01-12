***本文正在参加[「金石计划 . 瓜分6万现金大奖」](https://juejin.cn/post/7162096952883019783 "https://juejin.cn/post/7162096952883019783")***

# 前言
本文使用 cpu 版本的 tensorflow-2.1 来说明过拟合与欠拟合现象，并介绍多种方法来解决此类问题。

# 本文大纲

1. 获取数据
2. 处理数据并进行归一化
3. 搭建模型训练过程函数
4. small_model 的模型训练效果
5. large_model 的模型训练效果
6. 在 large_model 基础上加入 L2 正则的模型训练效果
7. 在 large_model 基础上加入 Dropout 的模型训练效果
8. 在 large_model 基础上加入 L2 和 Dropout 的模型训练效果
9. 方法总结

# 主要思路和实现过程

### 1. 获取数据

（1）本文使用的是一份 HIGGS 数据，我们无需关注细节，只需要知道每一行数据的第一列是标签，其余列都是特征即可。

（2）可以通过这个内置的函数进行下载。

	gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz') 

（3）也可以直接复制下面地址在迅雷下载到文件夹。

	http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz 

（4）我这里直接下载到本地，通过 pandas 进行读取前 10000 行来完成本次的任务，本次任务主要说明过拟合和欠拟合及其解决方法，不需要关心训练的准确程度是否能达到实际使用效果。

	import tensorflow as tf
	import pandas as pd
	from tensorflow.keras import layers
	from tensorflow.keras import regularizers
	from matplotlib import pyplot as plt
	N_VALIDATION = int(1e3)
	N_TRAIN = int(1e4)
	BUFFER_SIZE = int(1e4)
	BATCH_SIZE = 500
	STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
	FEATURES = 28
	size_histories = {}
	regularizer_histories = {}
	dataset = pd.read_csv('C:/Users/QJFY-VR/.keras/datasets/HIGGS.csv.gz', names=[str(i) for i in range(29)], compression="gzip", nrows=10000)


### 2. 处理数据并进行归一化

（1）按照一定的比例，取 90% 的数据为训练数据，取 10% 的数据为测试数据。

	train_datas = dataset.sample(frac=0.9, random_state=0)
	test_datas = dataset.drop(train_datas.index)

（2）这里主要是使用一些内置的函数来查看训练集对每一列数据的各种常见的统计指标，主要有 count、mean、std、min、25%、50%、75%、max ，这样省去了我们的计算，在后面进行归一化操作时候直接使用即可。

	train_stats = d.describe()
	train_stats.pop("0")
	train_stats = train_stats.transpose()

（3）数据中的第 0 列就是我们需要预测的回归目标，我们将这一列从训练集和测试集中弹出，单独做成标签。

	train_labels = train_datas.pop('0')
	test_labels = test_datas.pop('0')

（4）这里主要是对训练数据和测试数据进行归一化，将每个特征应独立缩放到相同范围，因为当输入数据特征值存在不同范围时，不利于模型训练的快速收敛，这个在我们之前的文章中多次强调并展示过了。

	def norm(stats, x):
	    return (x - stats['mean']) / stats['std']
	train_datas = norm(train_stats, train_datas)
	test_datas = norm(train_stats, test_datas)
	
### 3. 搭建模型训练过程函数

（1）这里主要是将模型的配置、编译、训练都放到同一个函数中，可以适配不同的模型结构。

（2）我们选择了最常用的优化器 Adam ，并且将其学习率设置随着训练迭代次数的不断增加而进行衰减，这有利于模型的训练。

（3）因为这是一个二分类分体，所以损失值我们选择了最常见的二分交叉熵 BinaryCrossentropy 。

（4）评估指标我们也选择了二分交叉熵来对训练中的模型进行效果评估。

（5）为了防止可能出现无效的过拟合，我们还配合使用了 EarlyStopping ，在经过 50 次 epochs 后没有改进，则自动停止训练。

（6）最后函数返回了训练时的各种指标结果。

	def compile_and_fit(model, optimizer=None, max_epochs=10000):
	    if optimizer is None:
	        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay( 0.001, decay_steps=STEPS_PER_EPOCH*1000, decay_rate=1, staircase=False)
	        optimizer = tf.keras.optimizers.Adam(lr_schedule)
	    model.compile(  optimizer=optimizer,
	                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	                    metrics=[  tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'), 'accuracy'])
	    history = model.fit(train_datas,
	                        train_labels,
	                        batch_size=BATCH_SIZE,
	                        epochs=max_epochs,
	                        validation_split = 0.2,
	                        callbacks=[ tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=50) ],
	                        verbose=2)
	    return history
### 4. small_model 的模型训练效果

（1）这里主要是定义一个绘图函数，为后面出现的各模型绘制训练过程和验证过程的 binary_crossentropy 变化情况。
	
	def plot_history(history):
	    hist = pd.DataFrame(history.history)
	    hist['epoch'] = history.epoch
	    plt.figure()
	    plt.xlabel("Epochs")
	    plt.ylabel('binary_crossentropy')
	    plt.plot(hist['epoch'], hist['binary_crossentropy'], label='Train binary_crossentropy')
	    plt.plot(hist['epoch'], hist['val_binary_crossentropy'], label = 'Val binary_crossentropy')
	    plt.legend()
	    plt.show()
	    
（2）先使用一个最简单的模型来进行模型的训练，这个模型只有一层全连接层和一层输出层，虽然有激活函数 elu ，但是输出维度较小，最后将训练指标和验证指标进行绘制。

（3）可以看出随着 epoch 的不断增加，这个简单模型在训练过程中验证指标无限收敛于某个值，无法再进一步，训练指标还在继续优化，说明欠拟合，这是由于模型太过简单导致的，需要加大模型的复杂度。

	small_model = tf.keras.Sequential([
	    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
	    layers.Dense(1)
	])
	size_histories['Small'] = compile_and_fit(small_model)
	plot_history(size_histories['Small'])
	
        
![小模型欠拟合.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/77960b04ace641d09036b01a583558be~tplv-k3u1fbpfcp-watermark.image?)


### 5. large_model 的模型训练效果

(1)对一个比较复杂的多层模型进行训练，这个模型中有四层全连接层和一层输出层，每层全连接输出 64 维，并搭配使用 elu 激活函数，整体结果较为复杂，最后将训练指标和验证指标进行绘制。

（2）可以看出随着 epoch 的不断增加，这个模型的验证指标呈现先降后升的趋势，而训练指标呈现一直下降的趋势，这是一种典型的过拟合现象，也就是在训练集中表现良好但是在验证集或者测试集上表现较差，我们在后续通过加入正则和 Dropout 来抑制这一现象。
	
	large_model = tf.keras.Sequential([
	    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
	    layers.Dense(64, activation='elu'),
	    layers.Dense(64, activation='elu'),
	    layers.Dense(64, activation='elu'),
	    layers.Dense(1)
	])
	size_histories['Large'] = compile_and_fit(large_model)
	plot_history(size_histories['Large'])


![大模型过拟合.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fe0e5d60537c4cbaa5efdd5a0bb7ff5f~tplv-k3u1fbpfcp-watermark.image?)

### 6. 在 large_model 基础上加入 L2 正则的模型训练效果

（1）在上面 large_model 的基础上，每一层全连接层多加入了 L2 正则，整体结构较为复杂，最后将训练指标和验证指标进行绘制。

（2）可以看出随着 epoch 的不断增加，这个模型的验证指标和训练指标虽然有抖动但已经趋于正常，标明在加入 L2 正则之后对上面产生的过拟合现象起到了一定的抑制作用，但是验证指标与训练指标有一定的差距，仍然有一点欠拟合。

	l2_model = tf.keras.Sequential([
	    layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(0.001), input_shape=(FEATURES,)),
	    layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
	    layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
	    layers.Dense(64, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
	    layers.Dense(1)
	])
	regularizer_histories['l2'] = compile_and_fit(l2_model)
	plot_history(regularizer_histories['l2'])
	
        
![大模型加正则.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/70e8d0a9ed38454ea933e8571dcf1b96~tplv-k3u1fbpfcp-watermark.image?)
	
### 7. 在 large_model 基础上加入 Dropout 的模型训练效果

（1）在上面 large_model 的基础上，每一层全连接层中间加入了一层 Dropout ，整体结构较为复杂，最后将训练指标和验证指标进行绘制。

（2）可以看出随着 epoch 的不断增加，这个模型的验证指标和训练指标都有相同的趋势，标明在加入 Dropout 之后不仅对上面产生的过拟合现象起到了一定的抑制作用，而且有利于模型在验证集的泛化效果。可以看出在只是用一种方法的情况下，使用 Dropout 比 L2 正则的效果要更好，而且单纯使用 Dropout 的效果已经很出色了。

	dropout_model = tf.keras.Sequential([
	    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
	    layers.Dropout(0.5),
	    layers.Dense(64, activation='elu'),
	    layers.Dropout(0.5),
	    layers.Dense(64, activation='elu'),
	    layers.Dropout(0.5),
	    layers.Dense(64, activation='elu'),
	    layers.Dropout(0.5),
	    layers.Dense(1)
	])
	regularizer_histories['dropout'] = compile_and_fit(dropout_model)
	plot_history(regularizer_histories['dropout'])

![大模型加Dropout.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b37b4f2c58ac4621b1a1615eef91d91d~tplv-k3u1fbpfcp-watermark.image?)


### 8. 在 large_model 基础上加入 L2 和 Dropout 的模型训练效果

（1）在上面 large_model 的基础上，每一层全连接层中不仅加入了 L2 正则，并且在每一层全连接层中间加入了一层 Dropout ，整体结构相较于上面的模型是最复杂，最后将训练指标和验证指标进行绘制。

（2）可以看出随着 epoch 的不断增加，这个模型的验证指标和训练指标都有相同的趋势，模型在验证集和训练集中的效果同步提升，标明在加入 L2 和 Dropout 之后不仅对上面产生的过拟合现象起到了一定的抑制作用，而且有利于模型在验证集的泛化效果，并且模型在保证效果的同时收敛也较快。

	combined_model = tf.keras.Sequential([
	    layers.Dense(64, kernel_regularizer=regularizers.l2(0.0003), activation='elu', input_shape=(FEATURES,)),
	    layers.Dropout(0.5),
	    layers.Dense(64, kernel_regularizer=regularizers.l2(0.0003), activation='elu'),
	    layers.Dropout(0.5),
	    layers.Dense(64, kernel_regularizer=regularizers.l2(0.0003), activation='elu'),
	    layers.Dropout(0.5),
	    layers.Dense(64, kernel_regularizer=regularizers.l2(0.0003), activation='elu'),
	    layers.Dropout(0.5),
	    layers.Dense(1)
	])
	
	regularizer_histories['combined'] = compile_and_fit(combined_model)
	plot_history(regularizer_histories['combined'])

![大模型加L2和Dropout.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6d898e5edd1740a5a811f5e2572a5d52~tplv-k3u1fbpfcp-watermark.image?)

# 方法总结

其实由于深度学习模型的复杂程度，我们最常见到的就是过拟合现象，欠拟合现象较为少碰到，结合上文在这做一下总结。

应对欠拟合的方法总结：
 
* 加大模型复杂程度 
* 使用更多的数据集 
* 数据增强

应对过拟合的方法总结： 

* 减小模型复杂程度 
* 使用更多的数据集 
* 数据增强 
* 使用 L1 、L2 正则 
* 使用 Dropout 
* 结合使用正则和 Dropout 
* 数据归一化