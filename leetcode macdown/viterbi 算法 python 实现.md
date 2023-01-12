### 前言

默认读者已经了解 HMM 相关的内容，并对 viterbi 算法有了解。

### 前提条件
	
state 隐藏状态列表，observation  是可观测状态列表，T 是转移矩阵， E 是发射矩阵， PI 是初始状态概率矩阵，O 是当前已知的三天的海藻湿度序列（这里直接用 observation 的索引表示，也就是对应的 ['Dry', 'Damp', 'Soggy'] ），让我们找出这三天在出现 O 情况下，最有可能的天气状况序列。
	
	state = ['Sunny','Cloud','Rainy']
	observation = ['Dry', 'Dryish', 'Damp', 'Soggy']
	PI = [0.63,0.17,0.20]
	T = [[0.5,0.375,0.125],
	     [0.25,0.125,0.625],
	     [0.25,0.375,0.375]]
	E = [[0.6,0.2,0.15,0.05],
	     [0.25,0.25,0.25,0.25],
	     [0.05,0.10,0.35,0.5]]
	O = [0,2,3]

### 算法实现

	import numpy as np
	def viterbi(T ,E, PI, O):
	    T = np.array(T)
	    E = np.array(E)
	    PI = np.array(PI)
	    row = T.shape[0]
	    col = len(O)
	    F = np.zeros((row, col))
	    F[:, 0] = PI * np.transpose(E[:, O[0]])
	    for t in range(1, col):
	        L_max = []
	        for i in range(row):
	            L = F[:, t-1] * np.transpose(T[:, i])
	            L_max.append(max(L))
	        F[:, t] = np.array(L_max) * np.transpose(E[:, O[t]])
	    return F

算法的关键就是，昨天出现某个天气的概率\*昨天变成今天某个天气的转移概率\*今天是某个天气情况下海藻的湿度概率，只不过上面的代码为了加速计算，都改成了矩阵的运算，但是原理不变。

代码最后返回的是 state 个数\* O 个数的结果矩阵，我们要通过这个矩阵，找出每一列中最大的概率对应的天气状况，这样将所有列找出的天气组成的序列就是最后的结果。

### 代码运行
	
	V = viterbi(T, E, PI, O)
	print(V)
	# [[ 0.378　　   0.02835　　      0.00070875]
	# [ 0.0425 　　 0.0354375 　　 0.00265781]
	# [ 0.01 　　     0.0165375 　　 0.01107422]]
	print([state[np.argmax(row)] for row in np.transpose(V)])
	# ['Sunny', 'Cloud', 'Rainy']
	
从结果看这三天最有可能的天气状况是	  ['Sunny', 'Cloud', 'Rainy']

### 参考推荐

这里有一位作者对这个经典的维比特算法案例有详细的推算过程，大家可以对着参考一下，有助于对算法和代码的理解，我就不再造轮子了，但是需要注意的是他计算的答案有误，只需要理解算法过程就可以了，自己也能手推一下: https://blog.csdn.net/jeiwt/article/details/8076739