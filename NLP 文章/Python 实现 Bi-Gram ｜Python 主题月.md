文中实现了 n-gram 中最简单的 bi-gram 的简单案例，具体的原理可以见[参考](https://blog.csdn.net/songbinxu/article/details/80209197)中的详解。

## 代码

	## 这里只给出了几个分词，得到字频和单词集合
	def init():
	    texts = ['政苑小区','万家花城','朝晖一区']
	    ## 字频统计
	    wordcount = {}
	    ## 加入了 s 和 e 表示开始和结束，如 政苑小区 --> s政苑小区e ，是为了进行之后的概率计算
	    words = []
	    for word in texts:
	        if word and len(word)>2:
	            word = 's' + word + 'e'
	            words.append(word)
	            for c in word:
	                if c not in wordcount:
	                    wordcount[c] = 1
	                else:
	                    wordcount[c] += 1
	    return wordcount,words
	
	
	## 字符和数字互相映射的字典
	def createDict(CHAR_FREQ):
	    c2n = {}
	    n2c = {}
	    count = 0
	    for c in CHAR_FREQ:
	        c2n[c] = count
	        n2c[count] = c
	        count += 1
	    return c2n,n2c
	
	## 字频统计 和 单词
	wordcount,words = init()
	
	## 映射字典 
	c2n,n2c = createDict(wordcount)
	
	## 得到 bi-gram 矩阵中的频率值
	def calMatrix():
	    matrix = [[0]*len(wordcount) for i in range(len(wordcount))] 
	    ## 统计 bi-gram 出现频率
	    for key in words:
	        for i,c in enumerate(key):
	            if i == 0 or len(key)-1==i:
	                continue
	            else:
	                matrix[c2n[key[i-1]]][c2n[c]] += 1
	    ## 计算 bi-gram 概率矩阵          
	    for i,line in enumerate(matrix):
	        for j,freq in enumerate(line):
	            matrix[i][j] = round(matrix[i][j]/wordcount[n2c[i]],7)
	    return matrix
	
	## 计算给定单词的概率值
	def predict(strings):
	    matrix = calMatrix()
	    result = []
	    for s in strings:
	        r = 0
	        s = 's' + s + 'e'
	        for i,c in enumerate(s):
	            if i==0:
	                continue
	            if c in c2n:
	                r += math.log(matrix[c2n[s[i-1]]][c2n[s[i]]]+1)
	            else:
	                r = 0
	                break
	        result.append(r)
	    return result
	
	print(predict(['政区区区','政朝小区','政苑小区','政家小区']))
	
	
结果打印：

	[0.28768204745178066, 0.9808292280117259, 2.3671235891316167, 0.9808292280117259]
	
正确的小区为“政苑小区”字符串，但是假如在不知道的情况下输入“政区区区”、“政朝小区”、“政苑小区”、“政家小区”四个词，从结果可以看出来“政苑小区”是得分最高的分词，所以优先选择这个答案。当然从事先知道的结果来看，这个答案也是正确的答案，所以我们在用 Bi-Gram 的时候，我们也可以找出分数值最大的字符串最为概率最大的结果。

## 拓展
其实在这个 Bi-Gram 基础之上，还可以拓展出 Tri-Gram ，而这个也是常用的模型，但是 Four-Gram 等等再大的话，虽然会对下一个词或者字的辨别能力更强，约束信息更多，但是生成的矩阵也更加洗漱，最后 N-Gram 的总数更多，为 V<sup>n</sup>
 个，V 为词典或者字典的大小。
 
 在修改上面代码的时候，只需要修改 calMatrix 方法即可。


## 参考	

* https://blog.csdn.net/songbinxu/article/details/80209197