本文正在参加「Python主题月」，详情查看 [活动链接](https://juejin.cn/post/6979532761954533390/)

## 介绍
上文介绍了[Python 实现 Bi-Gram ](https://juejin.cn/post/6980252632912756750)，本文继续深入，介绍 Python 实现 Tri-Gram 。

## 数据
数据都放在了 token.txt 文件中，每一行的格式就是一个分词：

	浙江省
	杭州市
	浙江县
	杭州县

当然每一行也可以是一句话。



## 实现
	import datetime, math
	
	# 起始字符
	START = 'S'
	# 终止字符
	END = 'E'
	
	class Corrector():
	    def __init__(self):
	        # 字频统计和单词
	        self.wordcount, self.words = self.init()
	        # 映射字典
	        self.c2n, self.n2c = self.createDict(self.wordcount)
	        # 字典个数
	        self.N = len(self.wordcount)
	        # 概率矩阵
	        self.matrix = self.calMatrix()
	
	
	    # 这里只给出了几个分词，得到字频和单词集合
	    def init(self):
	        start = datetime.datetime.now()
	        # 开始读数据
	        result = []
	        with open('token.txt', 'r') as f:
	            for line in f.readlines():
	                result.append(line.strip())
	        # 字频统计
	        wordcount = {}
	        # 加入了 s 和 e 表示开始和结束，如 政苑小区 --> s政苑小区e ，是为了进行之后的概率计算
	        words = []
	        for word in result:
	            if word and len(word) > 2:
	                word = START + word + END
	                words.append(word)
	                # 单个字的词频统计
	                for c in word:
	                    if c not in wordcount:
	                        wordcount[c] = 1
	                    else:
	                        wordcount[c] += 1
	                # 两个相邻字符的词频统计
	                for i, c in enumerate(word):
	                    if i < 2:
	                        continue
	                    if word[i - 2:i] not in wordcount:
	                        wordcount[word[i - 2:i]] = 1
	                    else:
	                        wordcount[word[i - 2:i]] += 1
	        print("Time taken to get token_freq and words : " + str(datetime.datetime.now() - start))
	        return wordcount, words
	
	    # 字符和数字互相映射的字典
	    def createDict(self, CHAR_FREQ):
	        start = datetime.datetime.now()
	
	        c2n = {}
	        n2c = {}
	        count = 0
	        for c in CHAR_FREQ:
	            c2n[c] = count
	            n2c[count] = c
	            count += 1
	        print("Time taken to get token_freq and words : " + str(datetime.datetime.now() - start))
	        return c2n, n2c
	
	    # 得到 tri-gram 矩阵中的频率值
	    def calMatrix(self):
	        start = datetime.datetime.now()
	
	        matrix = [[0] * len(self.wordcount) for i in range(len(self.wordcount))]
	        # 统计 tri-gram 出现频率
	        for key in self.words:
	            for i, c in enumerate(key):
	                if i < 2:
	                    continue
	                else:
	                    matrix[self.c2n[key[i - 2:i]]][self.c2n[c]] += 1
	        # 计算 tri-gram 概率矩阵
	        for i, line in enumerate(matrix):
	            for j, freq in enumerate(line):
	                matrix[i][j] = round((matrix[i][j]+1) / (self.wordcount[self.n2c[i]]+self.N), 10)
	        print("Time taken to get token_freq and words : " + str(datetime.datetime.now() - start))
	        return matrix
	
	    # 计算给定单词的概率值
	    def predict(self, strings):
	        result = {}
	        for s in strings:
	            r = 0
	            s = START + s + END
	            for i, c in enumerate(s):
	                if i < 2:
	                    continue
	                if s[i-2:i] in self.c2n and s[i] in self.c2n:
	                    r += math.log(self.matrix[self.c2n[s[i-2:i]]][self.c2n[s[i]]] + 1)
	                else:
	                    r = 0
	                    break
	            t = s.lstrip(START).rstrip(END)
	            if t not in result:
	                result[t] = r
	        return result
	
	
	c = Corrector()
	print(c.predict(['浙江省', '杭州', '这杭州']))

打印：

	Time taken to get token_freq and words : 0:00:00.000100
	Time taken to get token_freq and words : 0:00:00.000004
	Time taken to get token_freq and words : 0:00:00.000226
	{'浙江省': 0.3520474483650825, '杭州': 0.1978967684980622, '这杭州': 0}
	
## 体会
因为本文主要是实现代码，所以数据很简单，结果很快就计算出来了，但是当我用比较多的数据来进行计算的时候，会相当的慢，最后我试着将矩阵进行了存储，发现达到了 3GB ，真的是实践之后才能对矩阵的指数性爆炸增长有直观的体会。之前的 Bi-Gram 的矩阵存储才 30MB 左右。所以在实际项目中，避免矩阵太大都需要对其进行剪枝，最常见的方法就是 srilm 工具。

而且由于没有做平滑处理，矩阵会很稀疏，这也是需要注意的点，常见的方法有很多，这里只是列举一下，详细的内容可以自己上网学习：

* 拉普拉斯平滑（本文用到的就是这个）
* 内插与回溯
* Absolute Discounting
* Kneser-Ney Smoothing
* ...