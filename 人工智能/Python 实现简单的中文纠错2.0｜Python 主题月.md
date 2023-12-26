本文正在参加「Python主题月」，详情查看 [活动链接](https://juejin.cn/post/6979532761954533390/)

## 介绍
承接上文简单的[同音字纠错](https://juejin.cn/post/6980891267446079495)，本文介绍了稍微复杂的中文纠错，所改进的内容如下：

* 使用可以[容错的前缀树](https://juejin.cn/post/6981730849070776328)来存储所有分词
* 使用可以[容错的前缀树](https://juejin.cn/post/6981730849070776328)来存储所有拼音
* 使用 [Bi-gram](https://juejin.cn/post/6980252632912756750) 计算成词的概率值
* 中文分词的评价分数不光有 Bi-gram 值，还有拼音的编辑距离；对于拼音的评价分数目前只有拼音的编辑距离
* 输出的结果按评分降序排列，最多展示 5 个结果

## token.txt
token.txt 中存放着分词数据或者分词拼音数据，可以换成自己的文件，每行的格式为

	(单词,词频)
	或者
	(单词拼音,词率)
	
如下：

	浙江省,1
	杭州市,5
	西湖区,3
	zhejiangsheng,1
	hangzhoushi,5
	xihuqu,3


## main.py

主要是启动程序，构造前缀树，然后进行测试

	import csv, datetime, os
	from DictionaryTrie import Trie
	from corrector import Corrector
	from util import load_variavle, save_variable
	
	
	def buildTrieFromFile():
	    trie = Trie()
	    start = datetime.datetime.now()
	    savefile = 'save_tree'
	    if os.path.exists(savefile):
	        return load_variavle(savefile)
	    recommendFile = open('token.txt', 'r')
	    try:
	        recommendReader = csv.reader(recommendFile, delimiter=',')
	        for row in recommendReader:
	            trie.insert(row[0], int(row[1]))
	        save_variable(trie, savefile)
	    finally:
	        recommendFile.close()
	    end = datetime.datetime.now()
	    print("Time taken to build Dictionary Tree: " + str(end - start))
	    return trie
	
	
	def suggestorIO(trie, s, d):
	    suggestions = trie.findAll(s, d)
	    return [str(x.word) for x in suggestions]
	
	
	if __name__ == "__main__":
	    start = datetime.datetime.now()
	    trie = buildTrieFromFile()
	    c = Corrector()
	    r = suggestorIO(trie, '这江省', 1)
	    print(c.predict('这江省', r))
	
	    r = suggestorIO(trie, 'zhejiangshen', 1)
	    print(c.predict('zhejiangshen', r))
	
	    print("Time taken to get token_freq and words : " + str(datetime.datetime.now() - start))



## corrector.py

主要是对 Bi-Gram 概率矩阵、映射字典、预测评分方式的实现

	import datetime, math, pinyin, Levenshtein, os
	from util import load_variavle, save_variable
	
	# 起始字符
	START = 'S'
	# 终止字符
	END = 'E'
	class Corrector():
	    def __init__(self):
	        # 获得数据文本
	        self.texts = self.getTexts()
	        # 字频统计和单词
	        self.wordcount, self.words = self.init(self.texts)
	        # 映射字典
	        self.c2n, self.n2c = self.createDict(self.wordcount)
	        # 概率矩阵
	        self.matrix = self.calMatrix()
	
	    # 获得文本数据
	    def getTexts(self):
	        result = []
	        with open('token.txt', 'r') as f:
	            for line in f.readlines():
	                result.append(line.strip().split(',')[0])
	        return result
	
	    # 这里只给出了几个分词，得到字频和单词集合
	    def init(self, texts):
	        start = datetime.datetime.now()
	        savefile1 = 'save_wordcount'
	        savefile2 = 'save_words'
	        if os.path.exists(savefile1) and os.path.exists(savefile2):
	            return load_variavle(savefile1), load_variavle(savefile2)
	        # 字频统计
	        wordcount = {}
	        # 加入了 s 和 e 表示开始和结束，如 政苑小区 --> s政苑小区e ，是为了进行之后的概率计算
	        words = []
	        for word in texts:
	            if word and len(word) > 2:
	                word = START + word + END
	                words.append(word)
	                for c in word:
	                    if c not in wordcount:
	                        wordcount[c] = 1
	                    else:
	                        wordcount[c] += 1
	        save_variable(wordcount, savefile1)
	        save_variable(words, savefile2)
	        print("Time taken to get token_freq and words : " + str(datetime.datetime.now() - start))
	        return wordcount, words
	
	    # 字符和数字互相映射的字典
	    def createDict(self, CHAR_FREQ):
	        start = datetime.datetime.now()
	        savefile1 = 'save_c2n'
	        savefile2 = 'save_n2c'
	        if os.path.exists(savefile1) and os.path.exists(savefile2):
	            return load_variavle(savefile1), load_variavle(savefile2)
	        c2n = {}
	        n2c = {}
	        count = 0
	        for c in CHAR_FREQ:
	            c2n[c] = count
	            n2c[count] = c
	            count += 1
	        save_variable(c2n, savefile1)
	        save_variable(n2c, savefile2)
	        print("Time taken to get token_freq and words : " + str(datetime.datetime.now() - start))
	        return c2n, n2c
	
	    # 得到 bi-gram 矩阵中的频率值
	    def calMatrix(self):
	        start = datetime.datetime.now()
	        savefile = 'save_matrix'
	        if os.path.exists(savefile):
	            return load_variavle(savefile)
	        matrix = [[0] * len(self.wordcount) for i in range(len(self.wordcount))]
	        # 统计 bi-gram 出现频率
	        for key in self.words:
	            for i, c in enumerate(key):
	                if i == 0:
	                    continue
	                else:
	                    matrix[self.c2n[key[i - 1]]][self.c2n[c]] += 1
	        # 计算 bi-gram 概率矩阵
	        for i, line in enumerate(matrix):
	            for j, freq in enumerate(line):
	                matrix[i][j] = round(matrix[i][j] / self.wordcount[self.n2c[i]], 10)
	        save_variable(matrix, savefile)
	        print("Time taken to get token_freq and words : " + str(datetime.datetime.now() - start))
	        return matrix
	
	    # 计算给定单词的概率值
	    def predict(self, s, strings):
	        base = pinyin.get(s, format='strip', delimiter=' ')
	        result = {}
	        for s in strings:
	            r = 0
	            s = START + s + END
	            for i, c in enumerate(s):
	                if i == 0:
	                    continue
	                if c in self.c2n:
	                    r += math.log(self.matrix[self.c2n[s[i - 1]]][self.c2n[s[i]]] + 1)
	                else:
	                    r = 0
	                    break
	            t = s.lstrip(START).rstrip(END)
	            cmp = pinyin.get(t, format='strip', delimiter=' ')
	            if t not in result:
	                result[t] = r * 0.1 + (len(base) - Levenshtein.distance(base, cmp)) * 0.9
	        return sorted(result.items(), key=lambda x: x[1], reverse=True)[:5]

## DictionaryTrie.py

主要是对能够容错的前缀树的实现


	class Trie:
	    def __init__(self):
	        self.root = LetterNode('')
	        self.START = 3
	
	    def insert(self, word, freq):
	        self.root.insert(word, freq, 0)
	
	    def findAll(self, query, maxDistance):
	        suggestions = self.root.recommend(query, maxDistance, self.START)
	        return sorted(set(suggestions), key=lambda x: x.freq)
	
	
	class LetterNode:
	    def __init__(self, char):
	        self.REMOVE = -1
	        self.ADD = 1
	        self.SAME = 0
	        self.CHANGE = 2
	        self.START = 3
	        self.pointers = []
	        self.char = char
	        self.word = None
	
	    def charIs(self, c):
	        return self.char == c
	
	    def insert(self, word, freq, depth):
	        if depth < len(word):
	            c = word[depth].lower()
	            for next in self.pointers:
	                if next.charIs(c):
	                    return next.insert(word, freq, depth + 1)
	            nextNode = LetterNode(c)
	            self.pointers.append(nextNode)
	            return nextNode.insert(word, freq, depth + 1)
	        else:
	            self.word = Word(word, freq)
	
	    def recommend(self, query, movesLeft, lastAction):
	        suggestions = []
	        length = len(query)
	
	        if length >= 0 and movesLeft - length >= 0 and self.word:
	            suggestions.append(self.word)
	
	        if movesLeft == 0 and length > 0:
	            for next in self.pointers:
	                if next.charIs(query[0]):
	                    suggestions += next.recommend(query[1:], movesLeft, self.SAME)
	                    break
	
	        elif movesLeft > 0:
	            for next in self.pointers:
	                if length > 0:
	                    if next.charIs(query[0]):
	                        suggestions += next.recommend(query[1:], movesLeft, self.SAME)
	                    else:
	                        suggestions += next.recommend(query[1:], movesLeft - 1, self.CHANGE)
	                        if lastAction != self.CHANGE and lastAction != self.REMOVE:
	                            suggestions += next.recommend(query, movesLeft - 1, self.ADD)
	                        if lastAction != self.ADD and lastAction != self.CHANGE:
	                            if length > 1 and next.charIs(query[1]):
	                                suggestions += next.recommend(query[2:], movesLeft - 1, self.REMOVE)
	                            elif length > 2 and next.charIs(query[2]) and movesLeft == 2:
	                                suggestions += next.recommend(query[3:], movesLeft - 2, self.REMOVE)
	                else:
	                    if lastAction != self.CHANGE and lastAction != self.REMOVE:
	                        suggestions += next.recommend(query, movesLeft - 1, self.ADD)
	        return suggestions
	
	
	class Word:
	    def __init__(self, word, freq):
	        self.word = word
	        self.freq = freq


## util.py 

保存和加载变量的方法

	import pickle
	
	# 保存变量，之后运行会更加快速
	def save_variable(v, filename):
	    f = open(filename, 'wb')
	    pickle.dump(v, f)
	    f.close()
	    return filename
	
	# 加载变量
	def load_variavle(filename):
	    f = open(filename, 'rb')
	    r = pickle.load(f)
	    f.close()
	    return r

## 测试

运行 main.py 文件，会打印输出，并且会有 save\_c2n、save\_n2c、save\_matrix、save\_tree、save\_wordcount、save\_words 等文件产生，这些是为了保存变量，之后运行就不用耗时去构造树和计算概率矩阵等操作。结果打印如下：

	Time taken to get token_freq and words : 0:00:00.046511
	Time taken to get token_freq and words : 0:00:00.000983
	Time taken to get token_freq and words : 0:00:01.869512
	[('浙江省', 13.59619683451056)]
	[('zhejiangsheng', 19.375014751170497)]
	Time taken to get token_freq and words : 0:00:04.142090
	
如果用你们自己的字典，打印结果会和上面不同，但是类似。
## 展望

本次的实现虽然说比上次略微复杂，但是仍然是有提升的空间，后续会继续优化