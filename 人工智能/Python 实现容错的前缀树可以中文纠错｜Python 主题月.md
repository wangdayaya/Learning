本文正在参加「Python主题月」，详情查看 [活动链接](https://juejin.cn/post/6979532761954533390/)
## 介绍

本文使用 Python 实现了前缀树，并且支持编辑距离容错的查询。文中的前缀树只存储了三个分词，格式为 (分词字符串,频率) ，如：('中海晋西园', 2)、('中海西园', 24)、('中南海', 4)，可以换成自己的文件进行数据的替换。在查询的时候要指定一个字符串和最大的容错编辑距离。

## 实现

	class Word:
	    def __init__(self, word, freq):
	        self.word = word
	        self.freq = freq
	
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
	        if ' ' in word:
	            word = [i for i in word.split(' ')]
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
	
	
	
	def buildTrieFromFile():
	    trie = Trie()
	    rows = [('中海晋西园', 2),('中海西园', 24),('中南海', 4)]
	    for row in rows:
	        trie.insert(row[0], int(row[1]))
	    return trie
	
	
	def suggestor(trie, s, maxDistance):
	    if ' ' in s:
	        s = [x for x in s.split(' ')]
	    suggestions = trie.findAll(s, maxDistance)
	    return [str(x.word) for x in suggestions]
	
	
	if __name__ == "__main__":
	    trie = buildTrieFromFile()
	    r = suggestor(trie, '中海晋西园', 1)
	    print(r)
	
## 分析

结果打印：

	['中海晋西园', '中海西园']

可以看出“中海晋西园”是和输入完全相同的字符串，编辑距离为 0 ，所以符合最大编辑距离为 1 的要求，直接返回。

“中海西园”是“中海晋西园”去掉“晋”字之后的结果，编辑距离为 1， 所以符合最大编辑距离为 1 的要求，直接返回。

另外，“中南海”和“中海晋西园”的编辑距离为 4 ，不符合最大编辑距离为 1 的要求，所以结果中没有出现。

## 参考
https://github.com/leoRoss/AutoCorrectTrie