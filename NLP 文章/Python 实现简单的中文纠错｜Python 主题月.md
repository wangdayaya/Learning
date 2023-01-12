本文正在参加「Python主题月」，详情查看 [活动链接](https://juejin.cn/post/6979532761954533390/)
# 介绍

这篇文章主要是用 Python 实现了简单的中文分词的同音字纠错，目前的案例中只允许错一个字，自己如果有兴趣可以继续优化下去。具体步骤如下所示：

- 先准备一个文件，里面每一行中放一个中文分词，我这里的文件是下面代码中的 /Users/wys/Desktop/token.txt ，你们可以改成自己，再运行代码
- 将构建一个前缀树类，实现插入功能，将所有的标准分词都插入到前缀树中，另外实现一个搜索功能，用来搜索分词
- 将输入的错误分词中的每个字都找出 10 个同音字，将每个字都用 10 个同音字替换，结果可以最多得到 n*10 个分词，n 为分词的长度，因为有的音可能没有 10 个同音字。
- 将这些分词都经过前缀树的查找，如果能搜到，将其作为正确纠正就过返回



# 代码

    import re,pinyin
    from Pinyin2Hanzi import DefaultDagParams
    from Pinyin2Hanzi import dag


    class corrector():
        def __init__(self):
            self.re_compile = re.compile(r'[\u4e00-\u9fff]')
            self.DAG = DefaultDagParams()

        # 将文件中的词读取
        def getData(self):
            words = []
            with open("/Users/wys/Desktop/token.txt") as f:
                for line in f.readlines():
                    word = line.split(" ")[0]
                    if word and len(word) > 2:
                        res = self.re_compile.findall(word)
                        if len(res) == len(word): ## 保证都是汉字组成的分词
                            words.append(word)
            return words

        # 将每个拼音转换成同音的 10 个候选汉字，
        def pinyin_2_hanzi(self, pinyinList):
            result = []
            words = dag(self.DAG, pinyinList, path_num=10)
            for item in words:
                res = item.path  # 转换结果
                result.append(res[0])
            return result

        # 获得词经过转换的候选结结果
        def getCandidates(self, phrase):
            chars = {}
            for c in phrase:
                chars[c] = self.pinyin_2_hanzi(pinyin.get(c, format='strip', delimiter=',').split(','))
            replaces = []
            for c in phrase:
                for x in chars[c]:
                    replaces.append(phrase.replace(c, x))
            return set(replaces)

        # 获得纠错之后的正确结果
        def getCorrection(self, words):
            result = []
            for word in words:
                for word in self.getCandidates(word):
                    if Tree.search(word):
                        result.append(word)
                        break
            return result

    class Node:
        def __init__(self):
            self.word = False
            self.child = {}


    class Trie(object):
        def __init__(self):
            self.root = Node()

        def insert(self, words):
            for word in words:
                cur = self.root
                for w in word:
                    if w not in cur.child:
                        cur.child[w] = Node()
                    cur = cur.child[w]

                cur.word = True

        def search(self, word):
            cur = self.root
            for w in word:
                if w not in cur.child:
                    return False
                cur = cur.child[w]

            if cur.word == False:
                return False
            return True



    if __name__ == '__main__':
        # 初始化纠正器
        c = corrector()
        # 获得单词
        words = c.getData()
        # 初始化前缀树
        Tree = Trie()
        # 将所有的单词都插入到前缀树中
        Tree.insert(words)
        # 测试
        print(c.getCorrection(['专塘街道','转塘姐道','转塘街到']))

# 结果

打印结果为：

    ['转塘街道', '转塘街道', '转塘街道']
    
可以看出都纠正成功了，有一定的效果【狗头】