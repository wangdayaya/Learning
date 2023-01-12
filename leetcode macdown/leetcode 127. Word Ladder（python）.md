leetcode  127. Word Ladder（python）

### 描述


A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s<sub>1</sub> -> s<sub>2</sub> -> ... -> s<sub>k</sub> such that:

* Every adjacent pair of words differs by a single letter.
* Every s<sub>i</sub> for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
* s<sub>k</sub> == endWord

Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.


Example 1:


	Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
	Output: 5
	Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
	
Example 2:

	Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
	Output: 0
	Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.




Note:

	1 <= beginWord.length <= 10
	endWord.length == beginWord.length
	1 <= wordList.length <= 5000
	wordList[i].length == beginWord.length
	beginWord, endWord, and wordList[i] consist of lowercase English letters.
	beginWord != endWord
	All the words in wordList are unique.


### 解析


根据题意，就是给出了一个 beginWord ，经过在 wordList 中一系列的查找，最后能得到 endWord 的单词序列长度，这里的查找要求每次只能变化一个字母。其实这个题就是类似于走迷宫，只不过每次变化的可能从 4 个变成了 26 个。这里我用集合 visited 来存储已经用过的单词，然后利用栈变量 queue 保存查找到的有效的单词，再对栈中的每个单词进行进一步深入的变化和查找，将可能的单词再追加入栈中，直到找到 endWord 为止，并返回这个寻找过程中产生的变化序列的距离。

不知道为何会超时，尴了个尬。
### 解答
				

	class Solution(object):
	    def ladderLength(self, beginWord, endWord, wordList):
	        """
	        :type beginWord: str
	        :type endWord: str
	        :type wordList: List[str]
	        :rtype: int
	        """
	        queue = [(beginWord, 1)]
	        visited = set()
	
	        while queue:
	            word, dist = queue.pop(0)
	            if word == endWord:
	                return dist
	            for i in range(len(word)):
	                for j in 'abcdefghijklmnopqrstuvwxyz':
	                    tmp = word[:i] + j + word[i+1:]
	                    if tmp not in visited and tmp in wordList:
	                        queue.append((tmp, dist+1))
	                        visited.add(tmp)
	        return 0
            	      
			
### 运行结果


	Time Limit Exceeded
	
	
### 解析


换一种思路，很多时间都浪费在了制造出来的 tmp 判断是否已经用过并且是否存在于 wordList 种，我们可以缩减这一部分的时间，如果这个制造出来的单词出现在 wordList ，说明就会一定用到它，所以不需要记录在 visited ，直接将其从 wordList 删除即可，这样可以减少布尔语句的判断时间，并且随着查找的进行不断删减单词， wordList 残留会越来越少，也会同时减少查找的时间。

### 解答

	class Solution(object):
	    def ladderLength(self, beginWord, endWord, wordList):
	        """
	        :type beginWord: str
	        :type endWord: str
	        :type wordList: List[str]
	        :rtype: int
	        """
	        wordList = set(wordList)
	        queue = [(beginWord, 1)]
	        for word, dist in queue:
	            if word == endWord:
	                return dist
	            for i in range(len(word)):
	                for char in 'abcdefghijklmnopqrstuvwxyz':
	                    tmp = word[:i] + char + word[i + 1:]
	                    if tmp in wordList:
	                        queue.append([tmp, dist + 1])
	                        wordList.remove(tmp)
	        return 0

### 运行结果

	Runtime: 513 ms, faster than 25.82% of Python online submissions for Word Ladder.
	Memory Usage: 14.6 MB, less than 62.38% of Python online submissions for Word Ladder.

原题链接：https://leetcode.com/problems/word-ladder/



您的支持是我最大的动力
