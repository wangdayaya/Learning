leetcode  890. Find and Replace Pattern（python）




### 描述

Given a list of strings words and a string pattern, return a list of words[i] that match pattern. You may return the answer in any order. A word matches the pattern if there exists a permutation of letters p so that after replacing every letter x in the pattern with p(x), we get the desired word. Recall that a permutation of letters is a bijection from letters to letters: every letter maps to another letter, and no two letters map to the same letter.



Example 1:

	Input: words = ["abc","deq","mee","aqq","dkd","ccc"], pattern = "abb"
	Output: ["mee","aqq"]
	Explanation: "mee" matches the pattern because there is a permutation {a -> m, b -> e, ...}. 
	"ccc" does not match the pattern because {a -> c, b -> c, ...} is not a permutation, since a and b map to the same letter.


	
Example 2:

	Input: words = ["a","b","c"], pattern = "a"
	Output: ["a","b","c"]




Note:

	1 <= pattern.length <= 20
	1 <= words.length <= 50
	words[i].length == pattern.length
	pattern and words[i] are lowercase English letters.


### 解析

根据题意，给定一个字符串列表 words 和一个字符串 pattern ，返回一个匹配 pattern 的 words[i] 列表。如果存在字母排列 p 的，将模式中的每个字母 x 替换为 p(x) 之后，我们就得到了所需的单词。从字母到字母的映射：每个字母都映射到另一个字母，没有两个字母映射到同一个字母。

其实这道题很简单，就是遍历所有的 words 中的单词，然后比较单词和 pattern 是不是能够匹配到 pattern ，而这个关键就是字母之间的映射，只要满足了从单词到 pattern 的字符映射我们就认为是能够匹配的，否则说明不能够匹配。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def findAndReplacePattern(self, words, pattern):
	        """
	        :type words: List[str]
	        :type pattern: str
	        :rtype: List[str]
	        """
	        result = []
	        for word in words:
	            d = collections.defaultdict()
	            N = len(word)
	            i = 0
	            while i < N:
	                c,t = word[i],pattern[i]
	                if c not in d:
	                    if t in d.values():
	                        break
	                    else:
	                        d[c] = t
	                elif d[c] != t:
	                    break
	                i += 1
	            if i == N:
	                result.append(word)
	        return result
	


### 运行结果

	Runtime: 23 ms, faster than 82.11% of Python online submissions for Find and Replace Pattern.
	Memory Usage: 13.8 MB, less than 11.38% of Python online submissions for Find and Replace Pattern.

### 原题链接

	https://leetcode.com/problems/find-and-replace-pattern/


您的支持是我最大的动力
