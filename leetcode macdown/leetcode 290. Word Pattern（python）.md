leetcode  290. Word Pattern（python）

### 每日经典

《象传》 ——孔子（春秋）

天行健，君子以自强不息

地势坤，君子以厚德载物

### 描述

Given a pattern and a string s, find if s follows the same pattern.

Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in s.



Example 1:

	Input: pattern = "abba", s = "dog cat cat dog"
	Output: true

	
Example 2:

	Input: pattern = "abba", s = "dog cat cat fish"
	Output: false


Example 3:

	
	Input: pattern = "aaaa", s = "dog cat cat dog"
	Output: false
	


Note:

	1 <= pattern.length <= 300
	pattern contains only lower-case English letters.
	1 <= s.length <= 3000
	s contains only lowercase English letters and spaces ' '.
	s does not contain any leading or trailing spaces.
	All the words in s are separated by a single space.


### 解析

根据题意，给定一个 pattern 和一个字符串 s，找出 s 是否遵循 pattern 的范式。这里的“遵循”表示完全匹配，使得 pattern 中的字母和 s 中的非空词之间存在双映射，即字母和词之间是一对一的关系如 "a" -> "dog"  意味着 "dog" -> "a" ，因为如果是单向的未出现错误，如  "a" -> "dog" 同时  "b" -> "dog" ，那么 "aba" -> "dog dog dog" ，这种情况是不符合题意的，所以我们要限制题意，满足双映射这一条件。

如果满足 pattern 的集合与 s 中单词的集合大小不相等，或者 pattern 的长度与 s 中单词的长度不相等，直接返回 False ，因为不满足双映射。然后定义字典 d ，同时遍历 pattern 的字母和 s 中的单词一一对应的位置，如果不满足映射关系直接返回 False ，否则遍历结束直接返回 True 。

总体来说难度简单，时间复杂度就是 O(n) ，空间复杂度也是 O(n) 。这道题虽然是个 easy 难度，但是只有 39 % 的通过率，比 medium 还要低，就是因为这个双映射的条件大家都没搞明白就擅自写代码，结果被边界 case 一顿报错导致的。

### 解答
				
	class Solution(object):
	    def wordPattern(self, pattern, s):
	        """
	        :type pattern: str
	        :type s: str
	        :rtype: bool
	        """
	        if len(set(pattern))!=len(set(s.split())) or len(pattern)!=len(s.split()):
	            return False
	       
	        d = {}
	        for a,b in zip(pattern, s.split()):
	            if a not in d:
	                d[a] = b
	            elif d[a] == b:
	                continue
	            else:
	                return False
	        return True

            	      
			
### 运行结果

	Runtime: 12 ms, faster than 95.07% of Python online submissions for Word Pattern.
	Memory Usage: 13.7 MB, less than 17.49% of Python online submissions for Word Pattern.


原题链接：https://leetcode.com/problems/word-pattern/



您的支持是我最大的动力
