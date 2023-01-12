leetcode  1897. Redistribute Characters to Make All Strings Equal（python）

### 描述

You are given an array of strings words (0-indexed).

In one operation, pick two distinct indices i and j, where words[i] is a non-empty string, and move any character from words[i] to any position in words[j].

Return true if you can make every string in words equal using any number of operations, and false otherwise.





Example 1:

	Input: words = ["abc","aabc","bc"]
	Output: true
	Explanation: Move the first 'a' in words[1] to the front of words[2],
	to make words[1] = "abc" and words[2] = "abc".
	All the strings are now equal to "abc", so return true.

	
Example 2:
	
	Input: words = ["ab","a"]
	Output: false
	Explanation: It is impossible to make all the strings equal using the operation.





Note:

	1 <= words.length <= 100
	1 <= words[i].length <= 100
	words[i] consists of lowercase English letters.


### 解析

根据题意，就是在有单词列表，它们直接可以随意移动字符，判断是否移动过字符之后，所有的单词都能变一样。思路很简单，就是先用字典统计处所有字符的频率，只要所有字符的频率是单词个数的倍数就返回 True ，否则返回 False 。


### 解答
				

	class Solution(object):
	    def makeEqual(self, words):
	        """
	        :type words: List[str]
	        :rtype: bool
	        """
	        if len(words) == 1:
	            return True
	        N = len(words)
	        d = {}
	        for word in words:
	            for c in word:
	                if c not in d:
	                    d[c] = 1
	                else:
	                    d[c] += 1
	
	        for k, v in d.items():
	            if v % N != 0:
	                return False
	        return True
            	      
			
### 运行结果

	Runtime: 56 ms, faster than 73.97% of Python online submissions for Redistribute Characters to Make All Strings Equal.
	Memory Usage: 13.8 MB, less than 56.85% of Python online submissions for Redistribute Characters to Make All Strings Equal.


原题链接：https://leetcode.com/problems/redistribute-characters-to-make-all-strings-equal/



您的支持是我最大的动力
