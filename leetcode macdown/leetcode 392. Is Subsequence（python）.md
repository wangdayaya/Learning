leetcode  392. Is Subsequence（python）




### 描述

Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).



Example 1:


	Input: s = "abc", t = "ahbgdc"
	Output: true
	





Note:

	0 <= s.length <= 100
	0 <= t.length <= 10^4
	s and t consist only of lowercase English letters.


### 解析


根据题意，给定两个字符串 s 和 t，如果 s 是 t 的子序列，则返回 true，否则返回 false。

字符串的子序列是由原始字符串通过删除一些（可以不删除）字符而不干扰剩余字符的相对位置而形成的新字符串。 （即，“ace”是“abcde”的子序列，而“aec”不是）。

其实这道题目就是考察怎么判断一个字符串是另一个字符串的子序列，而且题目也给出来子序列的定义，因为这是一道 Eazy 难度的题目，限制条件也肯定宽松，s 的长度最大为 100 ， t 的长度最大为 10000 ，我们可以直接以 s 为基准从左到右遍历每一个字符，同时我们从左到右也遍历 t 中的每一个字符，如果能在相对位置不变的情况下找到相等的字符 ，我们就返回 True ，否则我们返回 False 。

其实这道题还可以用 python 的内置函数 index 去找索引位置，那样代码上会更简单一点。

时间复杂度为 O(s+t) ，空间复杂度为 O(1)。

### 解答
				
	class Solution(object):
	    def isSubsequence(self, s, t):
	        """
	        :type s: str
	        :type t: str
	        :rtype: bool
	        """
	        s_i = t_i = 0
	        while s_i < len(s):
	            while t_i < len(t) and s[s_i]!=t[t_i]:
	                t_i += 1
	            if t_i >= len(t):
	                return False
	            s_i += 1
	            t_i += 1
	        return True

            	      
			
### 运行结果


	Runtime: 33 ms, faster than 33.60% of Python online submissions for Is Subsequence.
	Memory Usage: 13.6 MB, less than 53.84% of Python online submissions for Is Subsequence.

### 原题链接



https://leetcode.com/problems/is-subsequence/


您的支持是我最大的动力
