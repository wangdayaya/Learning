leetcode  1143. Longest Common Subsequence（python）




### 描述

Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.

 



Example 1:

	Input: text1 = "abcde", text2 = "ace" 
	Output: 3  
	Explanation: The longest common subsequence is "ace" and its length is 3.

	
Example 2:


	Input: text1 = "abc", text2 = "abc"
	Output: 3
	Explanation: The longest common subsequence is "abc" and its length is 3.

Example 3:

	Input: text1 = "abc", text2 = "def"
	Output: 0
	Explanation: There is no such common subsequence, so the result is 0.

	



Note:

	1 <= text1.length, text2.length <= 1000
	text1 and text2 consist of only lowercase English characters.


### 解析

根据题意，给定两个字符串 text1 和 text2 ，返回它们最长的公共子序列的长度。 如果没有公共子序列，则返回 0 。

这种求 LCS 的题目是典型的使用动态规划解决的类型题，我们定义二维数组 dp ，dp[i][j] 表示在 text1 前 i 个子字符串中和 text2 前 j 个子字符串中 LCS 的长度。如果 text1[i-1] == text2[j-1] 说明有相同的字符出现，更新 dp 为 dp[i-1][j-1] + 1 ，否则说明没有相同的字符出现，更新 dp 为 max(dp[i-1][j], dp[i][j-1]) ，遍历完全部两个字符串我们直接返回 dp[-1][-1] 就是最后的结果。

时间复杂度为 O(MN)，空间复杂度为 O(MN) 。

### 解答
				

	class Solution(object):
	    def longestCommonSubsequence(self, text1, text2):
	        """
	        :type text1: str
	        :type text2: str
	        :rtype: int
	        """
	        M, N = len(text1), len(text2)
	        dp = [[0] * (N+1) for _ in range(M+1)]
	        for i in range(1, M+1):
	            for j in range(1, N+1):
	                if text1[i-1] == text2[j-1]:
	                    dp[i][j] = dp[i-1][j-1] + 1
	                else:
	                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
	        return dp[-1][-1]
            	      
			
### 运行结果

	Runtime: 269 ms, faster than 87.84% of Python online submissions for Longest Common Subsequence.
	Memory Usage: 21.7 MB, less than 68.56% of Python online submissions for Longest Common Subsequence.


### 原题链接


https://leetcode.com/problems/longest-common-subsequence/


您的支持是我最大的动力
