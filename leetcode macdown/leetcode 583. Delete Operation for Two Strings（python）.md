leetcode  583. Delete Operation for Two Strings（python）




### 描述

Given two strings word1 and word2, return the minimum number of steps required to make word1 and word2 the same.

In one step, you can delete exactly one character in either string.





Example 1:

	Input: word1 = "sea", word2 = "eat"
	Output: 2
	Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".

	
Example 2:

	Input: word1 = "leetcode", word2 = "etco"
	Output: 4





Note:


	1 <= word1.length, word2.length <= 500
	word1 and word2 consist of only lowercase English letters.

### 解析

根据题意，给定两个字符串 word1 和 word2 ，返回使 word1 和 word2 相同所需的最小步数。在每一个步骤中，可以删除任一字符串中的一个字符。

这道题本质上就是在找最长公共子序列，如果没有基础可以做一道模版题 [1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/) 去练练手，再来做这个就得心应手了。

解决这种题就是使用典型的动态规划方法，我们定义二维数组 dp ，dp[i][j] 表示在 word1 前 i 个子字符串中和 word2 前 j 个子字符串中 LCS 的长度。如果 word1[i-1] == word2[j-1] 说明有相同的字符出现，更新 dp 为 dp[i-1][j-1] + 1 ，否则说明没有相同的字符出现，更新 dp 为 max(dp[i-1][j], dp[i][j-1]) ，遍历完全部两个字符串我们直接返回 dp[-1][-1] 就是最后得到的 LCS 长度，最后返回 M+N-2*dp[-1][-1]  就是我们需要删掉的字符个数。

时间复杂度为 O(MN) ，空间复杂度为 O(MN) 。

### 解答
				

	class Solution(object):
	    def minDistance(self, word1, word2):
	        """
	        :type word1: str
	        :type word2: str
	        :rtype: int
	        """
	        M, N = len(word1), len(word2)
	        dp = [[0] * (N+1) for _ in range(M+1)]
	        for i in range(1, M+1):
	            for j in range(1, N+1):
	                if word1[i-1] == word2[j-1]:
	                    dp[i][j] = dp[i-1][j-1] + 1
	                else:
	                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
	        return M+N-2*dp[-1][-1]
            	      
			
### 运行结果

	Runtime: 301 ms, faster than 56.63% of Python online submissions for Delete Operation for Two Strings.
	Memory Usage: 15.7 MB, less than 59.64% of Python online submissions for Delete Operation for Two Strings.


### 原题链接

https://leetcode.com/problems/delete-operation-for-two-strings/

您的支持是我最大的动力
