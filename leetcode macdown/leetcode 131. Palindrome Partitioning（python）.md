leetcode  131. Palindrome Partitioning（python）

### 每日经典

《登鹳雀楼》 ——王之涣（唐）

白日依山尽，黄河入海流。

欲穷千里目，更上一层楼。

### 描述


Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.

A palindrome string is a string that reads the same backward as forward.




Example 1:

	Input: s = "aab"
	Output: [["a","a","b"],["aa","b"]]

	
Example 2:


	Input: s = "a"
	Output: [["a"]]




Note:


	1 <= s.length <= 16
	s contains only lowercase English letters.

### 解析


根据题意，给定一个字符串 s ，对 s 分区使得分区的每个子串都是一个回文。 返回 s 的所有可能的回文分区，回文字符串就是向后读取与向前读取相同的字符串。

这道题是典型的 DFS + 动态规划的题，非常经典。因为这道题要找出所有的组合，所以肯定是要暴力搜索的，所以先用动态规划数组 dp[i][j] 保存 s[i][j] 是否为回文字符串。然后使用 DFS 找出所有的组合即可。

### 解答
				

	import numpy as np
	class Solution(object):
	    def __init__(self):
	        self.dp = None
	        self.result = []
	    def partition(self, s):
	        """
	        :type s: str
	        :rtype: List[List[str]]
	        """
	        N = len(s)
	        self.dp = [[False]*N for _ in range(N)]
	        for i in range(N):
	            self.dp[i][i] = True
	        for i in range(N-1):
	            self.dp[i][i+1] = (s[i] == s[i+1])
	        for L in range(3, N+1):
	            for i in range(N-L+1):
	                j = i+L-1
	                if s[i]==s[j] and self.dp[i+1][j-1]:
	                    self.dp[i][j] = True
	        self.dfs(0, [], s, N)
	        return self.result
	    
	    def dfs(self, i, t, s, N):
	        if i == N:
	            self.result.append(t)
	            return 
	        for j in range(i, N):
	            if self.dp[i][j]:
	                self.dfs(j+1, t+[s[i:j+1]], s, N)
	        
	      	      
			
### 运行结果

	Runtime: 668 ms, faster than 49.52% of Python online submissions for Palindrome Partitioning.
	Memory Usage: 39.2 MB, less than 5.03% of Python online submissions for Palindrome Partitioning.



原题链接：https://leetcode.com/problems/palindrome-partitioning/



您的支持是我最大的动力
