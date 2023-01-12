leetcode 5. Longest Palindromic Substring （python）




### 描述


Given a string s, return the longest palindromic substring in s.




Example 1:

	Input: s = "babad"
	Output: "bab"
	Explanation: "aba" is also a valid answer.

	
Example 2:


	Input: s = "cbbd"
	Output: "bb"

Example 3:


	1 <= s.length <= 1000
	s consist of only digits and English letters.
	



Note:

	1 <= s.length <= 1000
	s consist of only digits and English letters.

### 解析

根据题意， 给出一个字符串 s ，让我们找出最长的回文字符串。

这道题算是一到典型的回文字符串类型题，因为我们知道限制条件中 s 的长度最长为 1000 ，所以我们可以直接选择暴力的方法，也就是中间扩散法来找最长的回文字符串。根据回文字符串的特性，如果 s 的长度大于 0 ，那么其最小的回文字符串就为 1 ，也就是任意一个字符的长度。我们通过固定中间的一个字符来向两边扩散找最长的回文字符串，还要固定中间的两个相同字符来向两边扩散找最长的回文字符串，这样我们就能找到一个最长的回文字符串。

时间复杂度为 O(N^2) ，空间复杂度为 O(1) 。


### 解答
				

	class Solution(object):
	    def longestPalindrome(self, s):
	        """
	        :type s: str
	        :rtype: str
	        """
	        if len(s) <= 1: return s
	        N = len(s)
	        result = s[0]
	        tmp = 1
	        for i in range(N):
	            L, R = i - 1, i + 1
	            while 0 <= L < N and 0 <= R < N and s[L] == s[R]:
	                if R - L + 1 > tmp:
	                    tmp = R - L + 1
	                    result = s[L:R + 1]
	                L -= 1
	                R += 1
	            L, R = i, i + 1
	            while 0 <= L < N and 0 <= R < N and s[L] == s[R]:
	                if R - L + 1 > tmp:
	                    tmp = R - L + 1
	                    result = s[L:R + 1]
	                L -= 1
	                R += 1
	        return result
            	      
			
### 运行结果


	Runtime: 1446 ms, faster than 39.02% of Python online submissions for Longest Palindromic Substring.
	Memory Usage: 13.7 MB, less than 47.19% of Python online submissions for Longest Palindromic Substring.
	
### 解析

当然也可以使用动态规划来解决，根据回文字符串的特征，我们定义二维数组 dp ，dp[i][j] 表示的是在索引 i 到 j 的子字符串是不是回文字符串，转移方程就是：
	
	dp[i][j] = dp[i+1][j-1] (当 s[i]==s[j])

当然还需要注意一些边界条件，单个字符肯定是回文字符串，两个相连的字符就要看是否相等。

时间复杂度为 O(N^2) ，空间复杂度为 O(N^2) 。


### 解答
				

	class Solution(object):
	    def longestPalindrome(self, s):
	        """
	        :type s: str
	        :rtype: str
	        """
	        N = len(s)
	        if N == 1:
	            return s
	        start = 0
	        max_length = 1
	        dp = [[False] * N for _ in range(N)]
	        for i in range(N):
	            dp[i][i] = True
	        for L in range(2, N + 1):
	            for i in range(N):
	                j = L + i - 1
	                if j >= N:
	                    break
	                if s[i] != s[j]:
	                    dp[i][j] = False
	                else:
	                    if j - i <= 2:
	                        dp[i][j] = True
	                    else:
	                        dp[i][j] = dp[i + 1][j - 1]
	
	                if dp[i][j] and j - i + 1 > max_length:
	                    max_length = j - i + 1
	                    start = i
	        return s[start:start+max_length]
            	      
			
### 运行结果





### 原题链接



https://leetcode.com/problems/longest-palindromic-substring/


您的支持是我最大的动力
