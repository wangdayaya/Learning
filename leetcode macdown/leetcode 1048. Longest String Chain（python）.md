leetcode  1048. Longest String Chain（python）




### 描述

You are given an array of words where each word consists of lowercase English letters.

wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA without changing the order of the other characters to make it equal to wordB.

* For example, "abc" is a predecessor of "abac", while "cba" is not a predecessor of "bcad".

A word chain is a sequence of words [word1, word2, ..., wordk] with k >= 1, where word1 is a predecessor of word2, word2 is a predecessor of word3, and so on. A single word is trivially a word chain with k == 1.

Return the length of the longest possible word chain with words chosen from the given list of words.



Example 1:

	Input: words = ["a","b","ba","bca","bda","bdca"]
	Output: 4
	Explanation: One of the longest word chains is ["a","ba","bda","bdca"].

	
Example 2:


	Input: words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]
	Output: 5
	Explanation: All the words can be put in a word chain ["xb", "xbc", "cxbc", "pcxbc", "pcxbcf"].

Example 3:

	Input: words = ["abcd","dbqca"]
	Output: 1
	Explanation: The trivial word chain ["abcd"] is one of the longest word chains.
	["abcd","dbqca"] is not a valid word chain because the ordering of the letters is changed.

	



Note:

	1 <= words.length <= 1000
	1 <= words[i].length <= 16
	words[i] only consists of lowercase English letters.


### 解析


根据题意，给定一个单词数组，其中每个单词由小写英文字母组成。

wordA 是 wordB 的前身当且仅当我们可以在 wordA 的任意位置准确插入一个字母而不改变其他字符的顺序以使其等于 wordB。

* 例如，“abc”是“abac”的前身，而“cba”不是“bcad”的前身。

一个词链是一个词序列 [word1, word2, ..., wordk] ，其中 k >= 1，其中 word1 是 word2 的前身，word2 是 word3 的前身，以此类推。返回从给定单词列表中选择的单词的最长可能单词链的长度。

这是一个典型的动态规划题目，因为这条链有长度限制，所以我们先将  words 按照顺序进行升序排序，这样我们就能在长度限制的情况下，使用两重循环挨个找子数组能构成的最长的单词链长度。我们定义一个一维数组 dp ，dp[i] 表示的是以索引 i 单词为末尾的单词链最长的长度，如果两个单词 words[j] 和 words[i] 符合题意，那么就更新 dp[i]=  max(dp[i], dp[j]+1) 。遍历结束直接返回 dp 中的最大值即可。

时间复杂度为 O(16\*N^2) ，空间复杂度为 O(N^2) 。


### 解答
				
	class Solution(object):
	    def longestStrChain(self, words):
	        """
	        :type words: List[str]
	        :rtype: int
	        """
	        def check(s1, s2):
	            M, N = len(s1), len(s2)
	            if M + 1 != N:
	                return False
	            i, j = 0, 0
	            while i < M and j < N:
	                if s1[i] == s2[j]: i += 1
	                j += 1
	            return i == M
	
	        words.sort(key=lambda x: (len(x), x))
	        N = len(words)
	        dp = [1] * N  
	        for i in range(N):
	            for j in range(i):
	                if check(words[j], words[i]):
	                    dp[i] = max(dp[i], dp[j]+1)
	        return max(dp)

            	      
			
### 运行结果


	Runtime: 3021 ms, faster than 5.56% of Python online submissions for Longest String Chain.
	Memory Usage: 14.2 MB, less than 24.50% of Python online submissions for Longest String Chain.

### 原题链接


https://leetcode.com/problems/longest-string-chain/


您的支持是我最大的动力
