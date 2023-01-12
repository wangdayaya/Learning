leetcode  2311. Longest Binary Subsequence Less Than or Equal to K（python）




### 描述

You are given a binary string s and a positive integer k. Return the length of the longest subsequence of s that makes up a binary number less than or equal to k.

Note:

* The subsequence can contain leading zeroes.
* The empty string is considered to be equal to 0.
* A subsequence is a string that can be derived from another string by deleting some or no characters without changing the order of the remaining characters.



Example 1:


	Input: s = "1001010", k = 5
	Output: 5
	Explanation: The longest subsequence of s that makes up a binary number less than or equal to 5 is "00010", as this number is equal to 2 in decimal.
	Note that "00100" and "00101" are also possible, which are equal to 4 and 5 in decimal, respectively.
	The length of this subsequence is 5, so 5 is returned.
	
Example 2:

	Input: s = "00101001", k = 1
	Output: 6
	Explanation: "000001" is the longest subsequence of s that makes up a binary number less than or equal to 1, as this number is equal to 1 in decimal.
	The length of this subsequence is 6, so 6 is returned.





Note:

	1 <= s.length <= 1000
	s[i] is either '0' or '1'.
	1 <= k <= 10^9


### 解析


根据题意，给定一个二进制字符串 s 和一个正整数 k 。返回组成小于或等于 k 的二进制数的 s 的最长子序列的长度。

需要注意的有：

* 子序列可以包含前导零
* 空字符串被认为等于 0
* 子序列是一个字符串，可以通过对一个字符串删除一些字符或不删除字符而不改变剩余字符的顺序所产生的字符序列

这道题考查的是贪心思想，假如我们现在有一个长度为 1 的二进制字符串，如果在其左边加 0 不会改变大小，但是会增加长度，这符合题目要找最长子序列的要求，所有尽可能在左边加 0 ，在其左边加 1 会使其原数大小乘二，所以我们要让 1 尽量靠右，这样才能在左边不断加 1 找出更大的数字，题目要求找一个长度最大同时又不超过 k 的二进制子序列，我们可以从后往前遍历 s ，先找出刚好不超过 k 的二进制字符串，然后再加上其左边的所有 0 即为最后的结果长度。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution(object):
	    def longestSubsequence(self, s, k):
	        """
	        :type s: str
	        :type k: int
	        :rtype: int
	        """
	        N = len(s)
	        for i in range(1, N+1):
	            if s[-i] == '1':
	                a = int(s[-i:], 2)
	                if a > k:
	                    return s[:-i+1].count('0') + (i-1)
	        return len(s)
            	      
			
### 运行结果


	
	153 / 153 test cases passed.
	Status: Accepted
	Runtime: 46 ms
	Memory Usage: 13.6 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-298/problems/longest-binary-subsequence-less-than-or-equal-to-k/


您的支持是我最大的动力
