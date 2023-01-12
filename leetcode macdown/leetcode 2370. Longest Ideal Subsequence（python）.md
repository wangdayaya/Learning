leetcode 2370. Longest Ideal Subsequence （python）




### 描述

You are given a string s consisting of lowercase letters and an integer k. We call a string t ideal if the following conditions are satisfied:

* t is a subsequence of the string s.
* The absolute difference in the alphabet order of every two adjacent letters in t is less than or equal to k.

Return the length of the longest ideal string.A subsequence is a string that can be derived from another string by deleting some or no characters without changing the order of the remaining characters. Note that the alphabet order is not cyclic. For example, the absolute difference in the alphabet order of 'a' and 'z' is 25, not 1.



Example 1:

	Input: s = "acfgbd", k = 2
	Output: 4
	Explanation: The longest ideal string is "acbd". The length of this string is 4, so 4 is returned.
	Note that "acfgbd" is not ideal because 'c' and 'f' have a difference of 3 in alphabet order.

	
Example 2:

	Input: s = "abcd", k = 3
	Output: 4
	Explanation: The longest ideal string is "abcd". The length of this string is 4, so 4 is returned.





Note:

	1 <= s.length <= 10^5
	0 <= k <= 25
	s consists of lowercase English letters.


### 解析

根据题意，给定一个由小写字母组成的字符串 s 和整数 k 。 如果满足以下条件，我们称字符串 t 为理想的：

* t 是字符串 s 的子序列 
* t 中每两个相邻字母的字母顺序绝对差小于等于 k

返回最长的理想字符串的长度。子序列是可以通过删除一些或不删除字符而不改变剩余字符的顺序从另一个字符串派生的字符串。 请注意，“a” 和 “z” 的字母顺序的绝对差异是 25 ，而不是 1 。

这道题其实考察的就是动态规划，我们可以定义一个动态变化的列表 L 长度为 150 （因为 z 的 ASCII 码为 122 ，k 的最大值为 25 ，所以刚好不超过 150 ），L[i] 表示以  ASCII 码为 i 的字母为结尾的最长子序列长度，我们遍历 s 中的每个字符 s[i] 也就是 c ，我们可以找出在 s[i] 前后相差 k 个距离的所有字符，遍历这里面的每一个字符 t 我们可以找出 s 的前 i 个字符中以 t 结尾的能够形成的最长子序列长度，然后我们找出其中的最大值加上本身 s[i] 的长度 1 ，即可更新 L[ord[s[i]]] ，遍历 s 结束我们返回 L 中的最大值即可。

时间复杂度为 O(N\*2k)，空间复杂度为 O(150) ，也就是 O(1) 。

### 解答

	class Solution(object):
	    def longestIdealString(self, s, k):
	        """
	        :type s: str
	        :type k: int
	        :rtype: int
	        """
	        L = [0] * 150
	        for c in s:
	            left = ord(c)-k
	            right = ord(c)+k+1
	            tmp = 0
	            for i in range(left, right):
	                tmp = max(tmp, L[i])
	            L[ord(c)] = tmp + 1
	        return max(L)
	


### 运行结果

	85 / 85 test cases passed.
	Status: Accepted
	Runtime: 2830 ms
	Memory Usage: 14.6 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-305/problems/longest-ideal-subsequence/


您的支持是我最大的动力
