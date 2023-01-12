leetcode  1638. Count Substrings That Differ by One Character（python）

### 描述

Given two strings s and t, find the number of ways you can choose a non-empty substring of s and replace a single character by a different character such that the resulting substring is a substring of t. In other words, find the number of substrings in s that differ from some substring in t by exactly one character.

For example, the underlined substrings in "computer" and "computation" only differ by the 'e'/'a', so this is a valid way.

Return the number of substrings that satisfy the condition above.

A substring is a contiguous sequence of characters within a string.



Example 1:

	Input: s = "aba", t = "baba"
	Output: 6
	Explanation: The following are the pairs of substrings from s and t that differ by exactly 1 character:
	("aba", "baba")
	("aba", "baba")
	("aba", "baba")
	("aba", "baba")
	("aba", "baba")
	("aba", "baba")
	The underlined portions are the substrings that are chosen from s and t.

	
Example 2:

	Input: s = "ab", t = "bb"
	Output: 3
	Explanation: The following are the pairs of substrings from s and t that differ by 1 character:
	("ab", "bb")
	("ab", "bb")
	("ab", "bb")
	​​​​The underlined portions are the substrings that are chosen from s and t.


Example 3:


	Input: s = "a", t = "a"
	Output: 0
	
Example 4:

	
	Input: s = "abe", t = "bbc"
	Output: 10
	



Note:

	1 <= s.length, t.length <= 100
	s and t consist of lowercase English letters only.


### 解析


根据题意，给出了两个字符串 s 和 t ，让我们找出在 s 中的子字符串与 t 中的子字符串仅相差一个字符的子串数的组合对数。

当然你了最简单的就是暴力算法，题目中给出了的 s 和 t 的长度都在 100 以内，用暴力的话也是可以求解的。无非就是多重循环来找符和题意的组合数。


### 解答
				
	class Solution(object):
	    def countSubstrings(self, s, t):
	        """
	        :type s: str
	        :type t: str
	        :rtype: int
	        """
	        result = 0
	        m = len(s)
	        n = len(t)
	        for i in range(1, m + 1):
	            for j in range(m - i + 1):
	                for k in range(n - i + 1):
	                    count = 0
	                    for x, y in zip(s[j:j + i], t[k:k + i]):
	                        if x != y: count += 1
	                        if count > 1: break
	                    if count == 1:
	                        result += 1
	        return result

            	      
			
### 运行结果


	Runtime: 3152 ms, faster than 6.25% of Python online submissions for Count Substrings That Differ by One Character.
	Memory Usage: 13.3 MB, less than 90.63% of Python online submissions for Count Substrings That Differ by One Character.


### 解析

另外看了大佬的解读视频，可以用动态规划来解题。[视频链接](https://www.bilibili.com/video/av627645231/)，看大佬的精彩解答，并且有一把梭哈代码运行，速度即可赶超 100% 的优秀操作



### 解答

	class Solution(object):
	    def countSubstrings(self, s, t):
	        """
	        :type s: str
	        :type t: str
	        :rtype: int
	        """
	        dp_l = [[0]*105 for i in range(105)] 
	        dp_r = [[0]*105 for i in range(105)] 
	        result = 0
	        m = len(s)
	        n = len(t)
	        s = '#' + s + '#'
	        t = '#' + t + '#'
	        for i in range(1, m+1):
	            for j in range(1, n+1):
	                if s[i]==t[j]:
	                    dp_l[i][j] = dp_l[i-1][j-1] + 1
	                else:
	                    dp_l[i][j] = 0
	        
	        for i in range(m, 0, -1):
	            for j in range(n, 0, -1):
	                if s[i]==t[j]:
	                    dp_r[i][j] = dp_r[i+1][j+1] + 1
	                else:
	                    dp_r[i][j] = 0
	                     
	            
	        for i in range(1, m+1):
	            for j in range(1, n+1):
	                if s[i]!=t[j]:
	                    result += (dp_l[i-1][j-1]+1) * (dp_r[i+1][j+1]+1)
	        return result
	                


### 运行结果

	Runtime: 64 ms, faster than 65.63% of Python online submissions for Count Substrings That Differ by One Character.
	Memory Usage: 13.7 MB, less than 9.38% of Python online submissions for Count Substrings That Differ by One Character.


原题链接：https://leetcode.com/problems/count-substrings-that-differ-by-one-character/



您的支持是我最大的动力
