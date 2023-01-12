leetcode  2301. Match Substring After Replacement（python）




### 描述

You are given two strings s and sub. You are also given a 2D character array mappings where mappings[i] = [oldi, newi] indicates that you may replace any number of oldi characters of sub with newi. Each character in sub cannot be replaced more than once.

Return true if it is possible to make sub a substring of s by replacing zero or more characters according to mappings. Otherwise, return false.

A substring is a contiguous non-empty sequence of characters within a string.



Example 1:

	Input: s = "fool3e7bar", sub = "leet", mappings = [["e","3"],["t","7"],["t","8"]]
	Output: true
	Explanation: Replace the first 'e' in sub with '3' and 't' in sub with '7'.
	Now sub = "l3e7" is a substring of s, so we return true.

	
Example 2:

	Input: s = "fooleetbar", sub = "f00l", mappings = [["o","0"]]
	Output: false
	Explanation: The string "f00l" is not a substring of s and no replacements can be made.
	Note that we cannot replace '0' with 'o'.


Example 3:

	Input: s = "Fool33tbaR", sub = "leetd", mappings = [["e","3"],["t","7"],["t","8"],["d","b"],["p","b"]]
	Output: true
	Explanation: Replace the first and second 'e' in sub with '3' and 'd' in sub with 'b'.
	Now sub = "l33tb" is a substring of s, so we return true.

	




Note:


	1 <= sub.length <= s.length <= 5000
	0 <= mappings.length <= 1000
	mappings[i].length == 2
	oldi != newi
	s and sub consist of uppercase and lowercase English letters and digits.
	oldi and newi are either uppercase or lowercase English letters or digits.

### 解析

根据题意，给出两个字符串 s 和 sub。 还给出一个 2D 字符数组 mappings ，其中 mappings[i] = [oldi, newi] 表示可以用 newi 替换 sub 的任意的 oldi 字符。 每次只能替换 sub 中的一个字符。

如果可以通过根据 mappings 替换零个或多个字符来使 sub 成为 s 的子字符串，则返回 true。 否则，返回 false 。 子字符串是字符串中连续的非空字符序列。

我们看这道题目的限制条件 ，s 和 sub 的长度最长为 5000 ，所以我们可以使用暴力的解法进行解决，一开始的时候我们先将所有的映射关系都放入 d 中，然后我们遍历 sub 中的每个位置，找出一个子字符串（和 sub 长度相等）和 sub 进行比对，如果成了就直接返回 True ，不成就去找下一个位置的子字符串继续和 sub 进行比对，直到最后如果没有结果直接返回 False 。

时间复杂度为 O(N ^ 2) ，空间复杂度为 O(N) 。


### 解答
				
	class Solution(object):
	    def matchReplacement(self, s, sub, mappings):
	        """
	        :type s: str
	        :type sub: str
	        :type mappings: List[List[str]]
	        :rtype: bool
	        """
	        d = collections.defaultdict(set)
	        for a, b in mappings:
	            d[a].add(b)
	        N = len(s)
	        M = len(sub)
	        for i in range(N - M + 1):
	            check = True
	            for a, b in zip(s[i:i + M], sub):
	                if a == b or a in d[b]:
	                    continue
	                else:
	                    check = False
	                    break
	            if check:
	                return True
	        return False

            	      
			
### 运行结果


	109 / 109 test cases passed.
	Status: Accepted
	Runtime: 5858 ms
	Memory Usage: 14.2 MB

### 原题链接


https://leetcode.com/contest/biweekly-contest-80/problems/match-substring-after-replacement/

您的支持是我最大的动力
