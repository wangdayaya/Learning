leetcode  2255. Count Prefixes of a Given String（python）


这是第 77 场双周赛的第一题，难度 Eazy ，考察的是字符串的基本操作。

### 描述

You are given a string array words and a string s, where words[i] and s comprise only of lowercase English letters.

Return the number of strings in words that are a prefix of s.

A prefix of a string is a substring that occurs at the beginning of the string. A substring is a contiguous sequence of characters within a string.



Example 1:

	Input: words = ["a","b","c","ab","bc","abc"], s = "abc"
	Output: 3
	Explanation:
	The strings in words which are a prefix of s = "abc" are:
	"a", "ab", and "abc".
	Thus the number of strings in words which are a prefix of s is 3.

	
Example 2:

	Input: words = ["a","a"], s = "aa"
	Output: 2
	Explanation:
	Both of the strings are a prefix of s. 
	Note that the same string can occur multiple times in words, and it should be counted each time.




Note:


	1 <= words.length <= 1000
	1 <= words[i].length, s.length <= 10
	words[i] and s consist of lowercase English letters only.

### 解析

根据题意，给定一个字符串数组 words 和一个字符串 s ，其中 words[i] 和 s 仅由小写英文字母组成。返回单词中作为 s 前缀的字符串的数量。


这道题很明显就是考察一个基本的字符串操作，因为我用的是 python ，所以只需要对 words 中的每个元素 word 进行遍历判断即可，只要 s.startswith(word) 为 True ，就对计数器 result 加一，遍历结束后返回 result 即可。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。


### 解答
				
	class Solution(object):
	    def countPrefixes(self, words, s):
	        """
	        :type words: List[str]
	        :type s: str
	        :rtype: int
	        """
	        return sum(1 for word in words if s.startswith(word))
	        

            	      
			
### 运行结果

	123 / 123 test cases passed.
	Status: Accepted
	Runtime: 45 ms
	Memory Usage: 13.6 MB


### 原题链接



https://leetcode.com/contest/biweekly-contest-77/problems/count-prefixes-of-a-given-string/


您的支持是我最大的动力
