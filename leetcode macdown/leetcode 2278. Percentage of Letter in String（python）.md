leetcode  2278. Percentage of Letter in String（python）

这是第 294 场周赛的第一题，难度 Eazy ，主要考察的是简单的字符统计、小数计算以及小数取整等操作


### 描述

Given a string s and a character letter, return the percentage of characters in s that equal letter rounded down to the nearest whole percent.



Example 1:


	Input: s = "foobar", letter = "o"
	Output: 33
	Explanation:
	The percentage of characters in s that equal the letter 'o' is 2 / 6 * 100% = 33% when rounded down, so we return 33.
	
Example 2:

	
	Input: s = "jjjj", letter = "k"
	Output: 0
	Explanation:
	The percentage of characters in s that equal the letter 'k' is 0%, so we return 0.




Note:

	1 <= s.length <= 100
	s consists of lowercase English letters.
	letter is a lowercase English letter.


### 解析

根据题意，给定一个字符串 s 和一个字符 letter ，返回 s 中包含 letter 的数量占 s 长度的整数百分比。

其实这道题就是一个考察简单的字符串、小数计算以及小数取整的题目，没有什么好说的，直接使用内置函数 count 找 s 中 letter 的数量即可，然后计算所占长度的百分比取整即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				
	class Solution(object):
	    def percentageLetter(self, s, letter):
	        """
	        :type s: str
	        :type letter: str
	        :rtype: int
	        """
	        return int(1.0*s.count(letter)/len(s)*100)

            	      
			
### 运行结果

	
	85 / 85 test cases passed.
	Status: Accepted
	Runtime: 40 ms
	Memory Usage: 13.3 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-294/problems/percentage-of-letter-in-string/


您的支持是我最大的动力
