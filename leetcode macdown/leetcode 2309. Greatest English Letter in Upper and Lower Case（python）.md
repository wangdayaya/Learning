leetcode  2309. Greatest English Letter in Upper and Lower Case（python）




### 描述

Given a string of English letters s, return the greatest English letter which occurs as both a lowercase and uppercase letter in s. The returned letter should be in uppercase. If no such letter exists, return an empty string.

An English letter b is greater than another letter a if b appears after a in the English alphabet.



Example 1:

	Input: s = "lEeTcOdE"
	Output: "E"
	Explanation:
	The letter 'E' is the only letter to appear in both lower and upper case.

	
Example 2:

	Input: s = "arRAzFif"
	Output: "R"
	Explanation:
	The letter 'R' is the greatest letter to appear in both lower and upper case.
	Note that 'A' and 'F' also appear in both lower and upper case, but 'R' is greater than 'F' or 'A'.


Example 3:

	Input: s = "AbCdEfGhIjK"
	Output: ""
	Explanation:
	There is no letter that appears in both lower and upper case.


	





Note:


	1 <= s.length <= 1000
	s consists of lowercase and uppercase English letters.

### 解析

根据题意，给定一串英文字母 s ，返回在 s 中同时出现其大小写形式的英文字母。 返回的字母应为大写。 如果不存在这样的字母，则返回一个空字符串。如果 b 在英文字母表中出现在 a 之后，则英文字母 b 大于另一个字母 a 。

这道题其实考查的就是字母的字典顺序，我们逆序遍历 26 个小写字母，如果其在 s 中出现过，并且其大写字母在 s 中出现过，那么就说明该字母是字典序最大且其大小写都存在于 s 中的字母，我们返回其大写字母。

时间复杂度为 O(26\*N) ，空间复杂度为 O(1) 。


### 解答
				

	class Solution(object):
	    def greatestLetter(self, s):
	        """
	        :type s: str
	        :rtype: str
	        """
	        for i in range(ord('a')+25, ord('a')-1, -1):
	            c = chr(i)
	            if c in s and c.upper() in s:
	                return c.upper()
	        return ''
            	      
			
### 运行结果

	
	113 / 113 test cases passed.
	Status: Accepted
	Runtime: 17 ms
	Memory Usage: 13.5 MB


### 原题链接


https://leetcode.com/contest/weekly-contest-298/problems/greatest-english-letter-in-upper-and-lower-case/

您的支持是我最大的动力
