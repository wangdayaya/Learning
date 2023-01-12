leetcode  2351. First Letter to Appear Twice（python）




### 描述
Given a string s consisting of lowercase English letters, return the first letter to appear twice.

Note:

A letter a appears twice before another letter b if the second occurrence of a is before the second occurrence of b.
s will contain at least one letter that appears twice.




Example 1:

	Input: s = "abccbaacz"
	Output: "c"
	Explanation:
	The letter 'a' appears on the indexes 0, 5 and 6.
	The letter 'b' appears on the indexes 1 and 4.
	The letter 'c' appears on the indexes 2, 3 and 7.
	The letter 'z' appears on the index 8.
	The letter 'c' is the first letter to appear twice, because out of all the letters the index of its second occurrence is the smallest.

	
Example 2:

	Input: s = "abcdd"
	Output: "d"
	Explanation:
	The only letter that appears twice is 'd' so we return 'd'.


Example 3:


	2 <= s.length <= 100
	s consists of lowercase English letters.
	s has at least one repeated letter.


Note:

	2 <= s.length <= 100
	s consists of lowercase English letters.
	s has at least one repeated letter.

### 解析

根据题意，给定一个由小写英文字母组成的字符串  s，返回第一个出现两次的字母。

这道题其实很简单，只需要使用字典对字符进行计数，只要有最先出现两次的字母就是它。

时间复杂度为 O(N) ，空间复杂度为 O(26)。

### 解答

	class Solution:
	    def repeatedCharacter(self, s: str) -> str:
	        d = {}
	        for c in s:
	            if c not in d:
	                d[c] = 1
	            else:
	                d[c] += 1
	                if d[c] == 2:
	                    return c

### 运行结果

	
	92 / 92 test cases passed.
	Status: Accepted
	Runtime: 31 ms
	Memory Usage: 14 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-303/problems/first-letter-to-appear-twice/


您的支持是我最大的动力
