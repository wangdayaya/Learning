leetcode 1876. Substrings of Size Three with Distinct Characters（python）

### 描述

A string is good if there are no repeated characters.

Given a string s​​​​​, return the number of good substrings of length three in s​​​​​​.

Note that if there are multiple occurrences of the same substring, every occurrence should be counted.

A substring is a contiguous sequence of characters in a string.

 



Example 1:

	Input: s = "xyzzaz"
	Output: 1
	Explanation: There are 4 substrings of size 3: "xyz", "yzz", "zza", and "zaz". 
	The only good substring of length 3 is "xyz".

	
Example 2:

	
	Input: s = "aababcabc"
	Output: 4
	Explanation: There are 7 substrings of size 3: "aab", "aba", "bab", "abc", "bca", "cab", and "abc".
	The good substrings are "abc", "bca", "cab", and "abc".






Note:

	1 <= s.length <= 100
	s​​​​​​ consists of lowercase English letters.


### 解析


根据题意，就是在字符串 s 中，找出长度为 3 的没有重复字符的子字符串个数。只需要遍历 s ，每 3 个字符的集合长度为 3 结果计数加 1 ，遍历结束即可得到结果。

### 解答
				
	class Solution(object):
	    def countGoodSubstrings(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        if len(s) < 3:
	            return 0
	        N = len(s)
	        result = 0
	        for i in range(N - 3 + 1):
	            if len(set(s[i:i + 3])) == 3:
	                result += 1
	        return result

            	      
			
### 运行结果

	Runtime: 24 ms, faster than 56.22% of Python online submissions for Substrings of Size Three with Distinct Characters.
	Memory Usage: 13.6 MB, less than 33.64% of Python online submissions for Substrings of Size Three with Distinct Characters.

### 解析


因为是在字符串 s 中，找出长度为 3 的没有重复字符的子字符串个数。所以长度为 3 的子字符串可以两两字符进行比较，速度比用内置函数更快一点，如果 3 个字符户部巷等结果计数加 1 ，遍历结束即可得到结果。

运行结果表明速度确实更快，但是消耗内存更多，说明上面的内置函数 set 在内存优化方面做了工作。

### 解答
	class Solution(object):
	    def countGoodSubstrings(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        if len(s)<3:
	            return 0
	        result = 0
	        for i in range(1, len(s)-1):
	            if s[i-1]!=s[i] and s[i-1]!=s[i+1] and s[i]!=s[i+1]:
	                result += 1
	        return result
	        
### 运行结果
	
	Runtime: 16 ms, faster than 93.09% of Python online submissions for Substrings of Size Three with Distinct Characters.
	Memory Usage: 13.6 MB, less than 33.64% of Python online submissions for Substrings of Size Three with Distinct Characters.
	
### 解析
将上面的代码过程还能进行简化，使用 sum 内置函数来计数。

### 解答

	class Solution(object):
	    def countGoodSubstrings(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        if len(s) < 3:
	            return 0
	        return sum(s[i] != s[i - 1] and s[i] != s[i + 1] and s[i - 1] != s[i + 1] for i in range(1, len(s)-1))
	        
### 运行结果

	Runtime: 24 ms, faster than 56.22% of Python online submissions for Substrings of Size Three with Distinct Characters.
	Memory Usage: 13.6 MB, less than 33.64% of Python online submissions for Substrings of Size Three with Distinct Characters.
	
原题链接：https://leetcode.com/problems/substrings-of-size-three-with-distinct-characters/



您的支持是我最大的动力
