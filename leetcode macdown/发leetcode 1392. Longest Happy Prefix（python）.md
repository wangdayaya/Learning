leetcode  1392. Longest Happy Prefix（python）

### 描述


A string is called a happy prefix if is a non-empty prefix which is also a suffix (excluding itself).

Given a string s, return the longest happy prefix of s. Return an empty string "" if no such prefix exists.




Example 1:

	Input: s = "level"
	Output: "l"
	Explanation: s contains 4 prefix excluding itself ("l", "le", "lev", "leve"), and suffix ("l", "el", "vel", "evel"). The largest prefix which is also suffix is given by "l".

	
Example 2:

	Input: s = "ababab"
	Output: "abab"
	Explanation: "abab" is the largest prefix which is also suffix. They can overlap in the original string.


Example 3:


	Input: s = "leetcodeleet"
	Output: "leet"
	
Example 4:


	Input: s = "a"
	Output: ""
	



Note:

	1 <= s.length <= 10^5
	s contains only lowercase English letters.


### 解析

根据题意，我们可以知道这道题就是在求最长公共前后缀的问题（还什么最长欢乐前缀，我做题的时候可是一点都不欢乐）。

**举例**：首先我们要知道什么是前缀和后缀，意思就不解释了直接用例子一中的 “level” 为例，它的所有前缀有  ("l", "le", "lev", "leve") ，可以看到除了自身字符串之外，是个“前缀”就合法，另外所有后缀有  ("l", "el", "vel", "evel") ，可以看到除了自身字符串之外，是个“后缀”就合法。其实就是字面的意思，在座的各位高质量 XDM 应该都能理解。

**目标**：就是在给定的 s 的基础上，找出最长的既是前缀又是后缀的字符串，如果 s 是空字符串或者长度为 1 ，直接返回空字符串就行。

**思路**：这种题我们最直接想到的就是暴力解决，就是按照题意，找出所有的前缀和后缀，然后比较之后找出最长的公共前后缀字符串即可。因为限制条件明显 s 的长度可能为十万，那结果可想而知肯定是超时。


### 解答
				

	class Solution(object):
	    def longestPrefix(self, s):
	        """
	        :type s: str
	        :rtype: str
	        """
	        if not s or len(s)==1:return ""
	        for i in range(len(s)-1, 0, -1):
	            a = s[:i]
	            b = s[-i:]
	            if a == b:
	                return s[:i]
	        return ""
            	      
			
### 运行结果

	Time Limit Exceeded

### 解析

另外我们学过算法的 XDM ，或多或少都听说过大名鼎鼎的 KMP 算法，也就是在字符串中快速匹配子字符串的算法。这个算法中的核心就是寻找最长公共前后缀，所以我们可以用大神们改进的算法来解决这个问题。主要思路是定义了一个 next 列表，next[j] 表示存放的是模式串 P 中 P[0] 到 P[j-1] 这个子串中，最长公共前后缀的长度。具体细节比较复杂，bebug 两次代码就可以理解了，或者可以参考大佬的解释：https://blog.csdn.net/weixin_39561100/article/details/80822208


### 解答
				

	class Solution(object):
	    def longestPrefix(self, s):
	        """
	        :type s: str
	        :rtype: str
	        """
	        if not s or len(s)==1:return ""
	        index= 0
	        m = len(s)
	        next = [0] * m
	        i = 1
	        while i < m:
	            if (s[i] == s[index]):
	                next[i] = index + 1
	                index += 1
	                i += 1
	            elif (index != 0):
	                index = next[index - 1]
	            else:
	                next[i] = 0
	                i += 1
	        return s[:next[-1]]
			
### 运行结果

	Runtime: 240 ms, faster than 100.00% of Python online submissions for Longest Happy Prefix.
	Memory Usage: 18.2 MB, less than 66.67% of Python online submissions for Longest Happy Prefix.

原题链接：https://leetcode.com/problems/longest-happy-prefix/



您的支持是我最大的动力
