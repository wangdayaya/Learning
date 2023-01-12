leetcode  28. Implement strStr()（python）

### 描述


Implement strStr().

Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

**Clarification**:

What should we return when needle is an empty string? This is a great question to ask during an interview.

For the purpose of this problem, we will return 0 when needle is an empty string. This is consistent to C's strstr() and Java's indexOf().

Example 1:

	Input: haystack = "hello", needle = "ll"
	Output: 2

	
Example 2:


	Input: haystack = "aaaaa", needle = "bba"
	Output: -1

Example 3:

	Input: haystack = "", needle = ""
	Output: 0

	


Note:

	0 <= haystack.length, needle.length <= 5 * 10^4
	haystack and needle consist of only lower-case English characters.


### 解析

根据题意，就是在 haystack 中找到第一次出现 needle 的索引，如果没有则返回 -1 ，如果 needle 为空字符串，则返回 0 。可以直接使用暴力的方法：

* 当 needle 为空字符串，直接返回 0；
* 遍历 haystack ，当以索引 i 为开头的字符串与 needle 相等的时候，即 haystack[i:i+len(needle)] == needle ，返回 i ；
* 遍历结束，直接返回 -1 ；



我们看到上面 note 中的限制条件，haystack 和 needle 的长度最大有 50000 ，用暴力方法解决的话，本以为不会通过，没想到可以通过。其实还是不推荐，当长度更长的字符串出现就肯定会内存爆炸。

### 解答
				
	class Solution(object):
	    def strStr(self, haystack, needle):
	        """
	        :type haystack: str
	        :type needle: str
	        :rtype: int
	        """
	        if not needle:return 0
	        for i in range(len(haystack)-len(needle)+1):
	            if haystack[i:i+len(needle)] == needle:
	                return i
	        return -1
            	      
			
### 运行结果
	Runtime: 292 ms, faster than 23.53% of Python online submissions for Implement strStr().
	Memory Usage: 14.8 MB, less than 17.25% of Python online submissions for Implement strStr().

### 解析

既然暴力这种奇技淫巧都能拿出来，那么肯定可以想到用 python 的内置函数了，find 函数就可以满足题目要求，如果 needle 存在于  haystack.find ，那么就返回第一次出现的索引，否则就返回 -1 。但是这种方法我仍然不推荐，太 low ，体现不出一点技术含量。

### 解答
				
	class Solution(object):
	    def strStr(self, haystack, needle):
	        """
	        :type haystack: str
	        :type needle: str
	        :rtype: int
	        """
	        if not needle:return 0
	        return haystack.find(needle);
	    
	
	            	      
			
### 运行结果

	Runtime: 20 ms, faster than 69.53% of Python online submissions for Implement strStr().
	Memory Usage: 13.7 MB, less than 69.35% of Python online submissions for Implement strStr().

原题链接：https://leetcode.com/problems/implement-strstr/



您的支持是我最大的动力
