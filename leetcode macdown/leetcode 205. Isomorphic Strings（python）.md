leetcode  205. Isomorphic Strings（python）

### 描述



Given two strings s and t, determine if they are isomorphic.

Two strings s and t are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

Example 1:

	Input: s = "egg", t = "add"
	Output: true

	
Example 2:

	Input: s = "foo", t = "bar"
	Output: false


Example 3:

	Input: s = "paper", t = "title"
	Output: true

	





Note:

	1 <= s.length <= 5 * 10^4
	t.length == s.length
	s and t consist of any valid ascii character.



### 解析


根据题意，给定两个字符串 s 和 t，确定它们是否结构相同的。如果可以替换 s 中的字符得到 t ，则两个字符串 s 和 t 是同构的。

所有出现的字符都必须替换为另一个字符，同时保留字符的顺序。 不允许两个不同种类的字符映射到同一种类字符，但一种字符可以映射成自己的种类。

其实说白了就是看两个字符串 s 和 t 是否在样式上是类似的。思路比较简单:

* 如果 s 和 t 的长度不相等，直接返回 False 
* 初始化一个字典 d 
* 遍历字符串 s ，如果字符 c 不在 d 中，将对应索引的 t[i] 赋值给 d[c] ，也就是记住 c 在 t 中对应的映射字符。如果此时判断 d[c] 不等于 t[i] ，那么说明 c 在 t 中此时对应的映射与已有的映射有冲突，违反题意，直接返回 False 
* 遍历结束之后，因为可能有“不同的字符映射到相同的字符上”这种错误情况，需要判断字符串 t 的集合长度是否与 d 的长度相等，如果相等说明是 s 和 t 是同构的直接返回 True ，如果不等说明有 s 和 t 不是同构的，返回 False 

这个 easy 级别的题害的我错了三次，归不得只有 41.6 的正确提交率，以后不能小看 easy 级别的题了，骄兵必败！！
### 解答
				

	class Solution(object):
	    def isIsomorphic(self, s, t):
	        """
	        :type s: str
	        :type t: str
	        :rtype: bool
	        """
	        if len(s) != len(t): return False
	        d = {}
	        for i, c in enumerate(s):
	            if c not in d:
	                d[c] = t[i]
	            if d[c] != t[i]:
	                return False
	        return len(set(t))==len(d)
            	      
			
### 运行结果

	Runtime: 24 ms, faster than 93.54% of Python online submissions for Isomorphic Strings.
	Memory Usage: 13.9 MB, less than 92.50% of Python online submissions for Isomorphic Strings.


原题链接：https://leetcode.com/problems/isomorphic-strings/



您的支持是我最大的动力
