### 回顾

一刷解题思路见：https://juejin.cn/post/6989147514939277349

### 描述

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".



Example 1:

	Input: strs = ["flower","flow","flight"]
	Output: "fl"

	
Example 2:

	
	Input: strs = ["dog","racecar","car"]
	Output: ""
	Explanation: There is no common prefix among the input strings.





Note:

	1 <= strs.length <= 200
	0 <= strs[i].length <= 200
	strs[i] consists of only lower-case English letters.


### 解析

根据题意，就是找出 strs 列表中所有字符串的公共最长前缀字符串（LCP），如果没有 LCP 则直接返回空字符串。python 作为一门强大的语言，就是带着“人生苦短，我用 python ”的宗旨而出现，我们可以使用 python 的内置函数 zip 来解决这个 LCP 问题。将 strs 中的所有元素的第一位的字符组合成一个集合，然后判断集合的长度是否为一，如果为一则将该字符追加到结果 result 字符串后面，然后判断所有元素的第二位字符组合而成的集合长度是否为一，依次进行下去判断所有元素的第 n 个字符；如果当集合的长度不为一，则表示这些字符不全相等，直接返回 result 。



### 解答
				

	class Solution(object):
	    def longestCommonPrefix(self, strs):
	        """
	        :type strs: List[str]
	        :rtype: str
	        """
	        if not(strs): return ""
	        result = ""
	        for s in zip(*strs):
	            if len(set(s)) != 1: return result
	            result += s[0]
	        return result
	            
            	      
			
### 运行结果

		
	
	Runtime: 20 ms, faster than 81.21% of Python online submissions for Longest Common Prefix.
	Memory Usage: 13.5 MB, less than 88.38% of Python online submissions for Longest Common Prefix.

### 拓展


zip 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。举例：

	a = [1,2,3]
	b = [4,5,6]
	c = [4,5,6,7,8]

用法一：

	d = zip(a,b)              # 打包为元组的列表
	print([x for x in d])

打印：

	[(1, 4), (2, 5), (3, 6)]

用法二：

	e = zip(a,c)              # 元素个数与最短的列表一致
	print([x for x in e])

打印：
	
	[(1, 4), (2, 5), (3, 6)]

用法三：

	f = zip(*zip(a,b))        # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
	print([x for x in f])

打印：
	
	[(1, 2, 3), (4, 5, 6)]

原题链接：https://leetcode.com/problems/longest-common-prefix/



您的支持是我最大的动力