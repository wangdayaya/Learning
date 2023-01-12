leetcode  14. Longest Common Prefix（python）

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

根据题意，就是找出 strs 列表中所有字符串的公共最长前缀字符串（LCP），如果没有 LCP 则直接返回空字符串。我们可以按照题意，按照从左到右的扫描 strs 的方法，先找出第一个字符串 a 与第二字符串 b 的 LCP  ：

* while 循环判断，如果 b 不是以 a 为前缀，则将 result[:-1] 赋值给 result ；
* 判断 result 如果为空则直接返回空字符串；
* 如果 while 循环到 b 是以 a 为前缀的，跳出 while 循环

然后按照相同的算法，找出 result 和第三个字符串的 LCP  为 result ，然后找出 result 和第四个字符串的 LCP 为 result ，等等，遍历完 strs 中所有的字符串，即可得到答案，具体过程见代码。


### 解答
				

	class Solution(object):
	    def longestCommonPrefix(self, strs):
	        """
	        :type strs: List[str]
	        :rtype: str
	        """
	        if not strs: return ""
	        result = strs[0]
	        for i, v in enumerate(strs[1:]):
	            while not v.startswith(result):
	                result = result[:-1]
	                if not result: return ""
	        return result
            	      
			
### 运行结果

	Runtime: 16 ms, faster than 94.83% of Python online submissions for Longest Common Prefix.
	Memory Usage: 13.8 MB, less than 33.28% of Python online submissions for Longest Common Prefix.


### 解析

上面的可以看作是水平从左到右遍历每个字符串寻找 LCP ，其实还可以“垂直”寻找 LCP ，先判断 所有 strs 字符串中的第一个字符是否相同，如果第一个字符相同，再判断所有 strs 字符串的第二个字符是否相同，依次进行下去，如果到第 n 个字符不相同的时候，则寻找 LCP 结束，直接返回已经得到的 LCP 即可，具体实现过程，如下代码所示。

### 解答


	class Solution(object):
	    def longestCommonPrefix(self, strs):
	        """
	        :type strs: List[str]
	        :rtype: str
	        """
	        if not strs: return ""
	        for i, c in enumerate(strs[0]):
	            for v in strs[1:]:
	                if len(v)==i or v[i] != c:
	                    return strs[0][:i]
	        return strs[0]
	            
### 运行结果

	Runtime: 16 ms, faster than 94.83% of Python online submissions for Longest Common Prefix.
	Memory Usage: 13.7 MB, less than 62.61% of Python online submissions for Longest Common Prefix.


原题链接：https://leetcode.com/problems/longest-common-prefix/



您的支持是我最大的动力
