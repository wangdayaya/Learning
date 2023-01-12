leetcode  14. Longest Common Prefix（python）

### 回顾

* 一刷：https://juejin.cn/post/6989147514939277349
* 二刷：https://juejin.cn/post/6989442774919544846
* 三刷：https://juejin.cn/post/6990174541284638734

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

之前的几种解法，我们使用了水平扫描法、垂直扫描法、二分查找法、内置函数法，这次我们使用一种新的解决思路，利用分而治之的方法，我们仔细理解题目发现，如果要找最长公共前缀字符串 LCP ，那么假设最终的结果记为变量 result ，strs 列表中的字符串个数为 n ，那么 

* result = LCP(S<sub>0</sub>，S<sub>n-1</sub>) = LCP(LCP(S<sub>0</sub>，S<sub>k</sub>)，LCP(S<sub>k+1</sub>，S<sub>n-1</sub>)) 

这个公式可以一直细化下去进行，无限套娃。通过观察上面分而治之的公式，我们其实可以看出来任意的一个子列表的的结果都可以进行转换成以下的形式：

* LCP(S<sub>i</sub>，S<sub>j</sub>) = LCP(LCP(S<sub>i</sub>，S<sub>mid</sub>)，LCP(S<sub>mid+1</sub>，S<sub>j</sub>)) ，其中 mid 为 (i+j)//2 

通过将一个大问题分解成许多相同的小问题，这就是分而治之的思想，在这里我们可以将 strs 向下分解成一对一对的字符串，然后将每对得到的 LCP 再向上合并求 LCP ，直到最后将所有的答案都合并为一个 LCP 即为最终的结果。因此这里定了一个 divide 函数将 strs 不断的分解成一对对的字符串对，LCP 函数来求解两个字符串的最长公共前缀。

从结果我们可以看出来分而治之的速度比较慢，而且所占内存也比较大。

### 解答
					
	class Solution(object):
	    def longestCommonPrefix(self, strs):
	        """
	        :type strs: List[str]
	        :rtype: str
	        """
	        def LCP(a, b):
	            index = 0
	            for i in range(min(len(a), len(b))):
	                if a[i] != b[i]:
	                    break
	                index += 1
	            return a[:index]
	
	
	        def divide(strs, l, r):
	            if l == r:
	                return strs[l]
	            m = (l + r) // 2
	            a = divide(strs, l, m)
	            b = divide(strs, m + 1, r)
	            return LCP(a, b)
	
	
	        if not strs: return ""
	        return divide(strs, 0, len(strs) - 1)	        
            	      
			
### 运行结果

	Runtime: 32 ms, faster than 20.85% of Python online submissions for Longest Common Prefix.
	Memory Usage: 13.7 MB, less than 33.48% of Python online submissions for Longest Common Prefix.




原题链接：https://leetcode.com/problems/longest-common-prefix/



您的支持是我最大的动力
