leetcode  14. Longest Common Prefix（python）


### 回顾

* 一刷:https://juejin.cn/post/6989147514939277349
* 二刷:https://juejin.cn/post/6989442774919544846


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

另外，还可以运用二分法的思路去找 LCP ：

* 首先对 strs 按照长度大小进行排序
* 然后找出长度最短的字符串 s 
* 接着对字符串 s 进行二分，中间点记为 middle
* 用 t 表示二分法找出来的字符串 s[right,left] 
* 然后遍历 strs 中的所有元素，判断是否它们都以 t 为 LCP ，如果都为 True 的话则 left = middle + 1 表示为了寻找 LCP 可以再向右多寻找一位字符，如果有一个为 False ，则 right = middle - 1 表示为了寻找 LCP 得向左减少一位字符，直到 left > right 

结合之前刷题的几种解法我们可以看到，这种方法的速度不是最快的，只能说是一般般，因为执行了开始的列表中的字符串的排序和后面的二分法查找，在内存消耗上和之前的几种解法相比，可是实实在在的倒数第一，主要还是上面的两个步骤对内存的消耗，另外就是使用了内置函数 startswith 。

总的来说这道题是一道经典题目，值得一刷在刷，其实就是在求最长公共前缀字符串，其解法也是多种多样，有的比较容易理解，实现也比较方便，有的不容易理解，实现也比较麻烦，是一道仁者见仁智者见智的题目，高手的解法自然是会比较花样百出，新手看了也有自己的简单解法，无非就是效率的问题。

### 解答
				
	class Solution(object):
	    def longestCommonPrefix(self, strs):
	        """
	        :type strs: List[str]
	        :rtype: str
	        """
	        def isLCP(strs, t):
	            for string in strs[1:]:
	                if not string.startswith(t):
	                    return False
	            return True
	
	        if not (strs): return ""
	        strs.sort(key=lambda x:len(x))
	        s = strs[0]
	        left = 0
	        right = len(s)
	        while left<=right:
	            middle = (left+right)//2
	            if isLCP(strs, s[:middle]):
	                left = middle + 1
	            else:
	                right = middle - 1
	        return s[:(left+right)//2]	            
            	      
			
### 运行结果

		

	Runtime: 20 ms, faster than 81.34% of Python online submissions for Longest Common Prefix.
	Memory Usage: 13.8 MB, less than 10.88% of Python online submissions for Longest Common Prefix.



原题链接：https://leetcode.com/problems/longest-common-prefix/



您的支持是我最大的动力
