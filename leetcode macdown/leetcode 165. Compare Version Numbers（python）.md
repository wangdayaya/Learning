leetcode  165. Compare Version Numbers（python）




### 描述

Given two version numbers, version1 and version2, compare them.

Version numbers consist of one or more revisions joined by a dot '.'. Each revision consists of digits and may contain leading zeros. Every revision contains at least one character. Revisions are 0-indexed from left to right, with the leftmost revision being revision 0, the next revision being revision 1, and so on. For example 2.5.33 and 0.1 are valid version numbers.

To compare version numbers, compare their revisions in left-to-right order. Revisions are compared using their integer value ignoring any leading zeros. This means that revisions 1 and 001 are considered equal. If a version number does not specify a revision at an index, then treat the revision as 0. For example, version 1.0 is less than version 1.1 because their revision 0s are the same, but their revision 1s are 0 and 1 respectively, and 0 < 1.

Return the following:

* If version1 < version2, return -1.
* If version1 > version2, return 1.
* Otherwise, return 0.



Example 1:

	Input: version1 = "1.01", version2 = "1.001"
	Output: 0
	Explanation: Ignoring leading zeroes, both "01" and "001" represent the same integer "1".

	
Example 2:

	Input: version1 = "1.0", version2 = "1.0.0"
	Output: 0
	Explanation: version1 does not specify revision 2, which means it is treated as "0".


Example 3:


	Input: version1 = "0.1", version2 = "1.1"
	Output: -1
	Explanation: version1's revision 0 is "0", while version2's revision 0 is "1". 0 < 1, so version1 < version2.
	




Note:


	1 <= version1.length, version2.length <= 500
	version1 and version2 only contain digits and '.'.
	version1 and version2 are valid version numbers.
	All the given revisions in version1 and version2 can be stored in a 32-bit integer.

### 解析

根据题意，给定两个版本号，version1 和 version2，比较它们的大小。版本号由一个或多个数字字符串用点 “.” 连接的组成，每个数字字符串可能包含前导零，这些用 “.” 隔开的数字字符串就是常说的主版本号、次版本号、次次版本号等等。要比较版本号，要按照按从左到右的顺序，比较它们两个主版本号代表的整数数字大小，然后比较它们两个第次版本号代表的整数数字大小，以此类推，如果有前导零可以将其忽略，如果到了某个阶段其中一个版本号没有，则用 0 补齐继续进行比较。在比较过程中只要 version1 某个阶段的整数大于 version2 某个阶段的整数就返回 1 ，小于就返回 -1 ，如果直到最后都是相等则返回 0 。


上面的题目说了一大堆其实就是为了介绍版本号的构成和比较规则，说是比较繁琐，我们通过观察例子就能够理解快速理解题意，思路也相对比较简单：

* 将 version1, version2 两个字符串用 “.”  切分成列表
* 取两者最长的长度，将较短的列表用 0 补齐
* 然后遍历两个列表的相同索引的整数值 a 和 b ，如果 a > b 直接返回 1， 如果 a<b 直接返回 -1 ，如果 a==b 继续下一次循环
* 遍历结束直接返回 0

只要看懂题目，还是很简单的，主要还是题目太繁琐，让人望而生畏。

### 解答
				

	class Solution(object):
	    def compareVersion(self, version1, version2):
	        """
	        :type version1: str
	        :type version2: str
	        :rtype: int
	        """
	        L1 = version1.split('.')
	        L2 = version2.split('.')
	        max_L = max(len(L1), len(L2))
	        if len(L1) != max_L:
	            L1 += [0] * (max_L-len(L1))
	        else:
	            L2 += [0] * (max_L-len(L2))
	        for i in range(max_L):
	            a,b = int(L1[i]), int(L2[i])
	            if a>b:
	                return 1
	            elif a<b:
	                return -1
	            elif a==b:
	                continue
	        return 0
	        
            	      
			
### 运行结果


	Runtime: 28 ms, faster than 39.14% of Python online submissions for Compare Version Numbers.
	Memory Usage: 13.5 MB, less than 76.32% of Python online submissions for Compare Version Numbers.


### 原题链接


https://leetcode.com/problems/compare-version-numbers/



您的支持是我最大的动力
