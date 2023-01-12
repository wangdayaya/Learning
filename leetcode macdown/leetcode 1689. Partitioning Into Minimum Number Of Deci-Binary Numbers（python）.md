leetcode  1689. Partitioning Into Minimum Number Of Deci-Binary Numbers（python）

### 描述

A decimal number is called deci-binary if each of its digits is either 0 or 1 without any leading zeros. For example, 101 and 1100 are deci-binary, while 112 and 3001 are not.

Given a string n that represents a positive decimal integer, return the minimum number of positive deci-binary numbers needed so that they sum up to n.

 	



Example 1:

	Input: n = "32"
	Output: 3
	Explanation: 10 + 11 + 11 = 32	
	
Example 2:

	Input: n = "82734"
	Output: 8

Example 3:

	Input: n = "27346209830709182346"
	Output: 9




Note:
	
	1 <= n.length <= 10^5
	n consists of only digits.
	n does not contain any leading zeros and represents a positive integer.




### 解析
 
根据题意，就是给出了一个十进制的 n ，让我们找出最少数量的合法的 deci-binary 。deci-binary 在题目中的定义就是用 1 或 0 组成的非 0 开头的十进制数字。而给出的 n 是一个表示很大数字的字符串，一开始想到的最简单最无脑的方法，就是靠内置的函数或者暴力求解，通过循环来找出满足题意的 deci-binary 个数，但是一看例子三我就知道没戏，因为 n 表示的数字都太大了，根本没法运算，而根据 leetcode 的尿性，这种题一般都有规律可循，所以要找出最少数量的合法的 deci-binary 需要发现规律。

我们以 n=“82734” 为例子，挖掘其中的规律，想要找出最少数量的合法的 deci-binary ，只能是有一种组合，如下所示：

	1 1 1 1 1
	1 1 1 1 1
	1 0 1 1 1
	1 0 1 0 1
	1 0 1 0 0
	1 0 1 0 0
	1 0 1 0 0
	1 0 0 0 0
	---------
	8 2 7 3 4
   
我们可以看到，每个数字都是尽量最大的，这样才能保证最后个数的 deci-binary 最少，当我们列举更多其他的例子后，进一步我们分析每个例子的所有 deci-binary 结果的组成规律就可以发现，结果中的 deci-binary 个数和 n 中最大的数字字符是一样的。所以本题的答案也就呼之欲出，找出 n 中的最大数字字符转为整数，即为本题的答案，惊不惊喜！意不意外！我他娘的还真是个天才！

### 解答
				
	
	class Solution(object):
	    def minPartitions(self, n):
	        """
	        :type n: str
	        :rtype: int
	        """
	        return max(n)            	      
			
### 运行结果

	Runtime: 44 ms, faster than 84.72% of Python online submissions for Partitioning Into Minimum Number Of Deci-Binary Numbers.
	Memory Usage: 14.4 MB, less than 77.07% of Python online submissions for Partitioning Into Minimum Number Of Deci-Binary Numbers.


原题链接：https://leetcode.com/problems/partitioning-into-minimum-number-of-deci-binary-numbers/


您的支持是我最大的动力
