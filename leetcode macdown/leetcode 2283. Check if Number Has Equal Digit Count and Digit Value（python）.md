leetcode  2283. Check if Number Has Equal Digit Count and Digit Value（python）

这是 Biweekly Contest 79 的第一题，难度 Eazy，主要考察的是计数器和字符串的基本操作。


### 描述

You are given a 0-indexed string num of length n consisting of digits.

Return true if for every index i in the range 0 <= i < n, the digit i occurs num[i] times in num, otherwise return false.

 



Example 1:

	Input: num = "1210"
	Output: true
	Explanation:
	num[0] = '1'. The digit 0 occurs once in num.
	num[1] = '2'. The digit 1 occurs twice in num.
	num[2] = '1'. The digit 2 occurs once in num.
	num[3] = '0'. The digit 3 occurs zero times in num.
	The condition holds true for every index in "1210", so return true.

	
Example 2:

	Input: num = "030"
	Output: false
	Explanation:
	num[0] = '0'. The digit 0 should occur zero times, but actually occurs twice in num.
	num[1] = '3'. The digit 1 should occur three times, but actually occurs zero times in num.
	num[2] = '0'. The digit 2 occurs zero times in num.
	The indices 0 and 1 both violate the condition, so return false.







Note:

	n == num.length
	1 <= n <= 10
	num consists of digits.


### 解析


根据题意，给定一个长度为 n 的 0 索引字符串 num ，完全由数字组成。如果对于 0 <= i < n 范围内的每个索引 i ，数字 i 在 num 中出现 num[i] 次，则返回 true ，否则返回 false 。

这道题一开始读下来有点绕，但是结合例子捋清楚题意就好懂了，其实就是判断索引 i 字符串是否在 num 中出现 num[i] 次，考察的就是一个计数器的基本操作，使用 python 的内置函数进行统计即可，然后依次对每个索引进行题意的判断，如果出现不符合题意的情况直接返回 False ，否则遍历结束返回 True 。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。
### 解答
				

	class Solution(object):
	    def digitCount(self, num):
	        """
	        :type num: str
	        :rtype: bool
	        """
	        c = collections.Counter(num)
	        for i in range(len(num)):
	            if c[str(i)] == int(num[i]):
	                continue
	            else:
	                return False
	        return True
            	      
			
### 运行结果

	
	433 / 433 test cases passed.
	Status: Accepted
	Runtime: 37 ms
	Memory Usage: 13.6 MB


### 原题链接

https://leetcode.com/contest/biweekly-contest-79/problems/check-if-number-has-equal-digit-count-and-digit-value/



您的支持是我最大的动力
