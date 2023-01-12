
「这是我参与2022首次更文挑战的第N天，活动详情查看：[2022首次更文挑战](https://juejin.cn/post/7052884569032392740 "https://juejin.cn/post/7052884569032392740")」

### 前言

这是 leetcode 中 Biweekly Contest 72 的第二题，难度为 Medium ，考查的就是数学的基本方程思想，只要用数学的思想就很容易，用其他解法回把简单问题复杂化。

### 描述


Given an integer num, return three consecutive integers (as a sorted array) that sum to num. If num cannot be expressed as the sum of three consecutive integers, return an empty array.


Example 1:


	Input: num = 33
	Output: [10,11,12]
	Explanation: 33 can be expressed as 10 + 11 + 12 = 33.
	10, 11, 12 are 3 consecutive integers, so we return [10, 11, 12].
	
Example 2:

	Input: num = 4
	Output: []
	Explanation: There is no way to express 4 as the sum of 3 consecutive integers.




Note:


	0 <= num <= 10^15

### 解析

根据题意，给定一个整数 num，返回三个和为 num 的连续整数结果。 如果 num 不能表示为三个连续整数之和，则返回一个空数组。

题意同样简洁明了，比赛的时候就喜欢这种题意简单明晰的风格，一看那些啰里八嗦讲一堆都讲不清楚的题目就烦。

书归正传，我们看限制条件 num 最大可能为 10^15 ，所以用常规的方法是解不出来的，所以我们可以使用方程法，设三个数中最小的为 x ，那么三个数字的和为 num ，可以列出方程

	x + x + 1 + x + 2 = num
	
解这个方程，求 x 的值为 

	x = num/3 - 1
	
所以我们代码中直接判断，如果 num 不能被 3 整除，那肯定是无解的，直接返回空列表即可，否则按照求解的结果，直接计算出三个整数形成的列表返回即可。



### 解答
				


	class Solution(object):
	    def sumOfThree(self, num):
	        """
	        :type num: int
	        :rtype: List[int]
	        """
	        if num%3!=0:
	            return []
	        a = num//3 - 1
	        b = a + 1
	        c = b + 1
	        return [a,b,c]
      	      
			
### 运行结果


	379 / 379 test cases passed.
	Status: Accepted
	Runtime: 24 ms
	Memory Usage: 13.4 MB

### 原题链接


https://leetcode.com/contest/biweekly-contest-72/problems/find-three-consecutive-integers-that-sum-to-a-given-number/


您的支持是我最大的动力
