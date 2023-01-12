leetcode  263. Ugly Number（python）




### 描述

An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.

Given an integer n, return true if n is an ugly number.



Example 1:

	Input: n = 6
	Output: true
	Explanation: 6 = 2 × 3

	
Example 2:


	Input: n = 1
	Output: true
	Explanation: 1 has no prime factors, therefore all of its prime factors are limited to 2, 3, and 5.

Example 3:

	Input: n = 14
	Output: false
	Explanation: 14 is not ugly since it includes the prime factor 7.




Note:


-2^31 <= n <= 2^31 - 1

### 解析

根据题意，丑陋的数字是一个正整数，其素因数限制为 2、3 和 5 。题目给定一个整数 n ，如果 n 是一个丑陋的数字，则返回 true ，否则返回 False 。

这其实就是一道简单的数学运算题，我们先将 n 可能为小于等于 0 的可能性都判断成 False ，然后再判断正整数。

对于正整数 n ，我们循环判断如果其不等于 1 ，则不断判断其是否可被 2、3、5 取模为 0 ，如果出现其他情况直接返回 False ，最后循环判断到如果 n 等于 1 则直接返回 True ，说明其是个丑陋的数字，确实只能被 2、3、5 整除。

时间复杂度为 O(logn)  ，空间复杂度为 O(1) 。

### 解答

	class Solution:
	    def isUgly(self, n: int) -> bool:
	        if n <= 0:
	            return False
	        while n != 1:
	            if n % 2 == 0:
	                n //= 2
	            elif n % 3 == 0:
	                n //= 3
	            elif n % 5 == 0:
	                n //= 5
	            else:
	                return False
	        return True

### 运行结果

	Runtime Beats 90.83%
	Memory Beats 58.76%

### 原题链接

https://leetcode.com/problems/ugly-number/description/


您的支持是我最大的动力
