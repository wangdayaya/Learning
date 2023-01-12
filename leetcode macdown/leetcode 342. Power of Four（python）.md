leetcode  342. Power of Four（python）




### 描述

Given an integer n, return true if it is a power of four. Otherwise, return false.

An integer n is a power of three, if there exists an integer x such that n == 4^x.





Example 1:

	Input: n = 16
	Output: true

	
Example 2:

	Input: n = 5
	Output: false


Example 3:

	Input: n = 1
	Output: true



Note:


	-2^31 <= n <= 2^31 - 1


### 解析

根据题意，给定一个整数 n，如果它是 4 的幂，则返回 true 。 否则返回 False 。如果存在整数 x 使得 n == 4^x，则整数 n 是 4 的幂。

题目中已经给出来了定义，所以我们可以根据定义，不断对 n 进行除 4 的操作，并判断当前的 n 是否能被 4 整除，直到 n 等于 1 说明成功直接返回 True ，n 不等于 1 说明失败直接返回 False 。

时间复杂度为 O(logn) ，空间复杂度为 O(1) 。

### 解答
	class Solution(object):
	    def isPowerOfFour(self, n):
	        """
	        :type n: int
	        :rtype: bool
	        """
	        if n<1:
	            return False
	        while n%4==0:
	            n //= 4
	        return n==1
	
	



### 运行结果

	Runtime: 17 ms, faster than 93.01% of Python online submissions for Power of Four.
	Memory Usage: 13.5 MB, less than 38.24% of Python online submissions for Power of Four.

### 原题链接

	https://leetcode.com/problems/power-of-four/


您的支持是我最大的动力
