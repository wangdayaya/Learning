leetcode  326. Power of Three（python）




### 描述

Given an integer n, return true if it is a power of three. Otherwise, return false.

An integer n is a power of three, if there exists an integer x such that n == 3^x.





Example 1:

	Input: n = 27
	Output: true

	
Example 2:

	Input: n = 0
	Output: false


Example 3:

	Input: n = 9
	Output: true



Note:


	-2^31 <= n <= 2^31 - 1


### 解析

根据题意，给定一个整数 n，如果它是 3 的幂，则返回 true 。 否则返回 False 。如果存在整数 x 使得 n == 3^x，则整数 n 是 3 的幂。

题目中已经给出来了定义，所以我们可以根据定义，不断对 n 进行除 3 的操作，并判断当前的 n 是否能被 3 整除，直到 n 等于 1 说明成功直接返回 True ，n 不等于 1 说明失败直接返回 False 。

时间复杂度为 O(logn) ，空间复杂度为 O(1) 。

### 解答
	class Solution(object):
	    def isPowerOfThree(self, n):
	        """
	        :type n: int
	        :rtype: bool
	        """
	        if n < 1: return False
	        while n%3==0:
	            n//=3
	        return n==1


### 运行结果

	Runtime: 100 ms, faster than 76.72% of Python online submissions for Power of Three.
	Memory Usage: 13.5 MB, less than 14.97% of Python online submissions for Power of Three.


### 解析

这道题考察的是基本的数学知识，上面用了试错法，另外我们可以用打表法轻松解决，因为在限制条件的范围内，3 的幂最大的整数为 3 
^19 =1162261467 ，所以我们可以直接用它进行判断，当 n 大于 0 并且 1162261467 是 n 的倍数的时候，直接返回 True ，否则返回 False 。

时间复杂度为 O(1) ，空间复杂度为 O(1) 。
### 解答

	class Solution(object):
	    def isPowerOfThree(self, n):
	        """
	        :type n: int
	        :rtype: bool
	        """
	        return n>0 and 1162261467%n==0
        
### 运行结果

	Runtime: 75 ms, faster than 95.12% of Python online submissions for Power of Three.
	Memory Usage: 13.5 MB, less than 14.97% of Python online submissions for Power of Three.

### 原题链接

	https://leetcode.com/problems/power-of-three/


您的支持是我最大的动力
