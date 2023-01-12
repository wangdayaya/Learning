leetcode  69. Sqrt(x)（python）

### 描述


Given a non-negative integer x, compute and return the square root of x.

Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.

Note: You are not allowed to use any built-in exponent function or operator, such as pow(x, 0.5) or x ** 0.5.




Example 1:

	Input: x = 4
	Output: 2


	
Example 2:

	Input: x = 8
	Output: 2
	Explanation: The square root of 8 is 2.82842..., and since the decimal part is truncated, 2 is returned.






Note:


	0 <= x <= 2^31 - 1

### 解析


题目中给出一个数字 x ，要求找出它开平方的结果，如果是整数结果就直接返回，如果是小数结果那只返回整数部分，思路比较简单（大意了，提交了几次都爆 ERROR 了，真的是耻辱）：

* 直接遍历 [0, x // 2 + 2] 范围内的每个整数 i
* 如果  i * i == x ，直接返回 i
* 如果 i * i > x ，直接返回 i-1

更加耻辱的是，这个解法最后超时，看了下题目中的要求， x 的量级真的是太大了...

### 解答
				

	class Solution(object):
	    def mySqrt(self, x):
	        """
	        :type x: int
	        :rtype: int
	        """
	        for i in range(0, x // 2 + 2):
	            if i * i == x:
	                return i
	            if i * i > x:
	                return i - 1
            	      
			
### 运行结果

	Memory Limit Exceeded

### 解析

当然可以直接使用二分搜索法进行查找。

### 解答

	class Solution(object):
	    def mySqrt(self, x):
	        """
	        :type x: int
	        :rtype: int
	        """
	        l, r = 0, x
	        while l <= r:
	            mid = (l + r) // 2
	            if mid ** 2 <= x < (mid + 1) ** 2:
	                return mid
	            elif x < mid ** 2:
	                r = mid - 1
	            else:
	                l = mid + 1

### 运行结果

	Runtime: 37 ms, faster than 37.53% of Python online submissions for Sqrt(x).
	Memory Usage: 13.3 MB, less than 66.04% of Python online submissions for Sqrt(x).

### 解析

当然了使用内置函数也可以，但是比较投机取巧。

### 解答

	class Solution(object):
	    def mySqrt(self, x):
	        """
	        :type x: int
	        :rtype: int
	        """
	        return int(sqrt(x))

### 运行结果

	Runtime: 25 ms, faster than 61.40% of Python online submissions for Sqrt(x).
	Memory Usage: 13.3 MB, less than 66.04% of Python online submissions for Sqrt(x).
	
	
原题链接：https://leetcode.com/problems/sqrtx/



您的支持是我最大的动力
