leetcode  29. Divide Two Integers（python）




### 描述


Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator.

The integer division should truncate toward zero, which means losing its fractional part. For example, 8.345 would be truncated to 8, and -2.7335 would be truncated to -2.

Return the quotient after dividing dividend by divisor.

Note: Assume we are dealing with an environment that could only store integers within the 32-bit signed integer range: [−2^31, 2^31 − 1]. For this problem, if the quotient is strictly greater than 2^31 - 1, then return 2^31 - 1, and if the quotient is strictly less than -2^31, then return -2^31.

 


Example 1:

	Input: dividend = 10, divisor = 3
	Output: 3
	Explanation: 10/3 = 3.33333.. which is truncated to 3.

	
Example 2:

	Input: dividend = 7, divisor = -3
	Output: -2
	Explanation: 7/-3 = -2.33333.. which is truncated to -2.




Note:

* -2^31 <= dividend, divisor <= 2^31 - 1
* divisor != 0


### 解析

根据题意，给定两个整数 dividend 和 divisor ，在不使用乘法、除法和模运算符的情况下将两个整数相除。结果保留整数部分，丢掉小数部分。 例如，8.345 将被截断为 8 ，-2.7335 将被截断为 -2 。

注意：假设正在处理的环境只能存储 32 位有符号整数范围内的整数：[−2^31, 2^31 − 1] 。 对于这个问题，如果结果大于 2^31 - 1 ，则返回 2^31 - 1 ，如果结果小于 -2^31，则返回 -2^31 。

这道题已经明确告诉我们不能使用乘法、除法和取模的运算，那么只能用加法，所以方法很明确就是使用倍增法来解决该题。假如 10 除以 3 ，我们知道 10 大于 3 ，所以结果至少是 1 ，我们让 3 用加法翻一倍变成 6 ，10 肯定大于 6 ，所以结果肯定大于 2 ，我们再让 6 使用假饭翻一倍变成 12 ，10 小于 12 ，此时我们知道结果肯定在 2 到 4 之间，那么我们就用 10 减去刚才的 6 得到 4 ，看 4 是 3 的多少倍，把这个数字加到 2 上就是最终的结果 3 ，这就是整个的思路。

时间复杂度近似看作是 O(log(dividend/divisor)) ， 空间复杂度为 O(1) 。

### 解答
				

	class Solution(object):
	    def divide(self, dividend, divisor):
	        """
	        :type dividend: int
	        :type divisor: int
	        :rtype: int
	        """
	        is_negative = (dividend < 0) != (divisor < 0)
	        divisor, dividend = abs(divisor), abs(dividend)
	        current, mul = divisor, 1
	        result = 0
	        while dividend >= current:
	            if current + current <= dividend:
	                current += current
	                mul += mul
	            else:
	                result += mul
	                dividend -= current
	                current, mul = divisor, 1
	        return max(-result, -2**31) if is_negative else min(result, 2**31-1)
            	      
			
### 运行结果

	Runtime: 19 ms, faster than 90.96% of Python online submissions for Divide Two Integers.
	Memory Usage: 13.6 MB, less than 18.40% of Python online submissions for Divide Two Integers.



### 原题链接

https://leetcode.com/problems/divide-two-integers/


您的支持是我最大的动力
