leetcode 989. Add to Array-Form of Integer （python）




### 描述

The array-form of an integer num is an array representing its digits in left to right order.

For example, for num = 1321, the array form is [1,3,2,1].

Given num, the array-form of an integer, and an integer k, return the array-form of the integer num + k.



Example 1:

	Input: num = [1,2,0,0], k = 34
	Output: [1,2,3,4]
	Explanation: 1200 + 34 = 1234

	
Example 2:

	
	Input: num = [2,7,4], k = 181
	Output: [4,5,5]
	Explanation: 274 + 181 = 455

Example 3:


	
	Input: num = [2,1,5], k = 806
	Output: [1,0,2,1]
	Explanation: 215 + 806 = 1021


Note:


* 	1 <= num.length <= 10^4
* 	0 <= num[i] <= 9
* 	num does not contain any leading zeros except for the zero itself.
* 	1 <= k <= 10^4

### 解析

根据题意，整数 num 的数组是按从左到右的顺序表示一个整数数字的数组。例如，对于 num = 1321，数组形式为 [1，3，2，1]。给定 num 整数的数组形式和整数 k ，返回整数 num + k 的数组形式。

其实这道题本质上就是考查两个整数的加法计算原理，我们按照最朴素的思路，从右往左，不断取两个整数的对应位上的数字进行相加然后存放到结果列表中，直接返回即可。

N 为 num 数组的长度，时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def addToArrayForm(self, num, k):
	        """
	        :type num: List[int]
	        :type k: int
	        :rtype: List[int]
	        """
	        result = []
	        carry = 0
	        k = list(str(k))
	        while num or k or carry:
	            if num:
	                carry += num.pop()
	            if k:
	                carry += int(k.pop())
	            result.append(carry % 10)
	            carry //= 10
	        return result[::-1]

### 运行结果

	Runtime 246 ms，Beats 84.82%
	Memory 14.2 MB，Beats 35.31%


### 原题链接

https://leetcode.com/problems/add-to-array-form-of-integer/description/


您的支持是我最大的动力
