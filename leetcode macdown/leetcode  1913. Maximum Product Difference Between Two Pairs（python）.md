leetcode  1913. Maximum Product Difference Between Two Pairs（python）

### 描述

The product difference between two pairs (a, b) and (c, d) is defined as (a * b) - (c * d).

	For example, the product difference between (5, 6) and (2, 7) is (5 * 6) - (2 * 7) = 16.
Given an integer array nums, choose four distinct indices w, x, y, and z such that the product difference between pairs (nums[w], nums[x]) and (nums[y], nums[z]) is maximized.

Return the maximum such product difference.



Example 1:

	Input: nums = [5,6,2,7,4]
	Output: 34
	Explanation: We can choose indices 1 and 3 for the first pair (6, 7) and indices 2 and 4 for the second pair (2, 4).
	The product difference is (6 * 7) - (2 * 4) = 34.

	
Example 2:

	Input: nums = [4,2,5,9,7,4,8]
	Output: 64
	Explanation: We can choose indices 3 and 6 for the first pair (9, 8) and indices 1 and 5 for the second pair (2, 4).
	The product difference is (9 * 8) - (2 * 4) = 64.



Note:


	4 <= nums.length <= 10^4
	1 <= nums[i] <= 10^4

### 解析

根据题意，就是给出一个整数数组 nums ，然后从 nums 中找出四个数字 a 、b 、c 、d ，使得 (a * b) - (c * d) 的结果是最大的，然后返回这个最大的值。这里用到的是内置的函数，因为这种两两相乘然后做差值的运算，肯定是要 a 和 b 用最大的数， c 和 d 用最小的数，这样结果才能最大，直接对 nums 进行排序，然后取最后的两个作为 a 和 b ，取最开始的两个数作为 c 和 d ，然后计算  (a * b) - (c * d)  得到的值即为结果。


### 解答
				
	class Solution(object):
	    def maxProductDifference(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        nums.sort()
	        return nums[-1]*nums[-2]-nums[0]*nums[1]

            	      
			
### 运行结果

	Runtime: 140 ms, faster than 77.76% of Python online submissions for Maximum Product Difference Between Two Pairs.
	Memory Usage: 14.5 MB, less than 29.60% of Python online submissions for Maximum Product Difference Between Two Pairs.


### 解析

上面使用内置函数的解法虽然思路简单，操作也方便，但是不推荐。我们可以直接使用遍历列表的方法，一次梭哈，直接用 a 和 b 表示两个较大的数，c 和 d 表示两个较小的数，然后通过和 nums 中的每个元素进行比较，更新他们的值，遍历结束计算 a * b - c * d 即为答案。


### 解答
				
	class Solution(object):
	    def maxProductDifference(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        a = b = float('-inf')
	        c = d = float('inf')
	        for n in nums:
	            if n <= c:
	                c, d, = n, c
	            elif n < d:
	                d = n
	            if n >= a:
	                a, b = n, a
	            elif n > b:
	                b = n
	        return a * b - c * d
            	      
			
### 运行结果

	Runtime: 140 ms, faster than 77.76% of Python online submissions for Maximum Product Difference Between Two Pairs.
	Memory Usage: 14.4 MB, less than 76.29% of Python online submissions for Maximum Product Difference Between Two Pairs.

### 解析

当然了我们仍然可以使用最无脑的暴力解决办法，但是性能太差了，四重循环求最大值，不推荐。

### 解析

	class Solution(object):
	    def maxProductDifference(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        for i,a in enumerate(nums):
	            for j,b in enumerate(nums):
	                for k,c in enumerate(nums):
	                    for l,d in enumerate(nums):
	                        if len(set([i,j,k,l])) == len([i,j,k,l]):
	                            result = max(result, a*b-c*d)
	        return result
### 运行结果

	Time Limit Exceeded
	
原题链接：https://leetcode.com/problems/maximum-product-difference-between-two-pairs



您的支持是我最大的动力
