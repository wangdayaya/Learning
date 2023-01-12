leetcode  1856. Maximum Subarray Min-Product（python）

### 描述


The min-product of an array is equal to the minimum value in the array multiplied by the array's sum.

* For example, the array \[3,2,5] (minimum value is 2) has a min-product of 2 \* (3+2+5) = 2 \* 10 = 20.

Given an array of integers nums, return the maximum min-product of any non-empty subarray of nums. Since the answer may be large, return it modulo 10^9 + 7.

Note that the min-product should be maximized before performing the modulo operation. Testcases are generated such that the maximum min-product without modulo will fit in a 64-bit signed integer.

A subarray is a contiguous part of an array.


Example 1:

	Input: nums = [1,2,3,2]
	Output: 14
	Explanation: The maximum min-product is achieved with the subarray [2,3,2] (minimum value is 2).
	2 * (2+3+2) = 2 * 7 = 14.

	
Example 2:

	Input: nums = [2,3,3,1,2]
	Output: 18
	Explanation: The maximum min-product is achieved with the subarray [3,3] (minimum value is 3).
	3 * (3+3) = 3 * 6 = 18.


Example 3:

	
	Input: nums = [3,1,5,6,4,2]
	Output: 60
	Explanation: The maximum min-product is achieved with the subarray [5,6,4] (minimum value is 4).
	4 * (5+6+4) = 4 * 15 = 60.

	



Note:

	1 <= nums.length <= 10^5
	1 <= nums[i] <= 10^7


### 解析

根据题意，数组的最小乘积等于数组中的最小值乘以数组的总和。

* 例如，数组 \[3,2,5]（最小值为 2）的最小乘积为 2 \* (3+2+5) = 2 \* 10 = 20。

给定一个整数数组 nums，返回 nums 的任何非空子数组的最大的最小乘积。 由于答案可能很大，将其取模 10^9 + 7 返回。需要注意的是，在执行模运算之前应最大化最小乘积。

最暴力的解法就是 O(n^2) ，两层循环找出所有的子数组，然后找出每个子数组的最小值与和的乘积，最后找出最大值即可，但是这方法肯定会超时，因为题目条件说明了数组的长度可能为 10^5 。因为每个元素都是大于 0 的正整数，所以我们可以找出以每个元素 nums[i] 为最小值的最大子数组的左右边际，这个过程可以通过单调栈的思想来解决，然后使用前缀和求出这个左右边际范围内的子数组的和与 nums[i] 相乘，最后经过比较找出最大值，对 (10**9+7) 取模即可返回答案。

### 解答
				

	class Solution(object):
	    def maxSumMinProduct(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        N = len(nums)
	        
	        nextSmaller = [N] * N
	        preSmaller = [-1] * N
	        
	        stack = []
	        for i in range(N):
	            while stack and nums[stack[-1]] > nums[i]:
	                nextSmaller[stack.pop()] = i
	            stack.append(i)
	        
	        stack = []
	        for i in range(N-1,-1,-1):
	            while stack and nums[stack[-1]] > nums[i]:
	                preSmaller[stack.pop()] = i
	            stack.append(i)
	            
	        presum = [0] * N
	        presum[0] = nums[0]
	        for i in range(1, N):
	            presum[i] = presum[i-1] + nums[i]
	            
	        result = 0
	        for i in range(N):
	            a = 0 if preSmaller[i] == -1 else presum[preSmaller[i]]
	            b = presum[nextSmaller[i]-1]
	            result = max(result, (b-a) * nums[i])
	            
	        return result%(10**9+7)
	            
	        
            	      
			
### 运行结果


	Runtime: 1416 ms, faster than 41.67% of Python online submissions for Maximum Subarray Min-Product.
	Memory Usage: 28.9 MB, less than 41.67% of Python online submissions for Maximum Subarray Min-Product.

原题链接：https://leetcode.com/problems/maximum-subarray-min-product/



您的支持是我最大的动力
