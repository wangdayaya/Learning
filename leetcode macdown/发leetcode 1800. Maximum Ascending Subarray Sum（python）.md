leetcode  1800. Maximum Ascending Subarray Sum（python）

### 描述

Given an array of positive integers nums, return the maximum possible sum of an ascending subarray in nums.

A subarray is defined as a contiguous sequence of numbers in an array.

A subarray [nums[l], nums[l+1], ..., nums[r-1], nums[r]] is ascending if for all i where l <= i < r, nums[i] < nums[i+1]. Note that a subarray of size 1 is ascending.





Example 1:


	Input: nums = [10,20,30,5,10,50]
	Output: 65
	Explanation: [5,10,50] is the ascending subarray with the maximum sum of 65.
	
Example 2:

	Input: nums = [10,20,30,40,50]
	Output: 150
	Explanation: [10,20,30,40,50] is the ascending subarray with the maximum sum of 150.


Example 3:

	
	Input: nums = [12,17,15,13,10,11,12]
	Output: 33
	Explanation: [10,11,12] is the ascending subarray with the maximum sum of 33.
	
Example 4:

	Input: nums = [100,10,1]
	Output: 100

	

Note:

	1 <= nums.length <= 100
	1 <= nums[i] <= 100


### 解析


根据题意，就是给出一个正整数的列表 nums ，找出升序子列表的最大和。子序列就是列表 nums 中的若干个连续的元素组成的列表，升序列就是后一个元素的值比前一个元素的值要大，注意只有一个元素的子序列也算是升序列。思路：

* 如果 nums 长度为 1 ，nums 本身就是升序列，直接返回 nums[0] 
* 否则使用 total 表示当前子列表的和， max\_result 表示最终需要返回的最大和。遍历 nums ，如果当前的元素小于等于前一个元素，那么直接将 total 设置为当前元素，表示从这里开始往后找新的升序子列表。如果当前元素大于前一个元素，那么当前的升序列表还可以找下去，将该元素加入 total 
* 然后比较 total 和 max\_result 的最大值来更新 max\_result 
* 遍历结束即可得到最终的结果 max_result 。

### 解答
				

	class Solution(object):
	    def maxAscendingSum(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        if len(nums) == 1:
	            return nums[0]
	        total = nums[0]
	        max_result = nums[0]
	        for i in range(1, len(nums)):
	            e = nums[i]
	            if e>nums[i-1]:
	                total += e
	                max_result = max(max_result, total)
	            else:
	                total = e
	        return max_result
            	      
			
### 运行结果
	
	Runtime: 16 ms, faster than 93.91% of Python online submissions for Maximum Ascending Subarray Sum.
	Memory Usage: 13.6 MB, less than 9.68% of Python online submissions for Maximum Ascending Subarray Sum.


原题链接：https://leetcode.com/problems/maximum-ascending-subarray-sum/



您的支持是我最大的动力
