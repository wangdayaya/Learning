leetcode  581. Shortest Unsorted Continuous Subarray（python）




### 描述

Given an integer array nums, you need to find one continuous subarray that if you only sort this subarray in ascending order, then the whole array will be sorted in ascending order.

Return the shortest such subarray and output its length.



Example 1:

	Input: nums = [2,6,4,8,10,9,15]
	Output: 5
	Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array sorted in ascending order.

	
Example 2:

	Input: nums = [1,2,3,4]
	Output: 0


Example 3:

	Input: nums = [1]
	Output: 0




Note:

	1 <= nums.length <= 10^4
	-10^5 <= nums[i] <= 10^5


### 解析

根据题意，给定一个整数数组 nums ，需要找到一个连续的子数组，如果只对这个子数组进行升序排序，那么整个数组将按升序排序。返回这个最短的子数组的长度。


如果有中间一段子数组是题目中描述的那样，那么根据这段子数组，可以将整个数组划分为三段，左、中、右，中间的子数组就是我们要找的目标，其最小值肯定是大于左边的最大值，其最大值肯定小于右边的最小值，所以我们要维护两个变量 min 和 max ，然后找出最后一个小于 max 的索引 end ，找出最后一个大于 min 的索引 start ，那么答案就是 start-end+1 。

这里使用了一点技巧，从左往右遍历 nums 的时候，同时计算 min 和 max ，然后同时找出 start 和 end ，分开两次遍历 nums 也是可以的，整体来说时间复杂度和空间复杂度差不了多少的，但是一次遍历明显会快一点。


时间复杂度为 O(N) ， 空间复杂度为 O(1) 。


### 解答
				

	class Solution(object):
	    def findUnsortedSubarray(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        N = len(nums)
	        start = 0
	        end = -1
	        min = nums[-1]
	        max = nums[0]
	        for i in range(N):
	            if nums[i] >= max:
	                max = nums[i]
	            else:
	                end = i
	            if nums[N-i-1] <= min:
	                min = nums[N-i-1]
	            else:
	                start = N-i-1
	        return end-start+1
            	      
			
### 运行结果

	Runtime: 159 ms, faster than 98.52% of Python online submissions for Shortest Unsorted Continuous Subarray.
	Memory Usage: 14.9 MB, less than 17.04% of Python online submissions for Shortest Unsorted Continuous Subarray.


### 原题链接


https://leetcode.com/problems/shortest-unsorted-continuous-subarray/


您的支持是我最大的动力
