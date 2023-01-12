leetcode  287. Find the Duplicate Number（python）




### 描述

Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.



Example 1:

	Input: nums = [1,3,4,2,2]
	Output: 2

	
Example 2:


	Input: nums = [3,1,3,4,2]
	Output: 3






Note:


	1 <= n <= 10^5
	nums.length == n + 1
	1 <= nums[i] <= n
	All the integers in nums appear only once except for precisely one integer which appears two or more times.

### 解析

根据题意，给定一个包含 n + 1 个整数的整数数组 nums，其中每个整数都在 [1, n] 范围内。nums 中只有一个重复数，要求我们返回这个重复的数。题目还要求我们必须在不修改数组 nums 的情况下解决问题，并且只使用恒定的额外空间。其实看完题目我们就发现这道题目就是一道送分题，难度不像是 Medium ，而是 Eazy 。

其实这道题虽然要求了空间方面的限制，但是如果不用恒定的额外空间也没事，我当时解决这道题的思路就是用字典进行计数，哪一个数字的出现次数为 2 就直接将其返回。这种方法的时间复杂度为 O(N) ，空间复杂度为 O(N) 。

其实还有很多种其他的解法，比如我们可以对 nums 进行排序，然后遍历当前元素与下一个元素是否相等，如果相等就说明是重复的元素直接将其返回。这种解法的时间复杂度为 O(N) ，空间复杂度为 O(1) 。这种解法明显更符合题意。

从结果的统计来看，后一种解法的耗时和内存的使用都明显比前一种解法更少。


### 解答
				

	class Solution(object):
	    def findDuplicate(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        nums.sort()
	        for i in range(len(nums)-1):
	            if nums[i] == nums[i+1]:
	                return nums[i]

### 运行结果

	Runtime: 644 ms, faster than 71.73% of Python online submissions for Find the Duplicate Number.
	Memory Usage: 25.2 MB, less than 57.18% of Python online submissions for Find the Duplicate Number.


### 原题链接



https://leetcode.com/problems/find-the-duplicate-number/


您的支持是我最大的动力
