leetcode  34. Find First and Last Position of Element in Sorted Array（python）




### 描述


Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value. If target is not found in the array, return [-1, -1]. You must write an algorithm with O(log n) runtime complexity.




Example 1:

	Input: nums = [5,7,7,8,8,10], target = 8
	Output: [3,4]

	
Example 2:

	Input: nums = [5,7,7,8,8,10], target = 6
	Output: [-1,-1]


Example 3:


	Input: nums = [], target = 0
	Output: [-1,-1]


Note:

	0 <= nums.length <= 10^5
	-10^9 <= nums[i] <= 10^9
	nums is a non-decreasing array.
	-10^9 <= target <= 10^9


### 解析

根据题意，给定一个按非降序排序的整数数组 nums ，找到给定目标值的开始和结束位置。如果在数组中找不到目标，则返回 [-1, -1] 。要求必须编写一个具有 O(log n) 运行时复杂度的算法。

这很明显就是让我们用二分法去找目标值，我们先通过二分法找出目标值的位置，然后分别向左找开始位置，然后向右再找结束位置。

时间复杂度为 O(logN) ，空间复杂度为 O(1) 。


### 解答

	class Solution(object):
	    def searchRange(self, nums, target):
	        """
	        :type nums: List[int]
	        :type target: int
	        :rtype: List[int]
	        """
	        if len(nums) == 0:
	            return [-1, -1]
	        N = len(nums)
	        l, r = 0, N - 1
	        while l <= r:
	            mid = (l + r) // 2
	            if nums[mid] < target:
	                l = mid + 1
	            elif nums[mid] > target:
	                r = mid - 1
	            else:
	                l = mid
	                while l >= 0 and nums[l] == target:
	                    l -= 1
	                if l == -1:
	                    l = 0
	                else:
	                    l += 1
	                r = mid
	                while r < N and nums[r] == target:
	                    r += 1
	                if r == N:
	                    r = N - 1
	                else:
	                    r -= 1
	                return [l, r]
	        return [-1, -1]

### 运行结果

	Runtime: 106 ms, faster than 37.82% of Python online submissions for Find First and Last Position of Element in Sorted Array.
	Memory Usage: 14.4 MB, less than 95.12% of Python online submissions for Find First and Last Position of Element in Sorted Array.


### 原题链接


	https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/

您的支持是我最大的动力
