leetcode 16. 3Sum Closest （python）




### 描述

Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.



Example 1:

	Input: nums = [-1,2,1,-4], target = 1
	Output: 2
	Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

	
Example 2:

	Input: nums = [0,0,0], target = 1
	Output: 0
	Explanation: The sum that is closest to the target is 0. (0 + 0 + 0 = 0).


 

Note:


	3 <= nums.length <= 1000
	-1000 <= nums[i] <= 1000
	-10^4 <= target <= 10^4


### 解析

根据题意，给定一个长度为 n 的整数数组和一个整数 target ，在数字中查找三个整数，使总和最接近目标。返回三个整数之和。我们的题目可以假定每个输入将只有一种结果。

对于这种找“最接近” target 的三元组题目，最暴力直接的算法就是使用三重循环，但是因为 nums 的最大长度为 1000 ，所以肯定会超时。

这种情况我们可以先固定一个元素 A ，然后去找另外的两个元素 B 和 C ，这样我们只需要找最接近 target - A 的 B 和 C 的和即可，要想知道 B 和 C 的和到底是大还是小，就需要对整个的 nums 先进行排序，这样一来我们在从左向右遍历 nums 的时候，每确定一个元素 A ，就可以将剩下后面部分的第一个元素当作 B ，最后一个元素当作 C ，这个时候我们使用双指针算法，不断将指向 B 的指针向右移动，或者将指向 C 的指针向左移动，最后肯定能找到一个合适的结果。

N 是 nums 的长度，时间复杂度为 O(N^2) ，空间复杂度为 O(N)。

### 解答

	class Solution(object):
	    def threeSumClosest(self, nums, target):
	        """
	        :type nums: List[int]
	        :type target: int
	        :rtype: int
	        """
	        nums.sort()
	        result = nums[0] + nums[1] + nums[2]
	        for i in range(len(nums) - 2):
	            j, k = i + 1, len(nums) - 1
	            while j < k:
	                total = nums[i] + nums[j] + nums[k]
	                if total == target:
	                    return total
	                if abs(total - target) < abs(result - target):
	                    result = total
	                if total < target:
	                    j += 1
	                elif total > target:
	                    k -= 1
	        return result

### 运行结果


	Runtime: 3996 ms, faster than 77.88% of Python online submissions for 3Sum Closest.
	Memory Usage: 13.6 MB, less than 58.19% of Python online submissions for 3Sum Closest.
	
### 原题链接

https://leetcode.com/problems/3sum-closest/


您的支持是我最大的动力
