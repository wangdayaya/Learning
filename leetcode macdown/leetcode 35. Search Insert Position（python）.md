leetcode  35. Search Insert Position（python）

### 描述


Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.


Example 1:


	Input: nums = [1,3,5,6], target = 5
	Output: 2
	
Example 2:


	Input: nums = [1,3,5,6], target = 2
	Output: 1

Example 3:


	Input: nums = [1,3,5,6], target = 7
	Output: 4
	
Example 4:

	Input: nums = [1,3,5,6], target = 0
	Output: 0

	
Example 5:

	Input: nums = [1], target = 0
	Output: 0


Note:

	1 <= nums.length <= 10^4
	-10^4 <= nums[i] <= 10^4
	nums contains distinct values sorted in ascending order.
	-10^4 <= target <= 10^4


### 解析

根据题意，给出了一个已经有序的整数列表 nums 和一个整数 target ，如果 target 在 nums 中，则直接返回所在位置的索引，如果 target 不在 nums 中，则返回一个索引保证 target 插入 nums 之后仍然是有序的。思路比较简单：

* 如果 target 在 nums 中，直接返回 nums.index(target)
* 否则遍历 nums 中的每个元素 num ，如果 num 大于 target 直接返回当前位置的索引
* 遍历结束如果仍然没有找到合适的索引位置，说明 target 大于 nums 中的所有元素，直接返回 nums 的长度即可

题目中要求我们算法的时间复杂度为 O(log n) ，显然直接用二分法才能满足题意，但我就是不用，哎就是玩～


### 解答
				
	class Solution(object):
	    def searchInsert(self, nums, target):
	        """
	        :type nums: List[int]
	        :type target: int
	        :rtype: int
	        """
	        if target in nums:
	            return nums.index(target)
	        for i, num in enumerate(nums):
	            if num > target:
	                return i
	        return len(nums)

            	      
			
### 运行结果

	Runtime: 28 ms, faster than 96.37% of Python online submissions for Search Insert Position.
	Memory Usage: 14.2 MB, less than 25.57% of Python online submissions for Search Insert Position.

### 解析

骗你的啦，我肯定会用二分搜索法求解的。略略略～


### 解答
	class Solution(object):
	    def searchInsert(self, nums, target):
	        """
	        :type nums: List[int]
	        :type target: int
	        :rtype: int
	        """
	        l, r = 0, len(nums)-1
	        while l <= r:
	            mid = (l + r)//2
	            if nums[mid] == target:
	                return mid
	            if nums[mid] < target:
	                l = mid + 1
	            else:
	                r = mid - 1
	        return l
	        
### 运行结果

	Runtime: 38 ms, faster than 43.70% of Python online submissions for Search Insert Position.
	Memory Usage: 14 MB, less than 97.13% of Python online submissions for Search Insert Position.

原题链接：https://leetcode.com/problems/search-insert-position/



您的支持是我最大的动力
