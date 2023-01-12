leetcode  2016. Maximum Difference Between Increasing Elements（python）

### 描述

Given a 0-indexed integer array nums of size n, find the maximum difference between nums[i] and nums[j] (i.e., nums[j] - nums[i]), such that 0 <= i < j < n and nums[i] < nums[j].

Return the maximum difference. If no such i and j exists, return -1.



Example 1:

	Input: nums = [7,1,5,4]
	Output: 4
	Explanation:
	The maximum difference occurs with i = 1 and j = 2, nums[j] - nums[i] = 5 - 1 = 4.
	Note that with i = 1 and j = 0, the difference nums[j] - nums[i] = 7 - 1 = 6, but i > j, so it is not valid.

	
Example 2:

	
	Input: nums = [9,4,3,2]
	Output: -1
	Explanation:
	There is no i and j such that i < j and nums[i] < nums[j].

Example 3:

	Input: nums = [1,5,2,10]
	Output: 9
	Explanation:
	The maximum difference occurs with i = 0 and j = 3, nums[j] - nums[i] = 10 - 1 = 9.



Note:

* n == nums.length
* 2 <= n <= 1000
* 1 <= nums[i] <= 10^9

### 解析


根据题意，给出了一个从 0 开始索引长度为 n 的列表 nums ，找出 nums[j] - nums[i] 的最大的差值，且 0 <= i < j < n ，且 nums[i] < nums[j] 。如果存在最大差值就返回，否则返回 -1 。

最简单的方法就是暴力双重循环：

* 初始化结果 result 为 -1
* 遍历 range(len(nums)-1) 中的每个元素 i ，再遍历 range(i+1, len(nums)) 中的每个元素 j ，如果 nums[j] <= nums[i] 则继续进行下一循环，否则用 result 记录 nums[j]-nums[i] 的最大值
* 遍历结束返回 result

### 解答
				
	class Solution(object):
	    def maximumDifference(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = -1 
	        for i in range(len(nums)-1):
	            for j in range(i+1, len(nums)):
	                if nums[j] <= nums[i]:
	                    continue
	                result = max( result , nums[j]- nums[i])
	        return result

            	      
			
### 运行结果

	
	Runtime: 316 ms, faster than 16.50% of Python online submissions for Maximum Difference Between Increasing Elements.
	Memory Usage: 13.6 MB, less than 20.57% of Python online submissions for Maximum Difference Between Increasing Elements.
### 解析


也可以遍历一次就找到最大的差值，，思路：

* 初始化结果 result 为 -1 ，minNum 为 nums[0] 表示当前的最小值
* 从第二个元素开始，左到右每遍历到一个元素，如果  nums[i] 大于当前最小值 minNum ，就执行 max(result, nums[i]-minNum) 更新 result ，然后执行 min(minNum, nums[i]) 更新当前的最小值 minNum 
* 遍历结束返回 result 即可


### 解答

	class Solution(object):
	    def maximumDifference(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = -1 
	        minNum = nums[0]
	        for i in range(1, len(nums)):
	            if nums[i]>minNum:
	                result = max(result, nums[i]-minNum)
	            minNum = min(minNum, nums[i])
	        return result
### 运行结果

	Runtime: 36 ms, faster than 60.29% of Python online submissions for Maximum Difference Between Increasing Elements.
	Memory Usage: 13.6 MB, less than 20.57% of Python online submissions for Maximum Difference Between Increasing Elements.
	
原题链接：https://leetcode.com/problems/maximum-difference-between-increasing-elements/



您的支持是我最大的动力
