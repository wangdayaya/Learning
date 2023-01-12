leetcode  525. Contiguous Array（python）

### 描述

Given a binary array nums, return the maximum length of a contiguous subarray with an equal number of 0 and 1.






Example 1:

	Input: nums = [0,1]
	Output: 2
	Explanation: [0, 1] is the longest contiguous subarray with an equal number of 0 and 1.


	
Example 2:

	Input: nums = [0,1,0]
	Output: 2
	Explanation: [0, 1] (or [1, 0]) is a longest contiguous subarray with equal number of 0 and 1.






Note:

	1 <= nums.length <= 10^5
	nums[i] is either 0 or 1.


### 解析

根据题意，给定一个只包含了 0 或 1 的数组 nums，返回子数组中 0 和 1 数量相等的最大长度。答案也就是最小为 0 ，最大为 len(nums) 。


既然要求子数组中 0 和 1 的数量相等，那么将题目转化一下，将数组中的 0 都变成 -1 ，那么只要找到长度最长的和为 0 的子数组就可以。到这里我们就可以使用前缀和的思路来解决这道题。

* 初始化 result 和 presum 都为 0 ，分别表示结果和从头开始的前缀和，初始化字典 d 表示子数组的和及其最开始出现的索引，同时初始化 {0: -1} 表示在 presum[:i] 和为 0 的时候长度为 i-(-1) 
* 遍历 nums 中的每个索引 i 和元素 x ，如果 x 为 0 则将其变为 -1 ，如果为 1 则不变，并将元素加到 presum 中，如果 presum 在 d 中出现过，表示 nums[d[presum]+1:i] 的和为 0 ，比较可获取最长长度 result ，如果 presum 没有在 d 中出现过，则使用字典 d 记录其第一次出现的位置索引 i
* 遍历结束通过比较找到的 result 即为答案

### 解答
				

	class Solution(object):
	    def findMaxLength(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        d = {0: -1}
	        presum = 0
	        for i,x in enumerate(nums):
	            presum += -1 if x==0 else x
	            if presum in d:
	                result = max(result, i-d[presum])
	            else:
	                d[presum] = i
	        return result
            	      
			
### 运行结果

	Runtime: 844 ms, faster than 24.02% of Python online submissions for Contiguous Array.
	Memory Usage: 17.1 MB, less than 97.06% of Python online submissions for Contiguous Array.


原题链接：https://leetcode.com/problems/contiguous-array/



您的支持是我最大的动力
