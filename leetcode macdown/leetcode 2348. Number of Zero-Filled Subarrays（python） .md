leetcode  2348. Number of Zero-Filled Subarrays（python）




### 描述

Given an integer array nums, return the number of subarrays filled with 0.

A subarray is a contiguous non-empty sequence of elements within an array.



Example 1:

	Input: nums = [1,3,0,0,2,0,0,4]
	Output: 6
	Explanation: 
	There are 4 occurrences of [0] as a subarray.
	There are 2 occurrences of [0,0] as a subarray.
	There is no occurrence of a subarray with a size more than 2 filled with 0. Therefore, we return 6

	
Example 2:

	Input: nums = [0,0,0,2,0,0]
	Output: 9
	Explanation:
	There are 5 occurrences of [0] as a subarray.
	There are 3 occurrences of [0,0] as a subarray.
	There is 1 occurrence of [0,0,0] as a subarray.
	There is no occurrence of a subarray with a size more than 3 filled with 0. Therefore, we return 9.


Example 3:

	Input: nums = [2,10,2019]
	Output: 0
	Explanation: There is no subarray filled with 0. Therefore, we return 0.



Note:

	1 <= nums.length <= 10^5
	-10^9 <= nums[i] <= 10^9


### 解析

根据题意，给定一个整数数组 nums ，返回用 0 填充的所有子数组的数量。子数组是数组中元素的连续非空序列。

这道题其实也不难，只需要从左到右遍历 nums 中的元素，如果元素不为 0 ，则 t 一直设置为 0 ，如果出现元素为 0 则用 t 不断累积可能出现的全 0 的子数组的个数并将其加入到 result 中，不断继续后面的遍历操作直到结束。

时间复杂度为 O(N) ，空间复杂度为 O(1)。

### 解答

	class Solution(object):
	    def zeroFilledSubarray(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        t = 0
	        for c in nums:
	            if c:
	                t = 0
	            else:
	                t+= 1
	                result += t
	        return result
### 运行结果


	48 / 48 test cases passed.
	Status: Accepted
	Runtime: 1340 ms
	Memory Usage: 23.1 MB

### 原题链接

	https://leetcode.com/contest/biweekly-contest-83/problems/number-of-zero-filled-subarrays/


您的支持是我最大的动力
