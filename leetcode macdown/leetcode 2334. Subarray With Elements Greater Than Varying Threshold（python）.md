leetcode  2334. Subarray With Elements Greater Than Varying Threshold（python）




### 描述


You are given an integer array nums and an integer threshold.

Find any subarray of nums of length k such that every element in the subarray is greater than threshold / k.

Return the size of any such subarray. If there is no such subarray, return -1.

A subarray is a contiguous non-empty sequence of elements within an array.


Example 1:

	Input: nums = [1,3,4,3,1], threshold = 6
	Output: 3
	Explanation: The subarray [3,4,3] has a size of 3, and every element is greater than 6 / 3 = 2.
	Note that this is the only valid subarray.


	
Example 2:


	Input: nums = [6,5,6,5,8], threshold = 7
	Output: 1
	Explanation: The subarray [8] has a size of 1, and 8 > 7 / 1 = 7. So 1 is returned.
	Note that the subarray [6,5] has a size of 2, and every element is greater than 7 / 2 = 3.5. 
	Similarly, the subarrays [6,5,6], [6,5,6,5], [6,5,6,5,8] also satisfy the given conditions.
	Therefore, 2, 3, 4, or 5 may also be returned.




Note:

	1 <= nums.length <= 10^5
	1 <= nums[i], threshold <= 10^9


### 解析

根据题意，给定一个整数数组 nums 和一个整数 threshold 。找到任何长度为 k 的 nums 子数组，使得子数组中的每个元素都大于 threshold / k 。返回任何此类子数组的大小 k 。 如果没有这样的子数组，则返回 -1 。子数组是数组中元素的连续非空序列。

因为子数组中的每个元素都要大于 threshold / k  ，我们假设每个元素是其所在的子数组中的最小值，那么我们使用单调栈来找出子数组最左边和最右边的边界，如果知道了左右边界即可算出子数组的长度 k 。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def validSubarraySize(self, nums, threshold):
	        """
	        :type nums: List[int]
	        :type threshold: int
	        :rtype: int
	        """
	        n = len(nums)
	        left, st = [-1] * n, []  # left[i] 为左侧小于 nums[i] 的最近元素位置（不存在时为 -1）
	        for i, v in enumerate(nums):
	            while st and nums[st[-1]] >= v: st.pop()
	            if st: left[i] = st[-1]
	            st.append(i)
	
	        right, st = [n] * n, []  # right[i] 为右侧小于 nums[i] 的最近元素位置（不存在时为 n）
	        for i in range(n - 1, -1, -1):
	            while st and nums[st[-1]] >= nums[i]: st.pop()
	            if st: right[i] = st[-1]
	            st.append(i)
	
	        for i, num in enumerate(nums):
	            k = right[i] - left[i] - 1
	            if num > threshold // k:
	                return k
	        return -1
	


### 运行结果


	64 / 64 test cases passed.
	Status: Accepted
	Runtime: 1804 ms
	Memory Usage: 25.7 MB

### 原题链接

https://leetcode.com/contest/biweekly-contest-82/problems/subarray-with-elements-greater-than-varying-threshold/


您的支持是我最大的动力
