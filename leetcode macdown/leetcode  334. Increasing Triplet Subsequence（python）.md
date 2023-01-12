leetcode 334. Increasing Triplet Subsequence （python）




### 描述



Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.



Example 1:

	Input: nums = [1,2,3,4,5]
	Output: true
	Explanation: Any triplet where i < j < k is valid.

	
Example 2:


	Input: nums = [5,4,3,2,1]
	Output: false
	Explanation: No triplet exists.

Example 3:

	Input: nums = [2,1,5,0,4,6]
	Output: true
	Explanation: The triplet (3, 4, 5) is valid because nums[3] == 0 < nums[4] == 4 < nums[5] == 6.



Note:


	1 <= nums.length <= 5 * 10^5
	-2^31 <= nums[i] <= 2^31 - 1

### 解析

根据题意，给定一个整数数组 nums ，如果存在三重索引 （i， j， k），使得 i < j < k 并且 nums[i] < nums[j] < nums[k] ，则返回 true 。如果不存在此类索引，则返回 false 。

该题本质上考查的是最长上升子序列问题，通常情况我们使用单调栈数据结构，然后结合二分查找的方式进行解题即可。但是本题只是一个简化的版本，只需要找任意一个符合题意的三重索引即可，所以相对来说比较简单。

我们只需要用两个变量 first 和 second 来表示第一个数字，第二个数字，然后从左到右依次遍历 nums 数组即可 ：

* 如果当前的数字小于等于 first ，那么就更新 first 为当前数字
* 否则如果当前数字小于等于 second ，那么就更新 seconde 为当前数字
* 如果当前数字大于 second ，表示存在符合题目的三重索引关系，就直接返回 True 
* 如果遍历结束没有发现，直接返回 False 

时间复杂度为 O(N) ，空间复杂度为 O(1) 。该题还可以使用双向遍历的方法进行解题，但是这种方法的空间复杂度会变成 O(N^2) 。

### 解答

	class Solution(object):
	    def increasingTriplet(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: bool
	        """
	        first = second = float("inf")
	        for n in nums:
	            if n <= first:
	                first = n
	            elif n <= second:
	                second = n
	            else:
	                return True
	        return False


### 运行结果

	Runtime: 578 ms, faster than 79.79% of Python online submissions for Increasing Triplet Subsequence.
	Memory Usage: 23 MB, less than 36.70% of Python online submissions for Increasing Triplet Subsequence.

### 原题链接


https://leetcode.com/problems/increasing-triplet-subsequence/

您的支持是我最大的动力
