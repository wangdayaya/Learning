leetcode  219. Contains Duplicate II（python）




### 描述


Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.




Example 1:


	Input: nums = [1,2,3,1], k = 3
	Output: true
	
Example 2:

	Input: nums = [1,0,1,1], k = 1
	Output: true


Example 3:

	Input: nums = [1,2,3,1,2,3], k = 2
	Output: false



Note:

	1 <= nums.length <= 10^5
	-109 <= nums[i] <= 10^9
	0 <= k <= 10^5


### 解析

根据题意，给定一个整数数组 nums 和一个整数 k ，如果数组中有两个不同的索引 i 和 j ，使得 nums[i] == nums[j] 且 abs（i - j） <= k，则返回 True ，否则返回 False 。


这是一道比较简单入门的考查贪心的题目，我们注意到题目中 nums 的最大长度为 10^5 ，所以时间复杂度肯定要控制在 O(logN) 以内。因为主要是在比较同类数字的索引的最小距离，所以朴素的思路就是使用字典将每个数字出现的索引都记录下来，这样再对索引进行求距离的操作即可。

再具体一点就是，在用字典 d 存储每个数字对应的最新的索引，这里就用到了贪心思想，因为我们要从左往右遍历 nums ，所以我们要时刻保持某个数字对应的索引位置是最新的，只有这样当我们遍历到一个新的数字的时候，它如果在之前已经出现过，那么当前索引和之前出现过的索引进行比较得到的距离绝对值肯定是最小的。如果有出现题目描述的合法情况则直接返回 True ，否则就更新字典 d ，继续向右进行遍历找满足提议的合法情况，如果遍历结束说明没有，直接返回 False 。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def containsNearbyDuplicate(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: bool
	        """
	        d = {}
	        for i, n in enumerate(nums):
	            if n not in d:
	                d[n] = i
	            else:
	                if i - d[n] <= k:
	                    return True
	                d[n] = i
	        return False

### 运行结果

	Runtime: 486 ms, faster than 99.71% of Python online submissions for Contains Duplicate II.
	Memory Usage: 24.2 MB, less than 76.42% of Python online submissions for Contains Duplicate II.

### 原题链接


https://leetcode.com/problems/contains-duplicate-ii/



您的支持是我最大的动力
