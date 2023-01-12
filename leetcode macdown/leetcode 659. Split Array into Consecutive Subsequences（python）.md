leetcode  659. Split Array into Consecutive Subsequences（python）


### 描述

You are given an integer array nums that is sorted in non-decreasing order. Determine if it is possible to split nums into one or more subsequences such that both of the following conditions are true:

* Each subsequence is a consecutive increasing sequence (i.e. each integer is exactly one more than the previous integer).
* All subsequences have a length of 3 or more.

Return true if you can split nums according to the above conditions, or false otherwise. A subsequence of an array is a new array that is formed from the original array by deleting some (can be none) of the elements without disturbing the relative positions of the remaining elements. (i.e., [1,3,5] is a subsequence of [1,2,3,4,5] while [1,3,2] is not).



Example 1:

	Input: nums = [1,2,3,3,4,5]
	Output: true
	Explanation: nums can be split into the following subsequences:
	[1,2,3,3,4,5] --> 1, 2, 3
	[1,2,3,3,4,5] --> 3, 4, 5

	
Example 2:


	Input: nums = [1,2,3,3,4,4,5,5]
	Output: true
	Explanation: nums can be split into the following subsequences:
	[1,2,3,3,4,4,5,5] --> 1, 2, 3, 4, 5
	[1,2,3,3,4,4,5,5] --> 3, 4, 5

Example 3:

	Input: nums = [1,2,3,4,4,5]
	Output: false
	Explanation: It is impossible to split nums into consecutive increasing subsequences of length 3 or more.



Note:


	1 <= nums.length <= 10^4
	-1000 <= nums[i] <= 1000
	nums is sorted in non-decreasing order.

### 解析

根据题意，给定一个整数数组 nums，它按非递减顺序排序。确定是否可以将 nums 拆分为一个或多个子序列，以使以下两个条件都满足：

* 每个子序列都是一个连续递增的序列（即每个整数正好比前一个整数大一）
* 所有子序列的长度都为 3 或更长

如果可以根据上述条件拆分 nums，则返回 true，否则返回 false 。

这道题考察的是贪心，我们定义了两个字典，一个是 remaining ，remaining[i] 记录的是数字 i 剩下可用的个数，另一个是 d ，d[i] 记录的是以数字 i 为结尾的长度至少为 3 且满足刚好递增 1 的不同的子序列个数，然后我们不断从左往右遍历 nums 中的元素，不断更新 remaining 和 d ，如果遍历到某个数字的时候无法构成符合题意的子序列直接返回 False ，如果正常遍历结束返回 True ，详细结果结合下面代码和注释理解更直观。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。
### 解答

	class Solution(object):
	    def isPossible(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: bool
	        """
	        remaining = collections.Counter(nums)
	        d = collections.Counter()
	        for i in nums:
	            if not remaining[i]:  # 如果没有可用的数字，直接跳过
	                continue
	            remaining[i] -= 1  # 可用数字的数量减一
	            if d[i - 1] > 0:  # 如果存在以前一个数字 i-1 结尾的长度最小为 3 的子序列，那么可以将当前的数字拼接其后面，更新 d[i-1] 和 d[i]
	                d[i - 1] -= 1
	                d[i] += 1
	            elif remaining[i + 1] and remaining[i + 2]:  # 否则向后找 i+1 和 i+2 自行拼接新的长度为 3 的子序列，更新 remaining 和 d[i+2]
	                remaining[i + 1] -= 1
	                remaining[i + 2] -= 1
	                d[i + 2] += 1
	            else:  # 如果上面两种情况都不存在，说明当前数字无法构成符合题意的子序列，直接返回 False 即可 
	                return False
	        return True

### 运行结果

	Runtime: 525 ms, faster than 86.54% of Python online submissions for Split Array into Consecutive Subsequences.
	Memory Usage: 15 MB, less than 23.08% of Python online submissions for Split Array into Consecutive Subsequences.


### 原题链接

	https://leetcode.com/problems/split-array-into-consecutive-subsequences/



您的支持是我最大的动力
