leetcode  2369. Check if There is a Valid Partition For The Array（python）




### 描述


You are given a 0-indexed integer array nums. You have to partition the array into one or more contiguous subarrays. We call a partition of the array valid if each of the obtained subarrays satisfies one of the following conditions:

* The subarray consists of exactly 2 equal elements. For example, the subarray [2,2] is good.
* The subarray consists of exactly 3 equal elements. For example, the subarray [4,4,4] is good.
* The subarray consists of exactly 3 consecutive increasing elements, that is, the difference between adjacent elements is 1. For example, the subarray [3,4,5] is good, but the subarray [1,3,5] is not.

Return true if the array has at least one valid partition. Otherwise, return false.


Example 1:

	Input: nums = [4,4,4,5,6]
	Output: true
	Explanation: The array can be partitioned into the subarrays [4,4] and [4,5,6].
	This partition is valid, so we return true.

	
Example 2:

	Input: nums = [1,1,1,2]
	Output: false
	Explanation: There is no valid partition for this array.




Note:

	2 <= nums.length <= 10^5
	1 <= nums[i] <= 10^6


### 解析

根据题意，给定一个 0 索引的整数数组 nums。 要求必须将数组划分为一个或多个连续的子数组。如果每个划分的子数组满足以下条件之一，我们称数组的分区有效：

* 子数组正好由 2 个相等的元素组成。 例如，子数组 [2,2] 是好的。
* 子数组正好由 3 个相等的元素组成。 例如，子数组 [4,4,4] 很好。
* 子数组正好由 3 个连续递增的元素组成，即相邻元素的差为1。例如子数组 [3,4,5] 是好的，但子数组 [1,3,5] 不是。

如果 nums 经过划分后的所有子数组符合上面条件返回 true。 否则返回 false 。

这道题其实就是考察动态规划，我们可以定义 dp ，dp[i] 表示前 i+1 个字符是否能分割合法，然后我们先使用上述的三条规则对 dp 前三个元素进行判断，然后我们再从 nums 的第四个元素开始不断遍历，并且回顾并判断 dp[i-1]、dp[i-2]、dp[i-3] 和 nums[i-2]、nums[i-1]、nums[i] 之间的关系是否合法来更新 dp[i] 。遍历结束返回 dp[-1] 即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def validPartition(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: bool
	        """
	        N = len(nums)
	        dp = [False] * N
	        if nums[0] == nums[1]:
	            dp[1] = True
	        if len(nums) > 2 and nums[0] == nums[1] and nums[1] == nums[2]:
	            dp[2] = True
	        if len(nums) > 2 and nums[1] - nums[0] == 1 and nums[2] - nums[1] == 1:
	            dp[2] = True
	        for i in range(3, len(nums)):
	            dp[i] = (dp[i - 2] and nums[i - 1] == nums[i]) \
	                    or (dp[i - 3] and nums[i - 2] == nums[i - 1] and nums[i-1] ==  nums[i]) \
	                    or (dp[i - 3] and nums[i - 1] - nums[i - 2] == 1 and  nums[i] - nums[i - 1] == 1)
	        return dp[-1]

### 运行结果

	117 / 117 test cases passed.
	Status: Accepted
	Runtime: 1115 ms
	Memory Usage: 26.6 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-305/problems/check-if-there-is-a-valid-partition-for-the-array/


您的支持是我最大的动力
