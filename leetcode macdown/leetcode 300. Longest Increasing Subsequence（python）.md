leetcode  300. Longest Increasing Subsequence（python）




### 描述


Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].


Example 1:

	Input: nums = [10,9,2,5,3,7,101,18]
	Output: 4
	Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

	
Example 2:

	Input: nums = [0,1,0,3,2,3]
	Output: 4


Example 3:

	Input: nums = [7,7,7,7,7,7,7]
	Output: 1



Note:

	1 <= nums.length <= 2500
	-10^4 <= nums[i] <= 10^4


### 解析

根据题意，给定一个整数数组 nums ，返回最长严格递增子序列的长度。子序列是可以通过删除一些元素或不删除元素而不改变剩余元素的顺序从数组派生的序列。 例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

这道题要找最长的严格递增子序列，很明显考察的就是动态规划，我们定义 dp[i] 表示前 i 个字符情况下，以  nums[i] 元素结尾的最长递增子序列的长度是多少。而要计算当前的 dp[i] ，我们需要遍历小于索引 i 的每个元素与 nums[i] 的关系，如果某个元素比 nums[i] 小，则根据转移方程更新 dp[i] :
	
	dp[i] = max(dp[i] , dp[j] + 1)
	
经过几乎双重循环的遍历，我们可以找出一个 dp ，dp[i] 就是以 nums[i] 为结尾的最长长度，所以结束返回 dp 中的最大值即可。时间复杂度为 O(N^2) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def lengthOfLIS(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        dp = [1] * len(nums)
	        for i,v in enumerate(nums):
	            for j in range(i):
	                if nums[j] < nums[i]:
	                    dp[i] = max(dp[i], dp[j] + 1)
	        return max(dp)

### 运行结果

	Runtime: 3298 ms, faster than 48.71% of Python online submissions for Longest Increasing Subsequence.
	Memory Usage: 13.9 MB, less than 15.36% of Python online submissions for Longest Increasing Subsequence.

### 原题链接


https://leetcode.com/problems/longest-increasing-subsequence/

您的支持是我最大的动力
