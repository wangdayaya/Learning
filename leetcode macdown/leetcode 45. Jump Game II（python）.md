leetcode  45. Jump Game II（python）




### 描述

You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0]. Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where:

* 0 <= j <= nums[i] and
* i + j < n

Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can reach nums[n - 1].



Example 1:


	Input: nums = [2,3,1,1,4]
	Output: 2
	Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
	
Example 2:


	Input: nums = [2,3,0,1,4]
	Output: 2




Note:


	1 <= nums.length <= 10^4
	0 <= nums[i] <= 1000

### 解析

根据题意，给定一个长度为 n 的 0 索引整数数组 nums 。初始位置为 nums[0] 。 每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处，但要满足:

* 0 <= j <= nums[i] 并且
* i + j < n

返回到达 nums[n - 1] 的最小跳跃次数。

这道题考查的就是贪心思想，我们要想找到从开始跳到最后的最少次数，那么我们可以从左往右依次遍历数组 nums ，在每个位置上找到可达的最远位置，那么就可以在最短的时间内找到答案。

如例子 1 中的 nums = [2,3,1,1,4] ，我们初始位置在 0 ，此时，我们可达的位置有索引 1 和 2 ，索引 1 的值为 3 ，从索引 1 出发最远的可达位置在索引 4 ，索引 2 的值为 1 ，从索引 2 出发的最远可达位置在索引 3 ，所以我们选择从索引 0 跳到索引 2 ，跳跃次数加一，以此类推，最后在遍历完 nums 时候得到最少的跳跃次数。

需要注意的是我们在遍历 nums 的时候，我们不访问最后一个元素，因为在访问最后一个元素之前，我们的上一个选择的索引位置的可达最远位置一定大于等于最后一个位置，否则就无法跳到最后一个位置了。如果访问最后一个元素，我们会增加一次「不必要的跳跃次数」。
 
 
 时间复杂度为 O(N) ，空间复杂度为 O(1)。

### 解答

	class Solution(object):
	    def jump(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        n = len(nums)
	        maxP = 0
	        end = 0
	        for i in range(n-1):
	            maxP = max(maxP, i + nums[i])
	            if i == end:
	                end = maxP
	                result += 1
	        return result

### 运行结果

	Runtime 84 ms，Beats 98.33%
	Memory 14.4 MB，Beats 69.25%

### 原题链接

https://leetcode.com/problems/jump-game-ii/


您的支持是我最大的动力
