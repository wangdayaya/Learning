leetcode  55. Jump Game（python）




### 描述


You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position. Return true if you can reach the last index, or false otherwise.


Example 1:

	Input: nums = [2,3,1,1,4]
	Output: true
	Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

	
Example 2:

	Input: nums = [3,2,1,0,4]
	Output: false
	Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.




Note:

	1 <= nums.length <= 10^4
	0 <= nums[i] <= 10^5


### 解析

根据题意，给定一个整数数组 nums 。人最初位于数组的第一个索引处，数组中的每个元素表示人在该位置能往右跳跃的最长长度。如果人可以通过不断跳跃到达最后一个索引，则返回 True ，否则返回 False ，每一次跳跃的长度不能超过在所处位置可跳跃的最长长度。

通过读题，和观察题目对于 nums 的条件限制，我们发现 nums 的最长长度为 10^4 ，切每个元素的最大值为 10^5 ，所以这肯定不能使用常规的递归或者暴力，我们对于这种有规律的大数据量的题目，一般使用贪心即可。

其实我们细想一下这道题的规律就会发现比较清晰，当我们在某个索引位置 i 的位置上，只要 i + nums[i]  的结果大于等于最后一个索引 len(nums) - 1 即可直接返回 True ，表示我们在 i 索引位置上可以直接达到数组的最后位置上。

所以我们只要从左到右遍历 nums 中的每个索引位置，并维护一个变量 farthest 表示当前位置可以到达的最远的位置。如果当前的位置索引 i 小于等于 farthest ，那么我们就可以从当前位置通过跳跃可以到达该位置，此时我们更新 farthest 。如果 farthest 大于等于数组最后位置索引，那就说明最后一个位置可达，我们就可以直接返回 True 即可。反之如果在遍历结束后，最后一个位置仍然不可达，我们就返回 False 。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。


### 解答

	class Solution(object):
	    def canJump(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: bool
	        """
	        N = len(nums)
	        farthest = 0
	        for i,n in enumerate(nums):
	            if i <= farthest:
	                farthest = max(farthest, i+n)
	                if farthest >= N - 1:
	                    return True
	        return False

### 运行结果

	Runtime 391 ms , Beats 89.25%
	Memory 14.8 MB , Beats 10.75%

### 原题链接

https://leetcode.com/problems/jump-game/description/


您的支持是我最大的动力
