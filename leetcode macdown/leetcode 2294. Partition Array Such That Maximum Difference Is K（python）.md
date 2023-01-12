

### 描述

You are given an integer array nums and an integer k. You may partition nums into one or more subsequences such that each element in nums appears in exactly one of the subsequences.

Return the minimum number of subsequences needed such that the difference between the maximum and minimum values in each subsequence is at most k.

A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.



Example 1:

	Input: nums = [3,6,1,2,5], k = 2
	Output: 2
	Explanation:
	We can partition nums into the two subsequences [3,1,2] and [6,5].
	The difference between the maximum and minimum value in the first subsequence is 3 - 1 = 2.
	The difference between the maximum and minimum value in the second subsequence is 6 - 5 = 1.
	Since two subsequences were created, we return 2. It can be shown that 2 is the minimum number of subsequences needed.

	
Example 2:

	Input: nums = [1,2,3], k = 1
	Output: 2
	Explanation:
	We can partition nums into the two subsequences [1,2] and [3].
	The difference between the maximum and minimum value in the first subsequence is 2 - 1 = 1.
	The difference between the maximum and minimum value in the second subsequence is 3 - 3 = 0.
	Since two subsequences were created, we return 2. Note that another optimal solution is to partition nums into the two subsequences [1] and [2,3].


Example 3:

	Input: nums = [2,2,4,5], k = 0
	Output: 3
	Explanation:
	We can partition nums into the three subsequences [2,2], [4], and [5].
	The difference between the maximum and minimum value in the first subsequences is 2 - 2 = 0.
	The difference between the maximum and minimum value in the second subsequences is 4 - 4 = 0.
	The difference between the maximum and minimum value in the third subsequences is 5 - 5 = 0.
	Since three subsequences were created, we return 3. It can be shown that 3 is the minimum number of subsequences needed.

	


Note:

    1 <= nums.length <= 10^5
    0 <= nums[i] <= 10^5
    0 <= k <= 10^5


### 解析


根据题意，给定一个整数数组 nums 和一个整数 k 。 我们可以将 nums 划分为一个或多个子序列，以便 nums 中的每个元素恰好出现在其中一个子序列中。返回所需的最小子序列数，使得每个子序列中的最大值和最小值之差最多为 k 。

这道题其实考察的就是贪心和排序，因为最后要求的是子序列的个数，所以我们不用考虑每个子序列中的元素的顺序，首先我们将数组 nums 进行升序排序。然后定义一个子序列最小值 mn = nums[0] 和结果 result=1（因为最少的子序列数肯定为 1） ，不断遍历 nums 中的元素 nums[i] ，如果 nums[i] - mn 大于 k ，说明这部分可以形成一个子序列，结果 result 加一，同时更新 mn ，遍历结束得到的 result 即为结果。

时间复杂度为 O(N) ，空间复杂度为 O（1） 。

### 解答
				

	class Solution(object):
	    def partitionArray(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        nums.sort()
	        result = 1
	        mn = nums[0]
	        for i in range(1,len(nums)):
	            if nums[i] - mn > k:
	                result += 1
	                mn = nums[i]
	        return result
	
### 运行结果

	

	92 / 92 test cases passed.
	Status: Accepted
	Runtime: 1208 ms
	Memory Usage: 25.6 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-296/problems/partition-array-such-that-maximum-difference-is-k/

您的支持是我最大的动力
