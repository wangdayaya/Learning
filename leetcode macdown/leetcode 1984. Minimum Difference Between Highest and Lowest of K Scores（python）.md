leetcode  1984. Minimum Difference Between Highest and Lowest of K Scores（python）

### 描述


	You are given a 0-indexed integer array nums, where nums[i] represents the score of the ith student. You are also given an integer k.
	
	Pick the scores of any k students from the array so that the difference between the highest and the lowest of the k scores is minimized.
	
	Return the minimum possible difference.


Example 1:

	Input: nums = [90], k = 1
	Output: 0
	Explanation: There is one way to pick score(s) of one student:
	- [90]. The difference between the highest and lowest score is 90 - 90 = 0.
	The minimum possible difference is 0.

	
Example 2:


	Input: nums = [9,4,1,7], k = 2
	Output: 2
	Explanation: There are six ways to pick score(s) of two students:
	- [9,4,1,7]. The difference between the highest and lowest score is 9 - 4 = 5.
	- [9,4,1,7]. The difference between the highest and lowest score is 9 - 1 = 8.
	- [9,4,1,7]. The difference between the highest and lowest score is 9 - 7 = 2.
	- [9,4,1,7]. The difference between the highest and lowest score is 4 - 1 = 3.
	- [9,4,1,7]. The difference between the highest and lowest score is 7 - 4 = 3.
	- [9,4,1,7]. The difference between the highest and lowest score is 7 - 1 = 6.
	The minimum possible difference is 2.



Note:

	1 <= k <= nums.length <= 1000
	0 <= nums[i] <= 10^5



### 解析

根据题意，给出一个从 0 开始索引的整数列表 nums ，其中的 nums[i] 表示的是第 i 个学生的分数，然后又给出了一个整数 k 。从 nums 中选择任何 k 个学生的分数，以便使 k 个分数中最高和最低之间的差异最小化，返回最小可能的差值。

最简单的肯定是暴力解法，使用内置函数 itertools.combinations 得到所有的组合，然后找出所有组合的最小的差值。但是结果肯定是超时的，哪有这么简单题呢，因为 k 最大是 1000 ，那组合的结果可是大了去了。
### 解答
				

	class Solution(object):
	    def minimumDifference(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        if k == 1: return 0
	        if k>=len(nums): return max(nums) - min(nums)
	        result = 10**5
	        for cb in itertools.combinations(nums, k):
	            result = min(result, max(cb) - min(cb))
	        return result
         
			
### 运行结果


	Time Limit Exceeded


### 解析

可以换一种思路，因为要找每个组合的最大值和最小值的差值，我们只需要把 nums 按照从小到大的顺序进行排序，然后比较从左往右 k 长度的不同子列表的最大值和最小值的差值即可，遍历 range(k-1, len(nums)) 中的每个索引 i ，计算当前组合的最大值 nums[i]  和最小值 nums[i]-nums[i-k+1] 的差值，并使用 result 记录最小值，遍历结束返回 result 即可。


### 解答

	class Solution(object):
	    def minimumDifference(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        if k == 1: return 0
	        if k>=len(nums): return max(nums) - min(nums)
	        result = 10**5
	        nums.sort()
	        print(nums)
	        for i in range(k-1, len(nums)):
	            result = min(result, nums[i]-nums[i-k+1])
	        return result
	            

### 运行结果

	Runtime: 104 ms, faster than 54.10% of Python online submissions for Minimum Difference Between Highest and Lowest of K Scores.
	Memory Usage: 13.7 MB, less than 9.84% of Python online submissions for Minimum Difference Between Highest and Lowest of K Scores.

原题链接：https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/



您的支持是我最大的动力
