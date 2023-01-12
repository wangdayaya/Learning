leetcode  377. Combination Sum IV（python）




### 描述



Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.

The test cases are generated so that the answer can fit in a 32-bit integer.

Example 1:

	Input: nums = [1,2,3], target = 4
	Output: 7
	Explanation:
	The possible combination ways are:
	(1, 1, 1, 1)
	(1, 1, 2)
	(1, 2, 1)
	(1, 3)
	(2, 1, 1)
	(2, 2)
	(3, 1)
	Note that different sequences are counted as different combinations.

	
Example 2:

	Input: nums = [9], target = 3
	Output: 0




Note:

	1 <= nums.length <= 200
	1 <= nums[i] <= 1000
	All the elements of nums are unique.
	1 <= target <= 1000


### 解析

根据题意，给定一个不同整数数组 nums 和一个目标整数 target ，返回加起来为 target 的可能组合的数量。

这道题很明显可以使用记忆化的 DFS 来完成解答，我们的定义函数 dfs(x) 为从 nums 中不断挑选数字（可以重复）可以构成 x 的方法数，像例子一所示，我们要求 dfs(4) ，假如我第一个数字从 nums 中选择了 1 ，这样我就要去求 dfs(3) ，假如我第一个数字选择了 2 ，这样我就要去求 dfs(2) ，假如我第一个数字选择了 3 ，这样我就要去求 dfs(1) ，所以 dfs(4) = dfs(3) + dfs(2) + dfs(1) ，然后递归求解  dfs(3) 、dfs(2) 、dfs(1) 。在递归过程中有很多运算都是重复的，我们把这些结果都存放在 m 中。

时间复杂度为 O(N\∗target)，空间复杂度为 O(target)。

当然了记忆化的 DFS 解法已经写出来，那就可以继续写动态规划的解法，核心都是一样的。

### 解答

	class Solution:
	    def combinationSum4(self, nums: List[int], target: int) -> int:
	        self.m = [-1] * (target + 1)
	        self.m[0] = 1
	        def dfs(total):
	            if total < 0:
	                return 0
	            if self.m[total] != -1:
	                return self.m[total]
	            result = 0
	            for n in nums:
	                result += dfs(total - n)
	            self.m[total] = result
	            return result
	        return dfs(target)

### 运行结果

	Runtime: 79 ms, faster than 23.53% of Python3 online submissions for Combination Sum IV.
	Memory Usage: 14.1 MB, less than 18.29% of Python3 online submissions for Combination Sum IV.

### 原题链接

https://leetcode.com/problems/combination-sum-iv/


您的支持是我最大的动力
