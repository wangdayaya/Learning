leetcode 473. Matchsticks to Square （python）




### 描述

You are given an integer array matchsticks where matchsticks[i] is the length of the ith matchstick. You want to use all the matchsticks to make one square. You should not break any stick, but you can link them up, and each matchstick must be used exactly one time. Return true if you can make this square and false otherwise.



Example 1:


![](https://assets.leetcode.com/uploads/2021/04/09/matchsticks1-grid.jpg)

	Input: matchsticks = [1,1,2,2,2]
	Output: true
	Explanation: You can form a square with length 2, one side of the square came two sticks with length 1.
	
Example 2:


	Input: matchsticks = [3,3,3,3,4]
	Output: false
	Explanation: You cannot find a way to form a square with all the matchsticks.




Note:

	1 <= matchsticks.length <= 15
	1 <= matchsticks[i] <= 10^8



### 解析

根据题意，给定一个整数数组 matchsticks，其中 matchsticks[i] 是第 i 个火柴的长度。 要求用所有的火柴做一个正方形，不应该折断任何一根火柴，但可以把它们连起来，而且每根火柴都必须使用一次。如果你可以使这个正方形返回 true，否则返回 false。

因为限制条件中 matchsticks 最长只有 15 ，所以我们碰到这种题直接使用记忆化的 DFS 来解题即可。首先我们知道当 matchsticks 的总和不是 4 的倍数，那么肯定无法拼接成正方形，直接返回 False 即可。为了加速进行递归我们可以使用三个技巧：

* 使 matchsticks 降序排列，我们在递归的时候，因为我们用火柴拼接一条边的时候，用的火柴越长，使用火柴的数量越少这就使得我们的搜索量会大幅下降
* 我们在递归的时候使用注解 @cache 可以实现记忆化的 DFS ，可以减少时间
* 当计算某个边的长度大于正常的长度时候，说明这种方案已经不合适，直接返回 False 即可，也就是实现了剪枝的操作

时间复杂度为 O(4^N) ，空间复杂度为 O(N) 。
 
### 解答


	class Solution:
	    def makesquare(self, matchsticks: List[int]) -> bool:
	        matchsticks.sort(reverse=True)
	        N = len(matchsticks)
	        total = sum(matchsticks)
	        m = matchsticks
	        if total % 4 != 0:
	            return False
	        L = total // 4
	        @cache
	        def dfs(a, b, c, d, i):
	            if a == b == c == d == L:
	                return True
	            if a > L or b > L or c > L or d > L or i > N - 1:
	                return False
	            return dfs(a + m[i], b, c, d, i + 1) or dfs(a, b + m[i], c, d, i + 1) or dfs(a, b, c + m[i], d, i + 1) or dfs(a, b, c, d + m[i], i + 1)
	        return dfs(0, 0, 0, 0, 0)
	        
### 运行结果

	Runtime: 3417 ms, faster than 43.83% of Python3 online submissions for Matchsticks to Square.
	Memory Usage: 580.1 MB, less than 5.04% of Python3 online submissions for Matchsticks to Square.



### 原题链接

https://leetcode.com/problems/matchsticks-to-square/


您的支持是我最大的动力
