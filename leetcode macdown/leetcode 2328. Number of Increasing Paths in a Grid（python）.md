leetcode  2328. Number of Increasing Paths in a Grid（python）




### 描述


You are given an m x n integer matrix grid, where you can move from a cell to any adjacent cell in all 4 directions. Return the number of strictly increasing paths in the grid such that you can start from any cell and end at any cell. Since the answer may be very large, return it modulo 109 + 7. Two paths are considered different if they do not have exactly the same sequence of visited cells.


Example 1:

![](https://assets.leetcode.com/uploads/2022/05/10/griddrawio-4.png)

	Input: grid = [[1,1],[3,4]]
	Output: 8
	Explanation: The strictly increasing paths are:
	- Paths with length 1: [1], [1], [3], [4].
	- Paths with length 2: [1 -> 3], [1 -> 4], [3 -> 4].
	- Paths with length 3: [1 -> 3 -> 4].
	The total number of paths is 4 + 3 + 1 = 8.

	
Example 2:

	Input: grid = [[1],[2]]
	Output: 3
	Explanation: The strictly increasing paths are:
	- Paths with length 1: [1], [2].
	- Paths with length 2: [1 -> 2].
	The total number of paths is 2 + 1 = 3.





Note:
	
	m == grid.length
	n == grid[i].length
	1 <= m, n <= 1000
	1 <= m * n <= 10^5
	1 <= grid[i][j] <= 10^5



### 解析


根据题意，给定一个 m x n 整数矩阵网格 grid ，可以在其中从一个单元格移动到所有 4 个方向上的任何相邻单元格。返回网格中严格递增的路径数，可以从任何单元格开始并在任何单元格处结束。 由于答案可能非常大，因此以 10^9 + 7 为模返回。如果两条路径没有完全相同的访问单元序列，则认为它们是不同的。

因为数据量不大，所以我们使用动态规划，我们假设 dp[i][j] 为以 grid[i][j] 为元素递增路径终点的路径总数，初始化全部为 1 ，因为每个元素自身可以组成一个长度为 1 的路径，我们对所有的 grid[i][j]  进行排序，这样我们遍历每个元素 grid[i][j] ，然后按照严格递增路径的要求使用动态规划更新 dp[i][j] ，遍历结束，我们将 dp 中所有的值相加即可得到结果。

时间复杂度为 O(M\*Nlog(M\*N)) ，空间复杂度为 O(M\*N)




当然使用记忆化 DFS 也可以完成题目，原理和动态规划是一样的，只是计算量太大，必须加注解 @cache ，否则肯定会超时。

### 解答

	class Solution(object):
	    def countPaths(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        import numpy as np
	        MOD = 10 ** 9 + 7
	        M, N = len(grid), len(grid[0])
	        dp = [[1 for _ in range(N)] for _ in range(M)]
	        L = [[grid[i][j], i, j] for i in range(M) for j in range(N)]
	        L.sort()
	        for n, i, j in L:
	            for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
	                if 0 <= x < M and 0 <= y < N and grid[x][y] < n:
	                    dp[i][j] = (dp[i][j] + dp[x][y]) % MOD
	        return np.sum(dp) % MOD
			
### 运行结果

	36 / 36 test cases passed.
	Status: Accepted
	Runtime: 5375 ms
	Memory Usage: 46.1 MB


### 解答
				
	class Solution:
	    def countPaths(self, grid: List[List[int]]) -> int:
	        MOD = 10 ** 9 + 7
	        M, N = len(grid), len(grid[0])
	
	        @cache
	        def dfs(i, j):
	            result = 1
	            for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
	                if 0 <= x < M and 0 <= y < N and grid[x][y] > grid[i][j]:
	                    result += dfs(x, y)
	            return result % MOD
	
	        return sum(dfs(i, j) for i in range(M) for j in range(N)) % MOD

            	      
			
### 运行结果

	36 / 36 test cases passed.
	Status: Accepted
	Runtime: 4262 ms
	Memory Usage: 107.3 MB


### 原题链接


https://leetcode.com/contest/weekly-contest-300/problems/number-of-increasing-paths-in-a-grid/


您的支持是我最大的动力
