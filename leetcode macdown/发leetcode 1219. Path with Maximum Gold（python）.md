leetcode  1219. Path with Maximum Gold（python）

### 描述

In a gold mine grid of size m x n, each cell in this mine has an integer representing the amount of gold in that cell, 0 if it is empty.

Return the maximum amount of gold you can collect under the conditions:

Every time you are located in a cell you will collect all the gold in that cell.
From your position, you can walk one step to the left, right, up, or down.
You can't visit the same cell more than once.
Never visit a cell with 0 gold.
You can start and stop collecting gold from any position in the grid that has some gold.



Example 1:


	Input: grid = [[0,6,0],[5,8,7],[0,9,0]]
	Output: 24
	Explanation:
	[[0,6,0],
	 [5,8,7],
	 [0,9,0]]
	Path to get the maximum gold, 9 -> 8 -> 7.
	
Example 2:


	Input: grid = [[1,0,7],[2,0,6],[3,4,5],[0,3,0],[9,0,20]]
	Output: 28
	Explanation:
	[[1,0,7],
	 [2,0,6],
	 [3,4,5],
	 [0,3,0],
	 [9,0,20]]
	Path to get the maximum gold, 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7.



Note:

	
	m == grid.length
	n == grid[i].length
	1 <= m, n <= 15
	0 <= grid[i][j] <= 100
	There are at most 25 cells containing gold.

### 解析


根据题意，就是给出了一个金矿的矩阵 grid ，大小为 m*n ，每个位置 grid[i][j] 都有一个数字代表金子数量，你现在是一个掘金者，可以从金矿矩阵的任意一个位置出发，然后向上下左右四个方向都能行进，但是不能访问金子数为 0 ，同时不能走走过的位置，求能得到的金子数最多数量。

真很明显就是一个 DFS 问题，只要注意不会触发边界禁止条件就可以，以所有位置为起点得到能获得的金子的最大数量，即为 dp ，然后从 dp 中找出最大的数字，即为答案。
### 解答
				

	class Solution(object):
	    def getMaximumGold(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        m = len(grid)
	        n = len(grid[0])
	
	        def countGold(gold, i, j):
	            used.add((i, j))
	            dp[i][j] = max(dp[i][j], gold)
	            for x, y in (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1):
	                if 0 <= x < m and 0 <= y < n and grid[x][y] and (x, y) not in used:
	                    countGold(gold + grid[x][y], x, y)
	            used.discard((i, j))
	
	        dp = [[0] * n for _ in range(m)]
	        for i in range(m):
	            for j in range(n):
	                used = set()
	                if grid[i][j]:
	                    countGold(grid[i][j], i, j)
	        return max(c for row in dp for c in row)
            	      
			
### 运行结果

	Runtime: 1288 ms, faster than 71.68% of Python online submissions for Path with Maximum Gold.
	Memory Usage: 13.4 MB, less than 84.96% of Python online submissions for Path with Maximum Gold.


原题链接：https://leetcode.com/problems/path-with-maximum-gold/



您的支持是我最大的动力
