leetcode  2304. Minimum Path Cost in a Grid（python）




### 描述


You are given a 0-indexed m x n integer matrix grid consisting of distinct integers from 0 to m * n - 1. You can move in this matrix from a cell to any other cell in the next row. That is, if you are in cell (x, y) such that x < m - 1, you can move to any of the cells (x + 1, 0), (x + 1, 1), ..., (x + 1, n - 1). Note that it is not possible to move from cells in the last row.

Each possible move has a cost given by a 0-indexed 2D array moveCost of size (m * n) x n, where moveCost[i][j] is the cost of moving from a cell with value i to a cell in column j of the next row. The cost of moving from cells in the last row of grid can be ignored.

The cost of a path in grid is the sum of all values of cells visited plus the sum of costs of all the moves made. Return the minimum cost of a path that starts from any cell in the first row and ends at any cell in the last row.

![](https://assets.leetcode.com/uploads/2022/04/28/griddrawio-2.png)




Example 1:

	Input: grid = [[5,3],[4,0],[2,1]], moveCost = [[9,8],[1,5],[10,12],[18,6],[2,4],[14,3]]
	Output: 17
	Explanation: The path with the minimum possible cost is the path 5 -> 0 -> 1.
	- The sum of the values of cells visited is 5 + 0 + 1 = 6.
	- The cost of moving from 5 to 0 is 3.
	- The cost of moving from 0 to 1 is 8.
	So the total cost of the path is 6 + 3 + 8 = 17.

	
Example 2:

	Input: grid = [[5,1,2],[4,0,3]], moveCost = [[12,10,15],[20,23,8],[21,7,1],[8,1,13],[9,10,25],[5,3,2]]
	Output: 6
	Explanation: The path with the minimum possible cost is the path 2 -> 3.
	- The sum of the values of cells visited is 2 + 3 = 5.
	- The cost of moving from 2 to 3 is 1.
	So the total cost of this path is 5 + 1 = 6.





Note:


	m == grid.length
	n == grid[i].length
	2 <= m, n <= 50
	grid consists of distinct integers from 0 to m * n - 1.
	moveCost.length == m * n
	moveCost[i].length == n
	1 <= moveCost[i][j] <= 100

### 解析

根据题意，给定一个 0 索引的 m x n 整数矩阵 grid ，由 0 到 m * n - 1 的不同整数组成。您可以在此矩阵中从一个单元格移动到下一行的任何其他单元格。 需要注意的是，到最后一行的单元格停止移动。 每个可能的移动都有一个由大小为 (m * n) x n 的 0 索引 2D 数组 moveCost 给出的代价，其中 moveCost[i][j] 是从值为 i 的单元格移动到第 j 列中的单元格的代价。网格中路径的成本是访问的所有单元格值的总和加上所有移动的成本总和。 返回从第一行的任何单元格开始到最后一行的任何单元格结束的路径的最小代价。


### 解答
				

	class Solution(object):
	    def minPathCost(self, grid, moveCost):
	        """
	        :type grid: List[List[int]]
	        :type moveCost: List[List[int]]
	        :rtype: int
	        """
	        M = len(grid)
	        N = len(grid[0])
	        dp = [[float('inf')] * N for _ in range(M)]
	        dp[0] = grid[0]
	        for i in range(1, M):
	            for j in range(N):
	                for k in range(N):
	                    dp[i][j] = min(dp[i][j], grid[i][j] + dp[i-1][k] + moveCost[grid[i-1][k]][j])
	        return min(dp[-1])
            	      
			
### 运行结果

	34 / 34 test cases passed.
	Status: Accepted
	Runtime: 2733 ms
	Memory Usage: 18.2 MB


### 原题链接


https://leetcode.com/contest/weekly-contest-297/problems/minimum-path-cost-in-a-grid/


您的支持是我最大的动力
