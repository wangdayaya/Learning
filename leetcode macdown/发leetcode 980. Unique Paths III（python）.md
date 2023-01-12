leetcode  980. Unique Paths III（python）

### 描述


You are given an m x n integer array grid where grid[i][j] could be:



* 1 representing the starting square. There is exactly one starting square.
* 2 representing the ending square. There is exactly one ending square.
* 0 representing empty squares we can walk over.
* -1 representing obstacles that we cannot walk over.


Return the number of 4-directional walks from the starting square to the ending square, that walk over every non-obstacle square exactly once.




Example 1:

![](https://assets.leetcode.com/uploads/2021/08/02/lc-unique1.jpg)

	Input: grid = [[1,0,0,0],[0,0,0,0],[0,0,2,-1]]
	Output: 2
	Explanation: We have the following two paths: 
	1. (0,0),(0,1),(0,2),(0,3),(1,3),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2)
	2. (0,0),(1,0),(2,0),(2,1),(1,1),(0,1),(0,2),(0,3),(1,3),(1,2),(2,2)

	
Example 2:


![](https://assets.leetcode.com/uploads/2021/08/02/lc-unique2.jpg)

	Input: grid = [[1,0,0,0],[0,0,0,0],[0,0,0,2]]
	Output: 4
	Explanation: We have the following four paths: 
	1. (0,0),(0,1),(0,2),(0,3),(1,3),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2),(2,3)
	2. (0,0),(0,1),(1,1),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,3),(1,3),(2,3)
	3. (0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(1,1),(0,1),(0,2),(0,3),(1,3),(2,3)
	4. (0,0),(1,0),(2,0),(2,1),(1,1),(0,1),(0,2),(0,3),(1,3),(1,2),(2,2),(2,3)


Example 3:

![](https://assets.leetcode.com/uploads/2021/08/02/lc-unique3-.jpg)

	Input: grid = [[0,1],[2,0]]
	Output: 0
	Explanation: There is no path that walks over every empty square exactly once.
	Note that the starting and ending square can be anywhere in the grid.




Note:

	m == grid.length
	n == grid[i].length
	1 <= m, n <= 20
	1 <= m * n <= 20
	-1 <= grid[i][j] <= 2
	There is exactly one starting cell and one ending cell.


### 解析

根据题意，就是给出了一个 MxN 大小的矩阵，每个格子可能有有四种数字表示不同的含义：

* 1 表示的是开始位置
* 2 表示的是结束为止
* 0 表示的是可以正常行走的空位置
* -1 表示的是无法跨越的障碍位置

题目要求我们要从开始位置走，到结束位置停下来，有多少种不同的走法可以将所有的空位置都经过一次。

有经验的通知一看这个找路径的题就知道得动态规划，但是这个题的 M 和 N 比较小，用回溯的思想也能找到可能路径，通过写递归函数来进行求解：

* 先对 grid 进行遍历，找到开始位置记为 [x,y] ，记录空格子的个数为 empty
* 使用递归函数 dfs 来搜索路径，这里面写法比较常规，就是 x 和 y 要在合法范围内进行上下左右的一步的走动，就是有一点技巧，为了递归时不重走来时的路，会暂时将当前的位置设置为 -1 ，在递归结束再恢复为 0 ，当走到结束位置且 empty== -1 表示空格都经过一次的时候才返回 1 ，否则其他情况都返回 0 
* 递归结束得到结果就为答案




### 解答
				
	class Solution(object):
	    def uniquePathsIII(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        def dfs(x,y,grid,empty):
	            if x<0 or x>=M or y<0 or y>=N:
	                return 0
	            if grid[x][y] == -1:
	                return 0
	            if grid[x][y] == 2:
	                if empty == -1:
	                    return 1
	                return 0
	            grid[x][y] = -1
	            count = dfs(x-1, y, grid, empty-1) + dfs(x+1, y, grid, empty-1) + dfs(x, y-1, grid, empty-1) + dfs(x, y+1, grid, empty-1)
	            grid[x][y] = 0
	            return count
	
	        M = len(grid)
	        N = len(grid[0])
	        empty = 0
	        for i in range(M):
	            for j in range(N):
	                if grid[i][j] == 1:
	                    x,y = i,j
	                elif grid[i][j] == 0:
	                    empty += 1
	        return dfs(x,y,grid,empty)
	 
            	      
			
### 运行结果
	
	Runtime: 85 ms, faster than 14.29% of Python online submissions for Unique Paths III.
	Memory Usage: 13.3 MB, less than 69.05% of Python online submissions for Unique Paths III.

### 相似题

* [62. Unique Paths](https://leetcode.com/problems/unique-paths/)
* [63. Unique Paths II](https://leetcode.com/problems/unique-paths-ii/)

原题链接：https://leetcode.com/problems/unique-paths-iii/



您的支持是我最大的动力
