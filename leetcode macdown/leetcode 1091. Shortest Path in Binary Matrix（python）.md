leetcode  1091. Shortest Path in Binary Matrix（python）




### 描述

Given an n x n binary matrix grid, return the length of the shortest clear path in the matrix. If there is no clear path, return -1.

A clear path in a binary matrix is a path from the top-left cell (i.e., (0, 0)) to the bottom-right cell (i.e., (n - 1, n - 1)) such that:

* All the visited cells of the path are 0.
* All the adjacent cells of the path are 8-directionally connected (i.e., they are different and they share an edge or a corner).


The length of a clear path is the number of visited cells of this path.





Example 1:


![](https://assets.leetcode.com/uploads/2021/02/18/example1_1.png)

	Input: grid = [[0,1],[1,0]]
	Output: 2
	
Example 2:

![](https://assets.leetcode.com/uploads/2021/02/18/example2_1.png)

	Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
	Output: 4


Example 3:

	Input: grid = [[1,0,0],[1,1,0],[1,1,0]]
	Output: -1



Note:


	n == grid.length
	n == grid[i].length
	1 <= n <= 100
	grid[i][j] is 0 or 1

### 解析

根据题意，给定一个 n x n 二进制矩阵 grid ，返回矩阵中最短畅通路径的长度。 如果没有明确的路径，则返回 -1 。

二元矩阵中的畅通路径是从左上角单元格到右下角单元格的路径，畅通路径的长度是该路径的访问单元数。要求：

* 路径的所有访问单元格都是 0
* 路径的所有相邻单元都是和相邻的 8 个方向连接的

这道题一看就是用 BFS 解题的，因为题目要求我们从左上角运动到右下角，这个过程就是先找左上角周围的 8 个相邻位置，然后再找相邻位置的相邻 8 个位置，相当于是一圈一圈的等高线，这就是一个典型的 BFS 过程，最常见的 BFS 解决方法就是使用队列，然后对每一圈等高线上的单元格计算距离起点的长度，然后再找他们周围 8 个方向的位置存入队列中，最后返回第一次出现右下角单元格时的距离即为最小值距离。如果无法到达右下角直接返回 -1 即可。

时间复杂度为 O(N * \8) ，空间复杂度为 O(N) ，N 表示单元格数量。


### 解答
				

	class Solution(object):
	    def shortestPathBinaryMatrix(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        N = len(grid)
	        if grid[0][0] == 1 or grid[N - 1][N - 1] == 1: return -1
	        queue = [[0, 0, 1]]
	        grid[0][0] = 1
	        while queue:
	            n = len(queue)
	            while n > 0:
	                i, j, L = queue.pop(0)
	                if i == N - 1 and j == N - 1:
	                    return L
	                for x, y in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
	                    if -1 < i + x < N and -1 < j + y < N and grid[i + x][j + y] == 0:
	                        queue.append([i + x, j + y, L + 1])
	                        grid[i + x][j + y] = 1
	                n -= 1
	        return -1
	        
            	      
			
### 运行结果


	Runtime: 823 ms, faster than 34.46% of Python online submissions for Shortest Path in Binary Matrix.
	Memory Usage: 13.7 MB, less than 91.95% of Python online submissions for Shortest Path in Binary Matrix.

### 原题链接



https://leetcode.com/problems/shortest-path-in-binary-matrix/


您的支持是我最大的动力
