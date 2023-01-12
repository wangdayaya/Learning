leetcode  695. Max Area of Island（python）




### 描述


You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water. The area of an island is the number of cells with a value 1 in the island. Return the maximum area of an island in grid. If there is no island, return 0.


Example 1:

![](https://assets.leetcode.com/uploads/2021/05/01/maxarea1-grid.jpg)

	Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
	Output: 6
	Explanation: The answer is not 11, because the island must be connected 4-directionally.

	
Example 2:

	Input: grid = [[0,0,0,0,0,0,0,0]]
	Output: 0






Note:

	m == grid.length
	n == grid[i].length
	1 <= m, n <= 50
	grid[i][j] is either 0 or 1.


### 解析

根据题意，给定一个 m x n 二进制矩阵 grid 。 岛是由一组代表土地的数字 1 在 4 个方向（水平或垂直）连接出来的块。岛四个边缘都被水包围。岛的面积是岛上值为 1 的个数。返回网格中岛屿的最大面积。 如果没有岛，则返回 0 。

这道题考察的其实就是 BFS ，因为我们在找最大面积的岛的时候，肯定是以某个点的位置进行向四个方向扩散去蔓延找出岛面积的，所以这正好贴合了 BFS 的定义，然后我们遍历 grid 中所有的点，只要这个点是还没有探查过的 1 ，那么我们就进行 BFS 的扩展找出当前岛的面积，防止重复进行查找已经探查的地方，我们将已经探查之后的地方设置为 0 即可，然后去更新最大的岛面积 result 即可。

时间复杂度为 O(M\*N)，空降复杂度为 O(M\*N) 。

### 解答

	class Solution(object):
	    def maxAreaOfIsland(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        import numpy
	        if numpy.sum(grid) == 0:
	            return 0
	        M = len(grid)
	        N = len(grid[0])
	        result = 0
	        for i in range(M):
	            for j in range(N):
	                if grid[i][j] == 1:
	                    tmp = 0
	                    stack = [[i, j]]
	                    grid[i][j] = 0
	                    while stack:
	                        x, y = stack.pop(0)
	                        tmp += 1
	                        for dx, dy in (-1, 0), (0, 1), (1, 0), (0, -1):
	                            if 0 <= dx + x < M and 0 <= dy + y < N and grid[dx + x][dy + y] == 1:
	                                grid[dx + x][dy + y] = 0
	                                stack.append([dx + x, dy + y])
	                    result = max(result, tmp)
	        return result

### 运行结果

	Runtime: 279 ms, faster than 10.92% of Python online submissions for Max Area of Island.
	Memory Usage: 25.9 MB, less than 6.52% of Python online submissions for Max Area of Island.

### 原题链接


https://leetcode.com/problems/max-area-of-island/

您的支持是我最大的动力
