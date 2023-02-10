leetcode  1162. As Far from Land as Possible（python）




### 描述


Given an n x n grid containing only values 0 and 1, where 0 represents water and 1 represents land, find a water cell such that its distance to the nearest land cell is maximized, and return the distance. If no land or water exists in the grid, return -1.

The distance used in this problem is the Manhattan distance: the distance between two cells (x<sub>0</sub>, y<sub>0</sub>) and (x<sub>1</sub>, y<sub>1</sub>) is |x<sub>0</sub> - x<sub>1</sub>| + |y<sub>0</sub> - y<sub>1</sub>|.




Example 1:

![](https://assets.leetcode.com/uploads/2019/05/03/1336_ex1.JPG)

	Input: grid = [[1,0,1],[0,0,0],[1,0,1]]
	Output: 2
	Explanation: The cell (1, 1) is as far as possible from all the land with distance 2.

	
Example 2:


![](https://assets.leetcode.com/uploads/2019/05/03/1336_ex2.JPG)

	Input: grid = [[1,0,0],[0,0,0],[0,0,0]]
	Output: 4
	Explanation: The cell (2, 2) is as far as possible from all the land with distance 4.




Note:

	n == grid.length
	n == grid[i].length
	1 <= n <= 100
	grid[i][j] is 0 or 1


### 解析

根据题意，给定一个仅包含值 0 和 1 的大小为 n x n 的格网 grid ，其中 0 表示水，1 表示土地，找到一个水格子，使其到最近陆地格子的距离最大化，并返回距离。如果网格中不存在土地或水，则返回 -1 ，需要注意的是此问题中提到的距离是曼哈顿距离，也就是对于两个点来说，他们之间的距离是水平距离的绝对值加垂直距离的绝对值。

读完这道题可能有的同学会被曼哈顿距离这个条件迷惑住，其实这个就是个干扰项，我们不用考虑它也可以解决本题，想要找出一个格子其距离最近的陆地的距离最大化，这个问题我们可以转化为一个 BFS 的问题，只要先找到所有的陆地格子，然后再找到那个距离某个的陆地格子最远的水格子即可。

借用这位[大佬的解说图](https://leetcode.cn/problems/as-far-from-land-as-possible/solution/jian-dan-java-miao-dong-tu-de-bfs-by-sweetiee/)，如下，我们先找到所有的陆地格子，然后从各个陆地同时开始一层层地向水面扩散，那么最后扩散到的水格子就是最远的海洋，并且这个海洋肯定是被离他最近的陆地是最远的，这样一来，使用多源的 BFS 即可完成题目解答。

![](https://pic.leetcode-cn.com/367df5172fd16d2637e591f6586d146772758438c66660c86719ffb2d36eb14d-image.png)


N 为网格中的格子数量，时间复杂度为 O(N)，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def maxDistance(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        land = []
	        result = -1
	        N = len(grid)
	        for i in range(N):
	            for j in range(N):
	                if grid[i][j] == 1:
	                    land.append([i, j])
	        if len(land) == 0 or len(land) == N * N:
	            return result
	        while land:
	            result += 1
	            L = len(land)
	            for _ in range(L):
	                x, y = land.pop(0)
	                if x + 1 < N and grid[x + 1][y] == 0:
	                    grid[x + 1][y] = N
	                    land.append([x + 1, y])
	                if x - 1 >= 0 and grid[x - 1][y] == 0:
	                    grid[x - 1][y] = N
	                    land.append([x - 1, y])
	                if y + 1 < N and grid[x][y + 1] == 0:
	                    grid[x][y + 1] = N
	                    land.append([x, y + 1])
	                if y - 1 >= 0 and grid[x][y - 1] == 0:
	                    grid[x][y - 1] = N
	                    land.append([x, y - 1])
	        return result

### 运行结果

	Runtime 477 ms，Beats 85.45%
	Memory 14.3 MB， Beats 54.55%

### 原题链接

https://leetcode.com/problems/as-far-from-land-as-possible/


您的支持是我最大的动力
