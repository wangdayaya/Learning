leetcode  2257. Count Unguarded Cells in the Grid（python）

这是第 77 场双周赛的第二题，难度 Medium ，考察的是字符串的基本操作。



### 描述

You are given two integers m and n representing a 0-indexed m x n grid. You are also given two 2D integer arrays guards and walls where guards[i] = [rowi, coli] and walls[j] = [rowj, colj] represent the positions of the ith guard and jth wall respectively.

A guard can see every cell in the four cardinal directions (north, east, south, or west) starting from their position unless obstructed by a wall or another guard. A cell is guarded if there is at least one guard that can see it.

Return the number of unoccupied cells that are not guarded.



Example 1:

![](https://assets.leetcode.com/uploads/2022/03/10/example1drawio2.png)

	Input: m = 4, n = 6, guards = [[0,0],[1,1],[2,3]], walls = [[0,1],[2,2],[1,4]]
	Output: 7
	Explanation: The guarded and unguarded cells are shown in red and green respectively in the above diagram.
	There are a total of 7 unguarded cells, so we return 7.

	
Example 2:

![](https://assets.leetcode.com/uploads/2022/03/10/example2drawio.png)

	Input: m = 3, n = 3, guards = [[1,1]], walls = [[0,1],[1,0],[2,1],[1,2]]
	Output: 4
	Explanation: The unguarded cells are shown in green in the above diagram.
	There are a total of 4 unguarded cells, so we return 4.







Note:

	1 <= m, n <= 10^5
	2 <= m * n <= 10^5
	1 <= guards.length, walls.length <= 5 * 10^4
	2 <= guards.length + walls.length <= m * n
	guards[i].length == walls[j].length == 2
	0 <= rowi, rowj < m
	0 <= coli, colj < n
	All the positions in guards and walls are unique.


### 解析


根据题意，给定两个整数 m 和 n ，表示一个 0 索引的 m x n 网格。 给定两个二维整数数组 guards 和 walls ，其中 guards[i] = [row<sub>i</sub>, col<sub>i</sub>] 和 walls[j] = [row<sub>j</sub>, col<sub>j</sub>] 分别代表第 i 个保安和第 j 个墙的位置。

守卫可以从他们的位置开始看到四个主要方向（北、东、南或西）的每个单元，除非被墙或其他守卫阻挡。 返回未被看守的单元格的数量。

这道题其实就是一个遍历格子的问题，思路也比较简单，初始化一个全都是 0 的 grid ，然后将保安和墙的格子 2 表示这些位置已经有了站位或者阻挡，然后再遍历每个保安能看到的四个方向的格子，如果 grid[i][j] 上的数字为 0 或者 1 表示其还没有保卫或者已经保卫，然后将 grid[i][j] 设置为 1 即可，遍历结束只需要返回 grid 中还是 0 的格子数量即可。


时间复杂度为 O(m\*n) ，空间复杂度为 O(m\*n) 。

### 解答
				
	class Solution(object):
	    def countUnguarded(self, m, n, guards, walls):
	        """
	        :type m: int
	        :type n: int
	        :type guards: List[List[int]]
	        :type walls: List[List[int]]
	        :rtype: int
	        """
	        grid = [[0]*n for _ in range(m)]
	        for i,j in guards:
	            grid[i][j] = 2
	        for i,j in walls:
	            grid[i][j] = 2
	        for i,j in guards:
	            for dx,dy in [[-1,0],[1,0],[0,-1],[0,1]]:
	                x = i+dx
	                y = j+dy
	                while 0<=x<m and 0<=y<n and (grid[x][y]==0 or grid[x][y]==1):
	                    grid[x][y] = 1
	                    x += dx
	                    y += dy
	        return sum(grid[i][j]==0 for i in range(m) for j in range(n))

            	      
			
### 运行结果

	47 / 47 test cases passed.
	Status: Accepted
	Runtime: 2600 ms
	Memory Usage: 45.1 MB


### 原题链接



https://leetcode.com/contest/biweekly-contest-77/problems/count-unguarded-cells-in-the-grid/

您的支持是我最大的动力
