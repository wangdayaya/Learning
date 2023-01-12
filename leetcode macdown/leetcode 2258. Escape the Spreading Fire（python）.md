leetcode  2258. Escape the Spreading Fire（python）

这是第 77 场双周赛的第三题，难度 Medium ，考察的是对实际题目的理解，然后使用标记法来解决即可


### 描述


You are given a 0-indexed 2D integer array grid of size m x n which represents a field. Each cell has one of three values:

* 0 represents grass,
* 1 represents fire,
* 2 represents a wall that you and fire cannot pass through.

You are situated in the top-left cell, (0, 0), and you want to travel to the safehouse at the bottom-right cell, (m - 1, n - 1). Every minute, you may move to an adjacent grass cell. After your move, every fire cell will spread to all adjacent cells that are not walls.

Return the maximum number of minutes that you can stay in your initial position before moving while still safely reaching the safehouse. If this is impossible, return -1. If you can always reach the safehouse regardless of the minutes stayed, return 109.

Note that even if the fire spreads to the safehouse immediately after you have reached it, it will be counted as safely reaching the safehouse.

A cell is adjacent to another cell if the former is directly north, east, south, or west of the latter (i.e., their sides are touching).




Example 1:

![](https://assets.leetcode.com/uploads/2022/03/10/ex1new.jpg)

	Input: grid = [[0,2,0,0,0,0,0],[0,0,0,2,2,1,0],[0,2,0,0,1,2,0],[0,0,2,2,2,0,2],[0,0,0,0,0,0,0]]
	Output: 3
	Explanation: The figure above shows the scenario where you stay in the initial position for 3 minutes.
	You will still be able to safely reach the safehouse.
	Staying for more than 3 minutes will not allow you to safely reach the safehouse.

	
Example 2:


![](https://assets.leetcode.com/uploads/2022/03/10/ex2new2.jpg)

	Input: grid = [[0,0,0,0],[0,1,2,0],[0,2,0,0]]
	Output: -1
	Explanation: The figure above shows the scenario where you immediately move towards the safehouse.
	Fire will spread to any cell you move towards and it is impossible to safely reach the safehouse.
	Thus, -1 is returned.

Example 3:

![](https://assets.leetcode.com/uploads/2022/03/10/ex3new.jpg)

	Input: grid = [[0,0,0],[2,2,0],[1,2,0]]
	Output: 1000000000
	Explanation: The figure above shows the initial grid.
	Notice that the fire is contained by walls and you will always be able to safely reach the safehouse.
	Thus, 109 is returned.

	



Note:

	m == grid.length
	n == grid[i].length
	2 <= m, n <= 300
	4 <= m * n <= 2 * 10^4
	grid[i][j] is either 0, 1, or 2.
	grid[0][0] == grid[m - 1][n - 1] == 0


### 解析

根据题意，给定一个大小为 m x n 的 0 索引 2D 整数数组网格，每个单元格具有以下三个值之一：

* 0 代表草
* 1 代表火
* 2 代表不能穿过的墙

人一开始左上角的单元格 (0, 0) ，并且您想前往位于右下角的单元格 (m - 1, n - 1) 的安全屋。每分钟都可以移动到相邻的格子。人移动后的同时，每个火苗都会蔓延到所有相邻的非墙壁单元。

返回移动前您可以在初始位置停留的最大分钟数，同时仍能再移动后安全到达安全屋。如果这是不可能的，则返回 -1 。如果无论停留多长时间，您总能到达安全屋，请返回 10^9。请注意，即使在您到达后立即火灾蔓延到安全屋，也将被视为安全到达安全屋。

这道题看起来比较吓人，其实还比较简单，因为要最少在初始位置停留为 0 分钟，最多在初始位置停留 m\*n 分钟也就是 2 \* 10^4 ，我们可以使用二分法来找出最少可以停留的时间，这是总体思路，然后我们可以使用 BFS 来计算出每个格子发生火灾所需要的最短时间，这样我们就能判断人在某分钟走到某格子会不会被烧焦，这样不断用二分法来找最少可以停留的时间即可。

时间复杂度为 O(MN) ，空间复杂度为 O(MN) 。




### 解答
				

	class Solution(object):
	    def maximumMinutes(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        M, N = len(grid), len(grid[0])
	        timeArrive = [[float("inf") for _ in range(N)] for _ in range(M)]
	        fire = []
	        for i in range(M):
	            for j in range(N):
	                if grid[i][j] == 1:
	                    fire.append([i, j])
	                    timeArrive[i][j] = 0
	        stack = []
	        for i, j in fire:
	            stack.append((i, j, 0))
	        while stack:
	            x, y, time = stack.pop(0)
	            for i, j in [[x + 1, y], [x, y + 1], [x, y - 1], [x - 1, y]]:
	                if 0 <= i < M and 0 <= j < N and grid[i][j] == 0 and timeArrive[i][j] > time + 1:
	                    timeArrive[i][j] = time + 1
	                    stack.append((i, j, time + 1))
	
	        def waitK(k):
	            h = [(0, 0, k)]
	            vis = set()
	            vis.add((0, 0))
	            while h:
	                x, y, time = h.pop(0)
	                for i, j in [[x + 1, y], [x, y + 1], [x, y - 1], [x - 1, y]]:
	                    if i == M - 1 and j == N - 1 and timeArrive[i][j] >= time + 1:
	                        return True
	                    if 0 <= i < M and 0 <= j < N and grid[i][j] == 0 and timeArrive[i][j] > time + 1 and (i, j) not in vis:
	                        h.append((i, j, time + 1))
	                        vis.add((i, j))
	            return False
	
	        if waitK(2*10**4):
	            return 10**9
	        elif not waitK(0):
	            return -1
	        l,r = 0, 2*10**4
	        while l<r:
	            mid = (l+r)//2
	            if waitK(mid):
	                if not waitK(mid+1):
	                    return mid
	                else:
	                    l = mid + 1
	            else:
	                r = mid
	        return l
            	      
			
### 运行结果



	55 / 55 test cases passed.
	Status: Accepted
	Runtime: 1033 ms
	Memory Usage: 16.2 MB
### 原题链接



https://leetcode.com/contest/biweekly-contest-77/problems/escape-the-spreading-fire/


您的支持是我最大的动力
