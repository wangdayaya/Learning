leetcode  1926. Nearest Exit from Entrance in Maze（python）




### 描述

You are given an m x n matrix maze (0-indexed) with empty cells (represented as '.') and walls (represented as '+'). You are also given the entrance of the maze, where entrance = [entrance<sub>row</sub>, entrance<sub>col</sub>] denotes the row and column of the cell you are initially standing at.

In one step, you can move one cell up, down, left, or right. You cannot step into a cell with a wall, and you cannot step outside the maze. Your goal is to find the nearest exit from the entrance. An exit is defined as an empty cell that is at the border of the maze. The entrance does not count as an exit.

Return the number of steps in the shortest path from the entrance to the nearest exit, or -1 if no such path exists.



Example 1:


![](https://assets.leetcode.com/uploads/2021/06/04/nearest1-grid.jpg)

	Input: maze = [["+","+",".","+"],[".",".",".","+"],["+","+","+","."]], entrance = [1,2]
	Output: 1
	Explanation: There are 3 exits in this maze at [1,0], [0,2], and [2,3].
	Initially, you are at the entrance cell [1,2].
	- You can reach [1,0] by moving 2 steps left.
	- You can reach [0,2] by moving 1 step up.
	It is impossible to reach [2,3] from the entrance.
	Thus, the nearest exit is [0,2], which is 1 step away.
	
Example 2:


![](https://assets.leetcode.com/uploads/2021/06/04/nearesr2-grid.jpg)

	Input: maze = [["+","+","+"],[".",".","."],["+","+","+"]], entrance = [1,0]
	Output: 2
	Explanation: There is 1 exit in this maze at [1,2].
	[1,0] does not count as an exit since it is the entrance cell.
	Initially, you are at the entrance cell [1,0].
	- You can reach [1,2] by moving 2 steps right.
	Thus, the nearest exit is [1,2], which is 2 steps away.

Example 3:

![](https://assets.leetcode.com/uploads/2021/06/04/nearest3-grid.jpg)

	Input: maze = [[".","+"]], entrance = [0,0]
	Output: -1
	Explanation: There are no exits in this maze.



Note:


* maze.length == m
* maze[i].length == n
* 1 <= m, n <= 100
* maze[i][j] is either '.' or '+'.
* entrance.length == 2
* 0 <= entrance<sub>row</sub> < m
* 0 <= entrance<sub>col</sub> < n
* entrance will always be an empty cell.

### 解析

给定一个 m x n 矩阵迷宫，其中包含空单元格（表示为“.”）和墙壁（表示为“+”）。还给了迷宫的入口 [entrance<sub>row</sub>, entrance<sub>col</sub>]  表示最初站立的单元格的位置。

在每步操作中，我们可以向上、向下、向左或向右移动一个单元格。但是不能踏入有墙的格子，也不能走出迷宫。我们的目标是找到离入口最近的出口。出口被定义为位于迷宫边界的空单元格。需要注意的是入口不算出口。

返回从入口到最近出口的最短路径中的步数，如果不存在此类路径，则返回 -1 。

其实这道题考察的就是 BFS 算法，我们只要以入口为起始点，不断将上、下、左、右四个方向的格子一层一层进行地毯式判断和搜索，就能找到最终的答案。具体如下：

* 初始化一个 step 保存最短路径，初始化一个使用队列用来保存 BFS 经过的位置，以起始点为开始点先将其存入队列，然后将该位置设置为 + 表示已经访问过
* 然后不断循环弹出每一层的每一个可用的位置 [r,c] ，判断其上、下、左、右四个位置是否是空格子的边界位置，如果是将 step 返回即可，如果不是则将其设置为 + 加入到队列中，继续重复上面的操作

时间复杂度为 O(N)，空间复杂度为 O(N)，N 表示所有格子的数量。



### 解答

	class Solution:
	    def nearestExit(self, M: List[List[str]], E: List[int]) -> int:
	        row = len(M)
	        col = len(M[0])
	        queue = collections.deque()
	        queue.append(E)
	        M[E[0]][E[1]] = '+'
	        step = 0
	        while queue:
	            step += 1
	            for _ in range(len(queue)):
	                r, c = queue.popleft()
	                for rr, cc in [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]]:
	                    if 0 <= rr < row and 0 <= cc < col:
	                        if M[rr][cc] == '.':
	                            if rr in (0, row - 1) or cc in (0, col - 1):
	                                return step
	                            M[rr][cc] = '+'
	                            queue.append([rr, cc])
	        return -1

### 运行结果

	Runtime Beats 46.8%
	Memory Beats 68.6%

### 原题链接

https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/description/



您的支持是我最大的动力
