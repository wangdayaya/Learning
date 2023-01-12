leetcode  1263. Minimum Moves to Move a Box to Their Target Location（python）
### 每日经典

《从军行七首·其四》 ——王昌龄（唐）

青海长云暗雪山，孤城遥望玉门关。

黄沙百战穿金甲，不破楼兰终不还。

### 描述



A storekeeper is a game in which the player pushes boxes around in a warehouse trying to get them to target locations.

The game is represented by an m x n grid of characters grid where each element is a wall, floor, or box.

Your task is to move the box 'B' to the target position 'T' under the following rules:

* The character 'S' represents the player. The player can move up, down, left, right in grid if it is a floor (empty cell).
* The character '.' represents the floor which means a free cell to walk.
* The character '#' represents the wall which means an obstacle (impossible to walk there).
* There is only one box 'B' and one target cell 'T' in the grid.
* The box can be moved to an adjacent free cell by standing next to the box and then moving in the direction of the box. This is a push.
* The player cannot walk through the box.

Return the minimum number of pushes to move the box to the target. If there is no way to reach the target, return -1.

Example 1:

![](https://assets.leetcode.com/uploads/2019/11/06/sample_1_1620.png)

	Input: grid = [["#","#","#","#","#","#"],
	               ["#","T","#","#","#","#"],
	               ["#",".",".","B",".","#"],
	               ["#",".","#","#",".","#"],
	               ["#",".",".",".","S","#"],
	               ["#","#","#","#","#","#"]]
	Output: 3
	Explanation: We return only the number of times the box is pushed.

	
Example 2:

	
	Input: grid = [["#","#","#","#","#","#"],
	               ["#","T","#","#","#","#"],
	               ["#",".",".","B",".","#"],
	               ["#","#","#","#",".","#"],
	               ["#",".",".",".","S","#"],
	               ["#","#","#","#","#","#"]]
	Output: -1

Example 3:

	Input: grid = [["#","#","#","#","#","#"],
	               ["#","T",".",".","#","#"],
	               ["#",".","#","B",".","#"],
	               ["#",".",".",".",".","#"],
	               ["#",".",".",".","S","#"],
	               ["#","#","#","#","#","#"]]
	Output: 5
	Explanation: push the box down, left, left, up and up.



Note:

	m == grid.length
	n == grid[i].length
	1 <= m, n <= 20
	grid contains only characters '.', '#', 'S', 'T', or 'B'.
	There is only one character 'S', 'B', and 'T' in the grid.


### 解析


根据题意，storekeeper 是一种游戏，玩家在仓库中四处推箱子，试图让它们到达目标位置。游戏由一个 m x n 的字符网格 grid 表示，其中每个元素是墙壁、地板或盒子。任务是根据以下规则将盒子 “B” 移动到目标位置 “T” ：

* 字符 “S” 代表玩家。 如果是地板（空单元格），玩家可以在网格中向上、向下、向左、向右移动。
* 字符 "."  代表地板，这意味着可以自由行走的单元格。
* 字符 “#” 代表墙，这意味着障碍物。
* 网格中只有一个盒子 “B” 和一个目标单元格 “T” 。
* 通过站在盒子旁边然后向盒子的方向移动，盒子可以移动到相邻的空闲单元格。 这是一个推动操作。
* 玩家不能穿过盒子。

返回将框移动到目标的最小推动次数。 如果无法到达目标，则返回-1。

题意很简单，只是其实使用 BFS 思想，对人和箱子的位置进行不断地移动即可，只是会有很多限制条件需要想清楚。我们使用一个四维的列表 memo 表示从起点移动到这个状态的时候推动的次数，[bx, by] 是箱子的位置， [px,py] 是人的位置，[tx,ty] 是目标位置。将队列 q 的最后一个元素弹出，找出当前状态的下人的可移动的位置放入队列 q 的开始，再判断是否当前人的位置和箱子位置紧挨，如果是的话则表示可以推动箱子，将推动箱子到可以到达的位置放入到队列 q 的结尾。重复以上过程如果箱子的位置到了目标位置则直接返回 memo[bx][by][px][py] ，否则说明不可能退到目标位置，直接返回 -1 。

### 解答
				

	import numpy as np
	class Solution(object):
	        
	    def minPushBox(self, grid):
	        """
	        :type grid: List[List[str]]
	        :rtype: int
	        """
	        M = len(grid)
	        N = len(grid[0])
	        bx,by,px,py,tx,ty = 0,0,0,0,0,0
	        dir = [[1,0],[-1,0],[0,1],[0,-1]]
	        for i in range(M):
	            for j in range(N):
	                if grid[i][j] == 'S':
	                    px = i
	                    py = j
	                    grid[i][j] = '.'
	                elif grid[i][j] == 'B':
	                    bx = i
	                    by = j
	                    grid[i][j] = '.'
	                elif grid[i][j] == 'T':
	                    tx = i
	                    ty = j
	                    grid[i][j] = '.'
	                    
	        q = [[bx,by,px,py]]
	        memo = np.ones((21,21,21,21),dtype=np.int) 
	        memo = memo * -1
	        memo[bx][by][px][py] = 0
	        
	        while q:
	            [bx,by,px,py] = q.pop(0)
	            
	            if bx == tx and by == ty:
	                return memo[bx][by][px][py]
	                
	            for k in range(4):
	                x = px+dir[k][0]
	                y = py+dir[k][1]
	                if x<0 or x>=M or y<0 or y>=N: continue
	                if grid[x][y]!='.': continue
	                if x==bx and y==by : continue
	                if memo[bx][by][x][y]>=0: continue
	                memo[bx][by][x][y] =  memo[bx][by][px][py]
	                q.insert(0,[bx,by,x,y])
	            
	            if abs(px-bx) + abs(py-by) == 1:
	                for k in range(4):
	                    if px+dir[k][0]==bx and py+dir[k][1]==by:
	                        bx2 = bx + dir[k][0]
	                        by2 = by + dir[k][1]
	                        if bx2<0 or bx2>=M or by2<0 or by2>=N: continue
	                        if grid[bx2][by2]!='.':continue
	                        if memo[bx2][by2][bx][by]>=0: continue
	                        memo[bx2][by2][bx][by] = memo[bx][by][px][py] + 1
	                        q.append([bx2,by2,bx,by])
	        return -1
            	      
			
### 运行结果

	Runtime: 1165 ms, faster than 7.23% of Python online submissions for Minimum Moves to Move a Box to Their Target Location.
	Memory Usage: 28.3 MB, less than 6.02% of Python online submissions for Minimum Moves to Move a Box to Their Target Location.
		


原题链接：https://leetcode.com/problems/minimum-moves-to-move-a-box-to-their-target-location/



您的支持是我最大的动力
