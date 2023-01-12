### 每日经典

《乌衣巷》 ——刘禹锡（唐）

朱雀桥边野草花，乌衣巷口夕阳斜。

旧时王谢堂前燕，飞入寻常百姓家。

### 描述


You are given a rows x cols matrix grid representing a field of cherries where grid[i][j] represents the number of cherries that you can collect from the (i, j) cell.

You have two robots that can collect cherries for you:

* Robot #1 is located at the top-left corner (0, 0), and
* Robot #2 is located at the top-right corner (0, cols - 1).
Return the maximum number of cherries collection using both robots by following the rules below:

* From a cell (i, j), robots can move to cell (i + 1, j - 1), (i + 1, j), or (i + 1, j + 1).
* When any robot passes through a cell, It picks up all cherries, and the cell becomes an empty cell.
* When both robots stay in the same cell, only one takes the cherries.
* Both robots cannot move outside of the grid at any moment.
* Both robots should reach the bottom row in grid.


Example 1:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/574562b459604ec081a76a800c3e2443~tplv-k3u1fbpfcp-zoom-1.image)
	
	Input: grid = [[3,1,1],[2,5,1],[1,5,5],[2,1,1]]
	Output: 24
	Explanation: Path of robot #1 and #2 are described in color green and blue respectively.
	Cherries taken by Robot #1, (3 + 2 + 5 + 2) = 12.
	Cherries taken by Robot #2, (1 + 5 + 5 + 1) = 12.
	Total of cherries: 12 + 12 = 24.
	
Example 2:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f4e0c52d788e4f2cb0000a4588f8d6aa~tplv-k3u1fbpfcp-zoom-1.image)

	Input: grid = [[1,0,0,0,0,0,1],[2,0,0,0,0,3,0],[2,0,9,0,0,0,0],[0,3,0,5,4,0,0],[1,0,2,3,0,0,6]]
	Output: 28
	Explanation: Path of robot #1 and #2 are described in color green and blue respectively.
	Cherries taken by Robot #1, (1 + 9 + 5 + 2) = 17.
	Cherries taken by Robot #2, (1 + 3 + 4 + 3) = 11.
	Total of cherries: 17 + 11 = 28.




Note:

	rows == grid.length
	cols == grid[i].length
	2 <= rows, cols <= 70
	0 <= grid[i][j] <= 100

### 解析

根据题意，给定一个 rows x cols 矩阵网格 grid，表示一个樱桃农田，其中 grid[i][j] 表示可以从 (i, j) 单元格收集的樱桃数量。手里有两个收集樱桃的机器人：

* 机器人 #1 位于左上角 (0, 0)
* 机器人 #2 位于右上角 (0, cols - 1) 

按照以下规则返回使用两个机器人收集的最大樱桃数量：

* 机器人可以从单元格 (i, j) 移动到单元格 (i + 1, j - 1)、(i + 1, j) 或 (i + 1, j + 1)
* 当任何机器人经过一个单元格时，它会捡起所有的樱桃，这个单元格就变成了一个空单元格
* 当两个机器人都呆在同一个格子里时，只有一个机器人拿走樱桃
* 两个机器人在任何时候都不能移出网格
* 两个机器人都应该到达网格的最后一行

题目很简单，看完之后第一个反应就是用动态规划，因为很明显下一行得到的樱桃取决于上一行，我们定义 dp[i][j] 表示在当前行的时候，两个机器人分别在位置 i 和 j 时我们可以得到的最大数量的樱桃，我们只需要考虑上一行的可能的 9 个位置到当前行的关系即可。第一行能得到的最大樱桃数量已知，也就是 dp[0][N-1] = grid[0][0] + grid[0][N-1] ，从第二行开始动态计算即可，动态规划公式：

	dp[i][j] = max(dp[i][j], dp_tmp[a][b]+grid[m][i]+(grid[m][j] if i!=j else 0))

### 解答
				
	class Solution(object):
	    def cherryPickup(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        M = len(grid)
	        N = len(grid[0])
	        dp = [[-float('inf')]*N for _ in range(N)]
	        dp[0][N-1] = grid[0][0] + grid[0][N-1]
	        for m in range(1, M):
	            dp_tmp = copy.deepcopy(dp)
	            for i in range(N):
	                for j in range(i, N):
	                    dp[i][j] = -float('inf')
	                    for a in range(i-1, i+2):
	                        for b in range(j-1, j+2):
	                            if a<0 or a>=N or b<0 or b>=N: continue
	                            dp[i][j] = max(dp[i][j], dp_tmp[a][b]+grid[m][i]+(grid[m][j] if i!=j else 0))
	        result = 0
	        for i in range(N):
	            for j in range(N):
	                result = max(result, dp[i][j])
	                
	        return result
	        
	        
            	      
			
### 运行结果

	Runtime: 1959 ms, faster than 40.91% of Python online submissions for Cherry Pickup II.
	Memory Usage: 14.1 MB, less than 86.36% of Python online submissions for Cherry Pickup II.
	
### 解析

再来一种代码比较简洁的解法，只是这种得使用缓存注解 @lru_cache(None) 来加速，否则可能会超时。我们定义函数 dp(x, y1, y2) 表示第一个机器人在 (x,y1) ，第二个机器人在 (x,y2) 时候可以拿到的最多樱桃数量。如果当 y1 == y2 的时候，只需要加一次即可，表示只允许一个机器人收集当前位置的樱桃。如果机器人位置越界直接返回无穷小即可。下一行我们可采集的位置只可能有 9 种可能，所以遍历一次，找出最大的数量加入结果即可。这个代码思路是网格从上往下顺序来设计动态规划的解法，但是计算过程是一个从下往上的回溯过程。

### 解答
	
	class Solution(object):
	    def cherryPickup(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        M = len(grid)
	        N = len(grid[0])
	        
	        @lru_cache(None)
	        def dp(x, y1, y2):
	            if y1 < 0 or y1 >= N or y2 < 0 or y2 >= N:
	                return -float('inf')
	            result = 0
	            result += grid[x][y1]
	            
	            if y1 != y2:
	                result += grid[x][y2]
	                
	            if x != M-1:
	                result += max(dp(x+1, new_y1, new_y2)
	                              for new_y1 in [y1, y1+1, y1-1]
	                              for new_y2 in [y2, y2+1, y2-1])
	            return result
	
	        return dp(0, 0, N-1)
	        
	        
	        
### 运行结果

	Runtime: 780 ms, faster than 94.05% of Python3 online submissions for Cherry Pickup II.
	Memory Usage: 31.3 MB, less than 39.82% of Python3 online submissions for Cherry Pickup II.

### 解析

还可以按照从下往上的顺序解题，思路和上面的解法类似。

### 解答

	class Solution(object):
	    def cherryPickup(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        M = len(grid)
	        N = len(grid[0])
	        dp = [[[-float('inf')]*N for _ in range(N)] for _ in range(M)]
	        
	        for x in range(M-1, -1, -1):
	            for y1 in range(N):
	                for y2 in range(N):
	                    result = 0
	                    result += grid[x][y1] + (grid[x][y2] if y1!=y2 else 0)
	
	                    if x != M-1:
	                        result += max(dp[x+1][new_y1][new_y2] for new_y1 in [y1, y1+1, y1-1] for new_y2 in [y2, y2+1, y2-1] if 0<=new_y1<N and  0<=new_y2<N )
	                    dp[x][y1][y2] = result
	
	        return dp[0][0][N-1]
	        

### 运行结果

	Runtime: 1936 ms, faster than 34.10% of Python3 online submissions for Cherry Pickup II.
	Memory Usage: 22.3 MB, less than 68.42% of Python3 online submissions for Cherry Pickup II.
原题链接：https://leetcode.com/problems/cherry-pickup-ii/


您的支持是我最大的动力