leetcode  63. Unique Paths II（python）

### 描述

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

Now consider if some obstacles are added to the grids. How many unique paths would there be?

An obstacle and space is marked as 1 and 0 respectively in the grid.





Example 1:


![](https://assets.leetcode.com/uploads/2020/11/04/robot1.jpg)

	Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
	Output: 2
	Explanation: There is one obstacle in the middle of the 3x3 grid above.
	There are two ways to reach the bottom-right corner:
	1. Right -> Right -> Down -> Down
	2. Down -> Down -> Right -> Right

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/11/04/robot2.jpg)

	Input: obstacleGrid = [[0,1],[0,0]]
	Output: 1




Note:

	
	m == obstacleGrid.length
	n == obstacleGrid[i].length
	1 <= m, n <= 100
	obstacleGrid[i][j] is 0 or 1.

### 解析

根据题意，就是给出了一个 m x n 的地盘，你在左上角只能向右、向下两种方式走路，最后走到右下角，碰到障碍物则无法前进，求有多少种不同的路径。

有经验的同学都知道这种一般都是动态规划来求解，但是 m 和 n 小的时候用回溯的思想也能找到可能路径，通过写递归函数来进行求解，本题的 m 和 n 比较大，从运行结果来看也是超时，但是不妨碍我们试着用这种方式解体。





### 解答
				

	class Solution(object):
	    def uniquePathsWithObstacles(self, obstacleGrid):
	        """
	        :type obstacleGrid: List[List[int]]
	        :rtype: int
	        """
	        M = len(obstacleGrid)
	        N = len(obstacleGrid[0])
	
	        def dfs(x, y):
	            if x >= M or y >= N or x < 0 or y < 0:
	                return 0
	            if obstacleGrid[x][y] == 1:
	                return 0
	            if x == M - 1 and y == N - 1:
	                return 1
	            count = dfs(x + 1, y) + dfs(x, y + 1)
	            return count
	
	        return dfs(0, 0)
            	      
			
### 运行结果

	Time Limit Exceeded

### 解析

直接用动态规划解体，初始化一个 MxN 大小的二维列表 dp ，每个元素表示的是从开始位置到当前位置一共有多少种不同的路径。思路比较简单：

* 遍历 obstacleGrid[0] ，如果索引 i 的元素为 0 ，表示此路通着，则 dp[0][i] 设置为 1 ，如果元素为 1 ，则表示此路不通，那此路向右的路线也一直不通，直接跳出当前遍历
* 遍历 obstacleGrid 第一列的元素，如果索引为 j 的元素为 0 ，表示此路通着，则 dp[j][0] 设置为 1 ，如果元素为 1 ，则表示此路不通，那此路向下的路线也一直不通，直接跳出当前遍历
* 从 [1,1] 位置开始遍历 obstacleGrid ，如果当前元素为 0 ，说明此路可走，那么从开始位置到当前位置的所有路线为 dp[i][j-1] + dp[i-1][j] ，如果当前元素为 1 ，说明此路不通，继续进行下一个元素的遍历
* 遍历结束，得到的 dp[-1][-1] 即为从开始位置到最后结束位置所有不同路径数

### 解答

	class Solution(object):
	    def uniquePathsWithObstacles(self, obstacleGrid):
	        """
	        :type obstacleGrid: List[List[int]]
	        :rtype: int
	        """
	        M = len(obstacleGrid)
	        N = len(obstacleGrid[0])
	        dp = [[0]*N for _ in range(M)]
	        for i,x in enumerate(obstacleGrid[0]):
	            if x==0:
	                dp[0][i] = 1
	            else:
	                break
	        for j in range(M):
	            if obstacleGrid[j][0] == 0:
	                dp[j][0] = 1
	            else:
	                break
	        for i in range(1,M):
	            for j in range(1,N):
	                if obstacleGrid[i][j] == 0:
	                    dp[i][j] = dp[i][j-1] + dp[i-1][j]
	        return dp[-1][-1]

### 运行结果

	Runtime: 32 ms, faster than 49.75% of Python online submissions for Unique Paths II.
	Memory Usage: 13.5 MB, less than 61.88% of Python online submissions for Unique Paths II.
### 相似题

* [62. Unique Paths](https://leetcode.com/problems/unique-paths/)
* [980. Unique Paths III](https://leetcode.com/problems/unique-paths-iii/)

原题链接：https://leetcode.com/problems/unique-paths-ii/



您的支持是我最大的动力
