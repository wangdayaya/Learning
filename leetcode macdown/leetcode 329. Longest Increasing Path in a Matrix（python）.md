leetcode  329. Longest Increasing Path in a Matrix（python）




### 描述

Given an m x n integers matrix, return the length of the longest increasing path in matrix.

From each cell, you can either move in four directions: left, right, up, or down. You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).





Example 1:

![](https://assets.leetcode.com/uploads/2021/01/05/grid1.jpg)

	Input: matrix = [[9,9,4],[6,6,8],[2,1,1]]
	Output: 4
	Explanation: The longest increasing path is [1, 2, 6, 9].

	
Example 2:


![](https://assets.leetcode.com/uploads/2021/01/27/tmp-grid.jpg)

	Input: matrix = [[3,4,5],[3,2,6],[2,2,1]]
	Output: 4
	Explanation: The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.

Example 3:

	Input: matrix = [[1]]
	Output: 1




Note:

	m == matrix.length
	n == matrix[i].length
	1 <= m, n <= 200
	0 <= matrix[i][j] <= 2^31 - 1



### 解析


根据题意，给定一个 m x n 整数矩阵 matrix ，返回矩阵中最长递增路径的长度。

从每个单元格中，可以向四个方向移动：左、右、上或下，但是不得沿对角线移动或移动到边界之外。

开始我的思路是用动态规划来做，但是发现不好写条件，只好放弃了，可能是我的方法有问题，所以后来改用常规的 DFS 来解决这道题，因为要找最长的递增路径，所以我们只需要对每个格子去进行 DFS 来找以该格子开始的最长递增路径，找完所有的格子的最长递增路径之后就知道最后的答案了。但是这里有一点需要注意，在进行 DFS 的使用要进行记忆化存储，因为很多格子的遍历都是重复的，会导致超时。

M 是行数，N 是列数，执行每个 DFS 栈深度最大为 O(MN) ，因为最长递增路径可能为 MN ，宽度为 O(4MN) ，所以每个 DFS 的时间复杂度为 O(MN) ，此时因为已经遍历了所有的节点进行了记忆化，二重循环已经没有意义，所以总的时间复杂度为 O(MN) 。

空间复杂度主要用于递归和记忆化为 O(MN) 。


### 解答
				
	class Solution:
	    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
	        M = len(matrix)
	        N = len(matrix[0])
	        @lru_cache(None)
	        def dfs(i, j):
	            if not matrix:
	                return 0
	            r = 1
	            for dx, dy in [[-1, 0], [0, -1], [0, 1], [1, 0]]:
	                if -1 < i + dx < M and -1 < j + dy < N and matrix[i + dx][j + dy] > matrix[i][j]:
	                    r = max(r, dfs(i + dx, j + dy) + 1)
	            return r
	        result = 0
	        for i in range(M):
	            for j in range(N):
	                t = dfs(i, j)
	                result = max(result, t)
	        return result

            	      
			
### 运行结果

	Runtime: 594 ms, faster than 54.68% of Python3 online submissions for Longest Increasing Path in a Matrix.
	Memory Usage: 22.2 MB, less than 6.31% of Python3 online submissions for Longest Increasing Path in a Matrix.


### 原题链接



https://leetcode.com/problems/longest-increasing-path-in-a-matrix/


您的支持是我最大的动力
