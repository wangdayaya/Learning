leetcode 2319. Check if Matrix Is X-Matrix （python）




### 描述

A square matrix is said to be an X-Matrix if both of the following conditions hold:

* All the elements in the diagonals of the matrix are non-zero.
* All other elements are 0.

Given a 2D integer array grid of size n x n representing a square matrix, return true if grid is an X-Matrix. Otherwise, return false.





Example 1:


![](https://assets.leetcode.com/uploads/2022/05/03/ex1.jpg)

	Input: grid = [[2,0,0,1],[0,3,1,0],[0,5,2,0],[4,0,0,2]]
	Output: true
	Explanation: Refer to the diagram above. 
	An X-Matrix should have the green elements (diagonals) be non-zero and the red elements be 0.
	Thus, grid is an X-Matrix.
	
Example 2:


![](https://assets.leetcode.com/uploads/2022/05/03/ex2.jpg)

	Input: grid = [[5,7,0],[0,3,1],[0,5,0]]
	Output: false
	Explanation: Refer to the diagram above.
	An X-Matrix should have the green elements (diagonals) be non-zero and the red elements be 0.
	Thus, grid is not an X-Matrix.


Note:


	n == grid.length == grid[i].length
	3 <= n <= 100
	0 <= grid[i][j] <= 10^5

### 解析

根据题意，如果以下两个条件都成立，则称方阵为 X 矩阵：

* 矩阵对角线上的所有元素都非零。
* 所有其他元素为 0。

 给定一个大小为 n x n 的二维整数数组 grid 表示一个方阵，如果 grid 是一个 X 矩阵，则返回 true 。 否则返回 false 。
 
 这道题其实就是考察二维数组的遍历，我们只要根据 X 矩阵的两个条件，然后遍历每一个元素 grid[i][j]  ，如果 i==j 或者 i+j == N-1 的时候 grid[i][j] 等于 0 不满足第一个条件直接返回 False ，否则当 i 和 j 为其他情况的时候，grid[i][j] 不等于 0 不满足第二个条件返回 False ，遍历结束说明正常返回 True 。
 
 时间复杂度为 O(N^2) ，空间复杂度为 O(1) 。

### 解答


	class Solution(object):
	    def checkXMatrix(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: bool
	        """
	        N = len(grid)
	        for i in range(N):
	            for j in range(N):
	                if i==j or i+j == N-1:
	                    if grid[i][j] == 0:
	                        return False
	                elif grid[i][j] != 0:
	                    return False
	        return True
### 运行结果

	
	84 / 84 test cases passed.
	Status: Accepted
	Runtime: 414 ms
	Memory Usage: 14.4 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-299/problems/check-if-matrix-is-x-matrix/


您的支持是我最大的动力
