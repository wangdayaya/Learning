leetcode  2373. Largest Local Values in a Matrix（python）




### 描述


You are given an n x n integer matrix grid. Generate an integer matrix maxLocal of size (n - 2) x (n - 2) such that:

* maxLocal[i][j] is equal to the largest value of the 3 x 3 matrix in grid centered around row i + 1 and column j + 1.

In other words, we want to find the largest value in every contiguous 3 x 3 matrix in grid. Return the generated matrix.


Example 1:

![](https://assets.leetcode.com/uploads/2022/06/21/ex1.png)

	Input: grid = [[9,9,8,1],[5,6,2,6],[8,2,6,4],[6,2,2,2]]
	Output: [[9,9],[8,6]]
	Explanation: The diagram above shows the original matrix and the generated matrix.
	Notice that each value in the generated matrix corresponds to the largest value of a contiguous 3 x 3 matrix in grid.

	
Example 2:

![](https://assets.leetcode.com/uploads/2022/07/02/ex2new2.png)

	Input: grid = [[1,1,1,1,1],[1,1,1,1,1],[1,1,2,1,1],[1,1,1,1,1],[1,1,1,1,1]]
	Output: [[2,2,2],[2,2,2],[2,2,2]]
	Explanation: Notice that the 2 is contained within every contiguous 3 x 3 matrix in grid.




Note:


	n == grid.length == grid[i].length
	3 <= n <= 100
	1 <= grid[i][j] <= 100

### 解析


根据题意，给定一个 n x n 整数矩阵 grid 。 生成一个大小为 (n - 2) x (n - 2) 的整数矩阵 maxLocal ，使得：

* maxLocal[i][j] 等于以 i + 1 行和 j + 1 列为中心的网格中 3 x 3 矩阵的最大值。

换句话说，我们想在网格中的每个连续 3 x 3 矩阵中找到最大值。 返回生成的矩阵。

这道题其实就是考察卷积神经网络中的池化，我们只需要对不同的九宫格取最大值，然后将这些最大值组合起来就是最后的结果矩阵。

时间复杂度为 O(N^2 \* 9) ，空间复杂度为 O(N^2) 。

### 解答

	class Solution(object):
	    def largestLocal(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: List[List[int]]
	        """
	        N = len(grid)
	        result = [[0] * (N-2) for _ in range(N-2)]
	        for i in range(1, N-1):
	            for j in range(1, N-1):
	                result[i-1][j-1] = max(grid[i-1][j-1],grid[i-1][j],grid[i-1][j+1],
	                                       grid[i][j-1],grid[i][j],grid[i][j+1],
	                                       grid[i+1][j-1],grid[i+1][j],grid[i+1][j+1])
	        return result

### 运行结果

	50 / 50 test cases passed.
	Status: Accepted
	Runtime: 100 ms
	Memory Usage: 14.1 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-306/problems/largest-local-values-in-a-matrix/


您的支持是我最大的动力
