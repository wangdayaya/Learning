leetcode  240. Search a 2D Matrix II（python）




### 描述

Write an efficient algorithm that searches for a value target in an m x n integer matrix matrix. This matrix has the following properties:

* Integers in each row are sorted in ascending from left to right.
* Integers in each column are sorted in ascending from top to bottom.




Example 1:


![](https://assets.leetcode.com/uploads/2020/11/24/searchgrid2.jpg)

	Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
	Output: true
	
Example 2:


![](https://assets.leetcode.com/uploads/2020/11/24/searchgrid.jpg)

	Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
	Output: false

Example 3:





Note:

	m == matrix.length
	n == matrix[i].length
	1 <= n, m <= 300
	-10^9 <= matrix[i][j] <= 10^9
	All the integers in each row are sorted in ascending order.
	All the integers in each column are sorted in ascending order.
	-10^9 <= target <= 10^9


### 解析

根据题意，编写一个在 m x n 整数矩阵 matrix 中搜索值目标 target 的高效算法。 该矩阵具有以下性质：

* 每行中的整数从左到右升序排序。
* 每列中的整数从上到下按升序排序。

其实最暴力的方法就是从整个矩阵中一个一个挨着找，尽管可以 AC 但是这是相当慢的算法，时间复杂度为 O(M\*N)，题目中已经说明数据从左到右从上到下都已经是升序的情况，我们就可以想到用二分法进行搜索。

所以在稍微优化之下，我们可以在每一行中进行二分法查找，如果没有找到就去下一行进行二分法查找，这虽然能够在时间复杂度上优化到 O(M\*logN) ，但是还是有可以优化的空间。

其实我们可以使用更加巧妙的二分法进行查找，我们将 [0,N-1] 的位置设置为初始查找的位置，值为 x ，当 target 小于 x 时候，我们就将纵坐标减一，当 target 大于 x 我们就讲横坐标加一，如果 target 等于 x 直接返回 True 。在每次变化中我们会将搜索范围不断缩小为以 matrix 左下角为左下角，以 x 为右上角的矩阵。时间复杂度为 O(M+N) ，空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def searchMatrix(self, matrix, target):
	        """
	        :type matrix: List[List[int]]
	        :type target: int
	        :rtype: bool
	        """
	        M = len(matrix)
	        N = len(matrix[0])
	        i, j = 0, N - 1
	        while i < M and j >= 0:
	            if matrix[i][j] == target:
	                return True
	            if matrix[i][j] < target:
	                i += 1
	            elif matrix[i][j] > target:
	                j -= 1
	        return False

### 运行结果

	Runtime: 211 ms, faster than 45.36% of Python online submissions for Search a 2D Matrix II.
	Memory Usage: 19.5 MB, less than 70.32% of Python online submissions for Search a 2D Matrix II.

### 原题链接

https://leetcode.com/problems/search-a-2d-matrix-ii/



您的支持是我最大的动力
