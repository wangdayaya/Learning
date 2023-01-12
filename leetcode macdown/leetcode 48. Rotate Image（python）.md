leetcode 48. Rotate Image （python）




### 描述


You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.


Example 1:

![](https://assets.leetcode.com/uploads/2020/08/28/mat1.jpg)

	Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
	Output: [[7,4,1],[8,5,2],[9,6,3]]

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/08/28/mat2.jpg)

	Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
	Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]



Note:

	n == matrix.length == matrix[i].length
	1 <= n <= 20
	-1000 <= matrix[i][j] <= 1000


### 解析

根据题意，给定一个表示图像的 n x n 的 2D 矩阵 matrix ，将图像顺时针旋转 90 度。题目要求我们必须就地旋转图像，这意味着必须直接修改 2D 矩阵。 不能分配另一个二维矩阵进行旋转。

其实考察将矩阵进行顺时针旋转 90 度，是一个公式题，因为这个操作相当于将矩阵先进行转置运算，然后在进行列交换（第一列和最后一列交换，第二列和倒数第二列交换，以此类推）。举例原始矩阵如下：

	[1,2,3]
	[4,5,6]
	[7,8,9]
	
先转置获得：

	[1,4,7]
	[2,5,8]
	[3,6,9]
然后列交换获得结果：

	[7,4,1]
	[8,5,2]
	[9,6,3]

时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def rotate(self, matrix):
	        """
	        :type matrix: List[List[int]]
	        :rtype: None Do not return anything, modify matrix in-place instead.
	        """
	        N = len(matrix)
	        for i in range(N):
	            for j in range(i+1):
	                matrix[i][j], matrix[j][i] =  matrix[j][i], matrix[i][j]
	        for i in range(N):
	            for j in range(N//2):
	                matrix[i][j], matrix[i][N-j-1] = matrix[i][N-j-1],matrix[i][j]

### 运行结果

	Runtime: 43 ms, faster than 18.69% of Python online submissions for Rotate Image.
	Memory Usage: 13.4 MB, less than 68.46% of Python online submissions for Rotate Image.

### 原题链接

https://leetcode.com/problems/rotate-image/


您的支持是我最大的动力
