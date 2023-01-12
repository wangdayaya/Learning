leetcode 59. Spiral Matrix II （python）




### 描述

Given a positive integer n, generate an n x n matrix filled with elements from 1 to n2 in spiral order.





Example 1:


![](https://assets.leetcode.com/uploads/2020/11/13/spiraln.jpg)

	Input: n = 3
	Output: [[1,2,3],[8,9,4],[7,6,5]]
	
Example 2:

	Input: n = 1
	Output: [[1]]





Note:

	
	1 <= n <= 20

### 解析

根据题意，给定正整数 n ，将 1 到 n^2 的元素以螺旋顺时针顺序填充到 n×n 矩阵中并返回。

这道题的题意简单明了，虽然是个 Medium 的难度的题，但是其实很简单，就是找规律而已。

首先我们结合案例 1 可以直观地理解怎么将元素填充到 matrix 中，按照规律就是在第一行先向右到底填充 matrix ，然后转向下到底填充 matrix ，然后再转向左到底填充 matrix ，再转向上到“底”填充 matrix ，这里不是真的到底，而是碰到的第一行已经填充了。之后的规律还是继续按照上面所说的四个方向继续填充还空着的单元格子。

这么一捋思路，基本上解决的框架就出来，我们定义四个变量 right 、down 、left 、up 然后开始循环填充每个位置的元素：

* 当我们向右填充的时候，只要关注第 up 行的 range(left, right+1) 范围内的位置进行填充即可，然后对 up 加一，因为最上一层空位置刚才已经填满了
* 当我们向下填充的时候，只要关注第 right 列的 range(up, down+1) 范围内的位置进行填充即可，然后对 right 减一，因为最右一层空位置刚才已经填满了
* 当我们向左填充的时候，只要关注第 down 行的 range(right, left-1, -1) 范围内的位置进行填充即可，然后对 down 减一，因为最下一层空位置刚才已经填满了
* 当我们向上填充的时候，只要关注第 left 列的 range(down, up-1, -1) 范围内的位置进行填充即可，然后对 left 加一，因为最左一层空位置刚才已经填满了

重复上面的过程直到将所有位置都填满，直接返回 matrix 即可。

时间复杂度为 O(n^2) ，空间复杂度为 O(n^2) 。

### 解答
				
	class Solution(object):
	    def generateMatrix(self, n):
	        """
	        :type n: int
	        :rtype: List[List[int]]
	        """
	        matrix = [[0]*n for _ in range(n)]
	        direct = 0
	        count = 0
	        up = 0 ; down = n-1; left = 0; right = n-1
	        while True:
	            if direct == 0: # 向右
	                for i in range(left, right+1):
	                    count += 1
	                    matrix[up][i] = count
	                up += 1
	            if direct == 1: # 向下
	                for i in range(up, down+1):
	                    count += 1
	                    matrix[i][right] = count
	                right -= 1
	            if direct == 2: # 向左
	                for i in range(right, left-1, -1):
	                    count += 1
	                    matrix[down][i] = count
	                down -= 1
	            if direct == 3: # 向上
	                for i in range(down, up-1, -1):
	                    count += 1
	                    matrix[i][left] = count
	                left += 1
	            if count == n * n: return matrix
	            direct = (direct+1)%4

            	      
			
### 运行结果



	Runtime: 15 ms, faster than 96.17% of Python online submissions for Spiral Matrix II.
	Memory Usage: 13.5 MB, less than 42.12% of Python online submissions for Spiral Matrix II.
### 原题链接


https://leetcode.com/problems/spiral-matrix-ii/


您的支持是我最大的动力
