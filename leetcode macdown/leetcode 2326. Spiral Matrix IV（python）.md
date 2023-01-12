leetcode  2326. Spiral Matrix IV（python）




### 描述

You are given two integers m and n, which represent the dimensions of a matrix. You are also given the head of a linked list of integers. Generate an m x n matrix that contains the integers in the linked list presented in spiral order (clockwise), starting from the top-left of the matrix. If there are remaining empty spaces, fill them with -1.

Return the generated matrix.



Example 1:


![](https://assets.leetcode.com/uploads/2022/05/09/ex1new.jpg)

	Input: m = 3, n = 5, head = [3,0,2,6,8,1,7,9,4,2,5,5,0]
	Output: [[3,0,2,6,8],[5,0,-1,-1,1],[5,2,4,9,7]]
	Explanation: The diagram above shows how the values are printed in the matrix.
	Note that the remaining spaces in the matrix are filled with -1.
	
Example 2:

![](https://assets.leetcode.com/uploads/2022/05/11/ex2.jpg)

	Input: m = 1, n = 4, head = [0,1,2]
	Output: [[0,1,2,-1]]
	Explanation: The diagram above shows how the values are printed from left to right in the matrix.
	The last space in the matrix is set to -1.




Note:

	1 <= m, n <= 10^5
	1 <= m * n <= 10^5
	The number of nodes in the list is in the range [1, m * n].
	0 <= Node.val <= 1000



### 解析

根据题意，给定两个整数 m 和 n ，它们代表矩阵的维数。给定一个整数链表的 head 。从矩阵的左上角开始，生成一个 m x n 矩阵，其中每个矩阵的位置以螺旋顺序（顺时针）呈现的链表中的整数。 如果还有剩余的空格，则用 -1 填充它们。返回生成的矩阵。

这道题就是在考察对二维数组的操作，我们只需要根据题意，顺时针模拟填充矩阵的过程即可完成题意，遍历每个元素进行填充不难，难点在与判断什么时候改变方向。

时间复杂度为 O(M\*N) ，空间复杂度为 O(M\*N) ，M 和 N 分别为矩阵的长和宽。

### 解答
				
	
	class Solution(object):
	    def spiralMatrix(self, m, n, head):
	        """
	        :type m: int
	        :type n: int
	        :type head: Optional[ListNode]
	        :rtype: List[List[int]]
	        """
	        matrix = [[-1 for _ in range(n)] for _ in range(m)]
	        dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
	        row, column = 0, 0
	        d = 0
	        while head:
	            matrix[row][column] = head.val
	            head = head.next
	            nextR, nextC = row + dirs[d][0], column + dirs[d][1]
	            if not (0 <= nextR < m and 0 <= nextC < n and matrix[nextR][nextC]==-1):
	                d = (d + 1) % 4
	            row, column = row + dirs[d][0], column + dirs[d][1]
	        return matrix
			
### 运行结果


	
	49 / 49 test cases passed.
	Status: Accepted
	Runtime: 2697 ms
	Memory Usage: 104.4 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-300/problems/spiral-matrix-iv/


您的支持是我最大的动力
