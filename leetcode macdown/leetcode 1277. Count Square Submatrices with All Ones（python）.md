leetcode  1277. Count Square Submatrices with All Ones（python）

### 描述



Given a m * n matrix of ones and zeros, return how many square submatrices have all ones.



Example 1:

	Input: matrix =
	[
	  [0,1,1,1],
	  [1,1,1,1],
	  [0,1,1,1]
	]
	Output: 15
	Explanation: 
	There are 10 squares of side 1.
	There are 4 squares of side 2.
	There is  1 square of side 3.
	Total number of squares = 10 + 4 + 1 = 15.	
	
Example 2:

	Input: matrix = 
	[
	  [1,0,1],
	  [1,1,0],
	  [1,1,0]
	]
	Output: 7
	Explanation: 
	There are 6 squares of side 1.  
	There is 1 square of side 2. 
	Total number of squares = 6 + 1 = 7.




Note:

	1 <= arr.length <= 300
	1 <= arr[0].length <= 300
	0 <= arr[i][j] <= 1

### 解析

根据题意，只需要找出 matrix 中由 1 组成的任意大小的正方形个数即可。其实只要找规律就可以发现，设计一个 m*n 的 new 矩阵来存放数量值，其中的 new[i][j] 表示以此单元格为右下角的正方形的个数可以有几个，然后遍历比较所有的 matrix[1:,1:] 的值，如果 matrix[i][j] 值为 1 ，则说明至少有一个边长为 1 的正方形，然后比较 grid[i-1][j-1]、grid[i][j-1]、grid[i-1][j]  三个中的最小值 min ，表示可以至少在以 [i,j] 为右下角的地方可以形成 min 个正方形，将 min+1 赋值给 gridp[i][j] 即可，如果完成将值填充到 grid 对应的位置上面，最后将所有的 new 元素求和即可，

### 解答
				
	class Solution(object):
	    def countSquares(self, matrix):
	        """
	        :type matrix: List[List[int]]
	        :rtype: int
	        """
	        import numpy as np
	        M = len(matrix)
	        N = len(matrix[0])
	        grid = [[0] * N for _ in range(M)]
	        for i in range(N):
	            grid[0][i] = matrix[0][i]
	        for j in range(M):
	            grid[j][0] = matrix[j][0]
	        for i in range(1, M):
	            for j in range(1, N):
	                if matrix[i][j] == 1:
	                    grid[i][j] = min(grid[i-1][j-1] , grid[i-1][j] , grid[i][j-1]) + 1
	        return np.sum(grid)
	                
            	      
			
### 运行结果

	Runtime: 520 ms, faster than 83.00% of Python online submissions for Count Square Submatrices with All Ones.
	Memory Usage: 15.2 MB, less than 81.30% of Python online submissions for Count Square Submatrices with All Ones.

### 解析

还可以进行三重循环遍历来找可能合法的正方形，但是这种方法肯定超时。
### 解答

	class Solution(object):
	    def countSquares(self, matrix):
	        """
	        :type matrix: List[List[int]]
	        :rtype: int
	        """
	        import numpy as np
	        M = len(matrix)
	        N = len(matrix[0])
	        L = min(M,N)
	        matrix = np.array(matrix)
	        result = 0
	        for l in range(1, L+1):
	            for i in range(M+1-l):
	                for j in range(N+1-l):
	                    if np.sum(matrix[i:i+l,j:j+l]) == l**2:
	                        result += 1
	        return result

### 运行结果

	Time Limit Exceeded

原题链接：https://leetcode.com/problems/count-square-submatrices-with-all-ones/


您的支持是我最大的动力
