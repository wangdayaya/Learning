leetcode 221. Maximal Square （python）

### 描述


Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

 


Example 1:

![](https://assets.leetcode.com/uploads/2020/11/26/max1grid.jpg)

	Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
	Output: 4

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/11/26/max2grid.jpg)

	Input: matrix = [["0","1"],["1","0"]]
	Output: 1


Example 3:

	
	Input: matrix = [["0"]]
	Output: 0
	




Note:

	m == matrix.length
	n == matrix[i].length
	1 <= m, n <= 300
	matrix[i][j] is '0' or '1'.


### 解析


根据题意，给出了一个 m x n 的矩阵 matrix ，里面每个格子填充了 0 或者 1 ，找到只包含 1 的最大的正方形面积并返回。其实这道题一看完题目，我们就能知道这种数数的题目基本上用动态规划会比较方便一点，我们定义 dp[i][j] 为以此位置做右下角的正方形的最大边长，公式为：

	dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
	
用图展示会更加清楚：

![](https://leetcode.com/media/original_images/221_Maximal_Square.PNG?raw=true)

最后只需要找到 dp 中的最大值作为正方形边长，得到正方形的最大面积即可。其实如果是经常刷题的朋友应该知道，这道题和 [1277. Count Square Submatrices with All Ones](https://leetcode.com/problems/count-square-submatrices-with-all-ones/) 几乎一样。

### 解答
				

	import numpy
	class Solution(object):
	    def maximalSquare(self, matrix):
	        """
	        :type matrix: List[List[str]]
	        :rtype: int
	        """
	        M = len(matrix)
	        N = len(matrix[0])
	        dp = [[0] * N for _ in range(M)]
	        for n in range(N):
	            dp[0][n] = int(matrix[0][n])
	        for m in range(M):
	            dp[m][0] = int(matrix[m][0])
	        for i in range(1, M):
	            for j in range(1, N):
	                if matrix[i][j] == '0':
	                    continue
	                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
	        L = numpy.max(dp)
	        return L * L
            	      
			
### 运行结果

	Runtime: 213 ms, faster than 39.94% of Python online submissions for Maximal Square.
	Memory Usage: 33.8 MB, less than 6.85% of Python online submissions for Maximal Square.

### 解析

其实也可以使用暴力求解，三重循环来寻找最大的面积，但是会超时。

### 解答

	import numpy as np
	class Solution(object):
	    def maximalSquare(self, matrix):
	        """
	        :type matrix: List[List[str]]
	        :rtype: int
	        """
	        result = 0
	        M = len(matrix)
	        N = len(matrix[0])
	        m = [[0]*N for _ in range(M)]
	        for i in range(M):
	            for j in range(N):
	                if matrix[i][j] == '1':
	                    m[i][j] = 1
	                else:
	                    m[i][j] = 0
	        m = np.array(m)
	        for l in range(1, min(M, N) + 1):
	            for i in range(M - l + 1):
	                for j in range(N - l + 1):
	                    if m[i][j]==0:
	                        continue
	                    if np.sum(m[i:i + l, j:j + l]) == l*l:
	                        result = max(result, l * l)
	        return result
	                    

### 运行结果

	Time Limit Exceeded	
	
原题链接：https://leetcode.com/problems/maximal-square/



您的支持是我最大的动力
