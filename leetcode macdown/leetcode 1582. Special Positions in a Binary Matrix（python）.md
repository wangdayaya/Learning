leetcode  1582. Special Positions in a Binary Matrix（python）

### 描述


Given a rows x cols matrix mat, where mat[i][j] is either 0 or 1, return the number of special positions in mat.

A position (i,j) is called special if mat[i][j] == 1 and all other elements in row i and column j are 0 (rows and columns are 0-indexed).

 




Example 1:


	Input: mat = [[1,0,0],
	              [0,0,1],
	              [1,0,0]]
	Output: 1
	Explanation: (1,2) is a special position because mat[1][2] == 1 and all other elements in row 1 and column 2 are 0.
	
Example 2:

	Input: mat = [[1,0,0],
	              [0,1,0],
	              [0,0,1]]
	Output: 3
	Explanation: (0,0), (1,1) and (2,2) are special positions. 


Example 3:


	Input: mat = [[0,0,0,1],
	              [1,0,0,0],
	              [0,1,1,0],
	              [0,0,0,0]]
	Output: 2
	
Example 4:

	
	Input: mat = [[0,0,0,0,0],
	              [1,0,0,0,0],
	              [0,1,0,0,0],
	              [0,0,1,0,0],
	              [0,0,0,1,1]]
	Output: 3
	


Note:

	rows == mat.length
	cols == mat[i].length
	1 <= rows, cols <= 100
	mat[i][j] is 0 or 1.


### 解析

根据题意，就是要找出 mat 中的特殊数字有几个。特殊数字就是必须为 1 且所在的行和列其他位置都为 0 。直接使用内置函数，计算出来水平和垂直两个方向的和 h 和 v ，然后遍历 mat 中的元素，只要该元素为 1 且所在行和列的元素和为 1 ，那就将结果 result 加一，遍历结束即可得到结果。当然按照我的习惯，这种使用内置函数的做法我不推荐。


### 解答
				
	class Solution(object):
	    def numSpecial(self, mat):
	        """
	        :type mat: List[List[int]]
	        :rtype: int
	        """
	        import numpy as np
	        mat = np.array(mat)
	        v = np.sum(mat,0)
	        h = np.sum(mat,1)
	        result = 0
	        for i in range(len(mat)):
	            if h[i]!=1:
	                continue
	            for j in range(len(mat[0])):
	                if mat[i][j]==1 and v[j]==1 and h[i]==1:
	                    result += 1
	                    break
	        return result
	


            	      
			
### 运行结果



	Runtime: 184 ms, faster than 20.65% of Python online submissions for Special Positions in a Binary Matrix.
	Memory Usage: 26 MB, less than 5.43% of Python online submissions for Special Positions in a Binary Matrix.



### 解析

直接遍历矩阵 mat 中的每一行，然后计算每一行中 1 的个数只有一个的情况下，找到这个 1 的索引，然后计算这个索引所在的列的中的 1 的个数是否也是只有一个，如果是则计数器 result 加一，遍历结束得到的 result 为最终的结果。结果证明这种方法的速度更快，所占内存更小。

### 解答

	class Solution(object):
	    def numSpecial(self, mat):
	        """
	        :type mat: List[List[int]]
	        :rtype: int
	        """
	        result = 0
	        for i in range(len(mat)):
	            if mat[i].count(1) == 1:
	                i = mat[i].index(1)
	                col = [row[i] for row in mat]
	                if col.count(1) == 1:
	                    result += 1
	        return result

### 运行结果

	Runtime: 128 ms, faster than 97.14% of Python online submissions for Special Positions in a Binary Matrix.
	Memory Usage: 13.5 MB, less than 87.14% of Python online submissions for Special Positions in a Binary Matrix.

原题链接：https://leetcode.com/problems/special-positions-in-a-binary-matrix/



您的支持是我最大的动力
