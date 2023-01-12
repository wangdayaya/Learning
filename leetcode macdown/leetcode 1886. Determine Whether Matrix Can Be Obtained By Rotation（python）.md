leetcode  1886. Determine Whether Matrix Can Be Obtained By Rotation（python）

### 描述


Given two n x n binary matrices mat and target, return true if it is possible to make mat equal to target by rotating mat in 90-degree increments, or false otherwise.




Example 1:

![](https://assets.leetcode.com/uploads/2021/05/20/grid3.png)

	Input: mat = [[0,1],[1,0]], target = [[1,0],[0,1]]
	Output: true
	Explanation: We can rotate mat 90 degrees clockwise to make mat equal target.

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/05/20/grid4.png)

	Input: mat = [[0,1],[1,1]], target = [[1,0],[0,1]]
	Output: false
	Explanation: It is impossible to make mat equal to target by rotating mat.

Example 3:

![](https://assets.leetcode.com/uploads/2021/05/26/grid4.png)

	Input: mat = [[0,0,0],[0,1,0],[1,1,1]], target = [[1,1,1],[0,1,0],[0,0,0]]
	Output: true
	Explanation: We can rotate mat 90 degrees clockwise two times to make mat equal target.





Note:

	n == mat.length == target.length
	n == mat[i].length == target[i].length
	1 <= n <= 10
	mat[i][j] and target[i][j] are either 0 or 1.


### 解析

根据题意，就是判断 mat 经过多次的 90 度的旋转后，能和 target 相等。关键在于定义反转二维列表的函数，只要定义好这个函数，然后依次旋转 3 次即可，每次旋转完之后都判断是否与 target 相等。如果相等直接返回 True 。旋转 3 次之后仍然不相等则直接返回 False 。因为旋转 4 次就是自己本身。


### 解答
				
	
	class Solution(object):
	    def findRotation(self, mat, target):
	        """
	        :type mat: List[List[int]]
	        :type target: List[List[int]]
	        :rtype: bool
	        """
	        if mat == target:
	            return True
	        def rotate(mat):
	            N = len(mat)
	            result = [[0]*N for _ in range(N)]
	            for i in range(N):
	                for j in range(N):
	                    result[j][N-i-1] = mat[i][j]  
	            return result
	        for i in range(3):
	            mat = rotate(mat)
	            if mat == target:
	                return True
	        return False
            	      
			
### 运行结果


	Runtime: 32 ms, faster than 73.10% of Python online submissions for Determine Whether Matrix Can Be Obtained By Rotation.
	Memory Usage: 13.4 MB, less than 72.22% of Python online submissions for Determine Whether Matrix Can Be Obtained By Rotation.


原题链接：https://leetcode.com/problems/determine-whether-matrix-can-be-obtained-by-rotation/



您的支持是我最大的动力
