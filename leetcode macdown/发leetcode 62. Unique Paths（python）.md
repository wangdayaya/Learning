leetcode  62. Unique Paths（python）

### 描述


A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?


Example 1:

![](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)

	Input: m = 3, n = 7
	Output: 28

	
Example 2:


	Input: m = 3, n = 2
	Output: 3
	Explanation:
	From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
	1. Right -> Down -> Down
	2. Down -> Down -> Right
	3. Down -> Right -> Down

Example 3:

	
	Input: m = 7, n = 3
	Output: 28
	
Example 4:

	Input: m = 3, n = 3
	Output: 6

	



Note:

	1 <= m, n <= 100
	It's guaranteed that the answer will be less than or equal to 2 * 109.




### 解析


根据题意，就是给出了一个  m x n 大小的方格矩阵，左上角是起始位置，右下角是结束位置，问从开始位置走到结束位置过程中，只能向下和向右走，一共有多少种不同的路径。

如果 m 和 n 小的话可以用回溯法，但是因为他们都太大了，这里用动态规划进行解题：

* 初始化一个全 1 元素的 m x n 大小的二维列表 dp ，每个元素来表示从起始位置到当前位置的不同路径数量
* 遍历除第一行和第一列的所有元素，每个位置上 dp[i][j] = dp[i][j-1] + dp[i-1][j] ，表示从起始位置到当前位置的不同路径数，是从起始位置到左边位置的不同路径数加上从起始位置到上边位置的不同路径数
* 遍历结束得到的 dp[-1][-1] 就是答案，即从起始位置到终点位置的不同路径数

### 解答
				
	class Solution(object):
	    def uniquePaths(self, m, n):
	        """
	        :type m: int
	        :type n: int
	        :rtype: int
	        """
	        dp = [[1]*n for _ in range(m)]
	        for i in range(1,m):
	            for j in range(1,n):
	                dp[i][j] = dp[i][j-1] + dp[i-1][j]
	        return dp[-1][-1]
            	      
			
### 运行结果
	
	Runtime: 23 ms, faster than 30.01% of Python online submissions for Unique Paths.
	Memory Usage: 13.4 MB, less than 69.94% of Python online submissions for Unique Paths.

### 解析

另外可以将动态规划的 dp 压缩成一维的列表进行解题，原理和上面大同小异，  df[j]  = df[j]+df[j-1] 就是相当于上面解法的 dp[i][j] = dp[i][j-1] + dp[i-1][j] ，能节省不少的内存。


### 解答

	class Solution(object):
	    def uniquePaths(self, m, n):
	        """
	        :type m: int
	        :type n: int
	        :rtype: int
	        """
	        df = [1] * n
	        for i in range(1, m):
	            for j in range(1,n):
	                df[j] += df[j-1]
	        return df[-1]

### 运行结果

	Runtime: 27 ms, faster than 14.44% of Python online submissions for Unique Paths.
	Memory Usage: 13.4 MB, less than 69.94% of Python online submissions for Unique Paths.
	
	
### 解析	

用数学的方式找规律，直接计算进行解题。
	
### 解答

	class Solution(object):
	    def uniquePaths(self, m, n):
	        """
	        :type m: int
	        :type n: int
	        :rtype: int
	        """
	        return math.factorial(m+n-2)/(math.factorial(n-1) * math.factorial(m-1))

### 运行结果

	Runtime: 38 ms, faster than 5.12% of Python online submissions for Unique Paths.
	Memory Usage: 13.5 MB, less than 44.34% of Python online submissions for Unique Paths.

### 相似题

* [63. Unique Paths II](https://leetcode.com/problems/unique-paths-ii/)
* [980. Unique Paths III](https://leetcode.com/problems/unique-paths-iii/)

原题链接：https://leetcode.com/problems/unique-paths/



您的支持是我最大的动力
