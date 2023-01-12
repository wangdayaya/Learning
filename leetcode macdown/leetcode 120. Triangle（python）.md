leetcode  120. Triangle（python）




### 描述

Given a triangle array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.



Example 1:

	Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
	Output: 11
	Explanation: The triangle looks like:
	   2
	  3 4
	 6 5 7
	4 1 8 3
	The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).

	
Example 2:

	Input: triangle = [[-10]]
	Output: -10



Note:


	Could you do this using only O(n) extra space, where n is the total number of rows in the triangle?

### 解析


根据题意，给定一个三角形数组，返回从上到下的最小路径和。对于每个步骤，可以移动到下一行的相邻编号。 如果您在当前行的索引 i 上，您可以移动到下一行的索引 i 或索引 i + 1 。

这道题和刚过去的第 297 场周赛的第二题非常像，考查的都是动态规划，如果想多练习一下可以去做做 [【2304. Minimum Path Cost in a Grid】](https://leetcode.com/contest/weekly-contest-297/problems/minimum-path-cost-in-a-grid/) 。

因为每个单位所移动的范围只有两个位置，想最后找出一条最小和的路径，这类题用动态规划是最好的。我们定义一个 N \* N 的 dp ，dp[i][j] 表示从上往下走到 triangle[i][j] 的最小路径和是多少，我们使用双重循环遍历每个位置  triangle[i][j]  ，并不断更新 dp[i][j] ，一直到最后，我们取 dp 中最后一行的最小值就是最后的结果。

时间复杂度为 O(N\^2) ，空间复杂度为 O(N^2) 。



### 解答
				
	class Solution(object):
	    def minimumTotal(self, triangle):
	        """
	        :type triangle: List[List[int]]
	        :rtype: int
	        """
	        M = len(triangle)
	        N = len(triangle)
	        dp = [[float('inf')]*N for _ in range(M)]
	        dp[0][0] = triangle[0][0]
	        for i in range(1, N):
	            for j in range(i+1):
	                for k in [j-1, j]:
	                    if 0 <= k <= j:
	                        dp[i][j] = min(dp[i-1][k] + triangle[i][j], dp[i][j])
	        return min(dp[-1])
            	      
			
### 运行结果


	Runtime: 74 ms, faster than 41.15% of Python online submissions for Triangle.
	Memory Usage: 14.6 MB, less than 20.14% of Python online submissions for Triangle.
	
### 解析


题目中给出更高的要求，让我们使用 O(N) 的空间复杂度，其实我们可以使用 O(1) 的空间复杂度，我们发现当在遍历下面行的时候，之前遍历的行就不再使用了，因此我们可以在遍历的时候直接将已经经过的和放入 triangle 中，原理和上面大同小异，在代码结构上要仔细构造即可实现。

这种是从上往下进行的，其实从下往上进行的也是可以的，原理大同小异，大家有兴趣的可以试试，AC 速度会更快一点，因为经过计算最后我们在 triangle[0][0] 得到的值就是答案。

### 解答

	class Solution(object):
	    def minimumTotal(self, triangle):
	        """
	        :type triangle: List[List[int]]
	        :rtype: int
	        """
	        M = len(triangle)
	        for i in range(1, M):
	            for j in range(i+1):
	                a = triangle[i-1][min(j,i-1)]
	                b = triangle[i-1][max(0, j-1)]
	                triangle[i][j] += min(a, b)
	        return min(triangle[-1])
	        
### 运行结果

	Runtime: 73 ms, faster than 43.23% of Python online submissions for Triangle.
	Memory Usage: 14.4 MB, less than 36.63% of Python online submissions for Triangle.

### 解析
如果非要想按照题目要求的使用 O(N) 的空间复杂度，那么我们就将二维的 dp 压缩成一维的 dp ，因为我们已经用的行的 dp 在之后的计算中就不会再出现了，但是使用一维 dp 我们需要注意的是，我们要定义两个一维数组 pre 和 cur ，pre 保存的是上一行已经计算之后的路径和，cur 保存的是当前行计算之后的路径和，然后我们再将 pre 和 cur 交换，继续进行后面的路径和的计算。计算细节和上面的一样。

当然了这个解法也可以从底向上进行计算，有兴趣的同学可以试试。

时间复杂度为 O(N^2) ，空间复杂度为 O(N) 。

### 解答
	class Solution(object):
	    def minimumTotal(self, triangle):
	        """
	        :type triangle: List[List[int]]
	        :rtype: int
	        """
	        N = len(triangle)
	        cur_row, pre_row = [0] * N, [0] * N
	        pre_row[0] = triangle[0][0]
	        for i in range(1, N):
	            for j in range(i + 1):
	                cur_row[j] = triangle[i][j] + min(pre_row[max(0, j-1)], pre_row[min(j, i-1)])
	            pre_row, cur_row = cur_row, pre_row
	        return min(pre_row)

### 运行结果

Runtime: 72 ms, faster than 45.31% of Python online submissions for Triangle.
Memory Usage: 14.3 MB, less than 78.99% of Python online submissions for Triangle.


### 原题链接

https://leetcode.com/problems/triangle/



您的支持是我最大的动力
