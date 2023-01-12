leetcode  1074. Number of Submatrices That Sum to Target（python）




### 描述

Given a matrix and a target, return the number of non-empty submatrices that sum to target. A submatrix x1, y1, x2, y2 is the set of all cells matrix[x][y] with x1 <= x <= x2 and y1 <= y <= y2. Two submatrices (x1, y1, x2, y2) and (x1', y1', x2', y2') are different if they have some coordinate that is different: for example, if x1 != x1'.



Example 1:

![](https://assets.leetcode.com/uploads/2020/09/02/mate1.jpg)

	Input: matrix = [[0,1,0],[1,1,1],[0,1,0]], target = 0
	Output: 4
	Explanation: The four 1x1 submatrices that only contain 0.

	
Example 2:


	Input: matrix = [[1,-1],[-1,1]], target = 0
	Output: 5
	Explanation: The two 1x2 submatrices, plus the two 2x1 submatrices, plus the 2x2 submatrix.

Example 3:

	Input: matrix = [[904]], target = 0
	Output: 0



Note:


	1 <= matrix.length <= 100
	1 <= matrix[0].length <= 100
	-1000 <= matrix[i] <= 1000
	-10^8 <= target <= 10^8

### 解析

根据题意，给定一个矩阵 matrix 和一个目标 target ，返回总和为 target 的非空子矩阵的数量。 子矩阵 x1, y1, x2, y2 是所有单元矩阵 [x][y] 的集合，其中 x1 <= x <= x2 和 y1 <= y <= y2。 如果两个子矩阵 (x1, y1, x2, y2) 和 (x1', y1', x2', y2') 中有一些不同的坐标，则它们是不同的。

这种子矩阵求和的问题基本上都是用前缀和来解决，我们先定义一个二维数组 dp ，dp[i][j] 表示从矩阵左上角到 [i,j] 形成的矩阵的和，然后我们只需要通过遍历，找出所有的子矩阵，然后通过公式计算出其面积是否等于 target 即可：

	dp[p + 1][q + 1] - dp[p+1][j] - dp[i][q+1] + dp[i][j]
	
但是这种解法可能会超时，因为会出现四层循环，每层循环数据量最大为 10^2 ，最多会出现 10^8 的时间复杂度，所以要进行优化。

我们在确定了一个子矩阵的上下边界的时候，只需要不断遍历其右边界过程中，把子矩阵的右边界到原矩阵左边界形成的矩阵和 current 存入字典中，字典统计了不同面积出现的次数，因为我们想找到 target 的子矩阵，也就是找到一个子矩阵的左边界使得矩阵和为 target ，这相当于从字典中找一个 x ，使得 current-x = target ，这个操作是 O(1) 的，这样可以使的时间复杂度下降到 10^6 。

时间复杂度为 O(M\*N^2) , 空间复杂度为(M\*N) 。



### 解答

	class Solution(object):
	    def numSubmatrixSumTarget(self, matrix, target):
	        """
	        :type matrix: List[List[int]]
	        :type target: int
	        :rtype: int
	        """
	        M = len(matrix)
	        N = len(matrix[0])
	        dp = [[0] * (N + 1) for _ in range(M + 1)]
	        for i in range(1, M + 1):
	            for j in range(1, N + 1):
	                dp[i][j] = dp[i - 1][j] + dp[i][j - 1] - dp[i - 1][j - 1] + matrix[i - 1][j - 1]
	        result = 0
	        for i in range(1, M+1):
	            for j in range(i,M+1):
	                d = collections.defaultdict(int)
	                for right in range(N + 1):
	                    current = dp[j][right] - dp[i - 1][right]
	                    if current - target in d:
	                        result += d[current - target]
	                    d[current] += 1
	        return result
	


### 运行结果

	Runtime: 718 ms, faster than 87.50% of Python online submissions for Number of Submatrices That Sum to Target.
	Memory Usage: 14.4 MB, less than 75.00% of Python online submissions for Number of Submatrices That Sum to Target.

### 原题链接


https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/

您的支持是我最大的动力
