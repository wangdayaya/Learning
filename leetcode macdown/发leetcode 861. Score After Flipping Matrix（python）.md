leetcode  861. Score After Flipping Matrix（python）

### 描述

You are given an m x n binary matrix grid.

A move consists of choosing any row or column and toggling each value in that row or column (i.e., changing all 0's to 1's, and all 1's to 0's).

Every row of the matrix is interpreted as a binary number, and the score of the matrix is the sum of these numbers.

Return the highest possible score after making any number of moves (including zero moves).





Example 1:

![](https://assets.leetcode.com/uploads/2021/07/23/lc-toogle1.jpg)

	Input: grid = [[0,0,1,1],[1,0,1,0],[1,1,0,0]]
	Output: 39
	Explanation: 0b1111 + 0b1001 + 0b1111 = 15 + 9 + 15 = 39

	
Example 2:


	Input: grid = [[0]]
	Output: 1



Note:

	m == grid.length
	n == grid[i].length
	1 <= m, n <= 20
	grid[i][j] is either 0 or 1.


### 解析

根据题意，就是给出了一个 m*n 的矩阵 grid ，执行任意次数的特殊的操作，然后将每一行的数字当作二进制数，将所有的行形成的二进制进行相加，可以得到的最大的和是多少。

特殊操作就是可以将任意一行或者任意一列中的数字从 0 变成 1 ，或者从 1 变成 0 。

其实分析案例我们就可以发现，只需要从两个维度来进行操作即可：

* 第一个维度就是行，每一行只要将 1 的位置尽可能地向前移动，那我们最后得到的每一行元素形成的二进制数就会越大，也就是第一个数字如果为 0 那么就进行一次操作，否则不进行操作
* 第二个维度就是列，每一列只要 1 的个数尽可能的多，那我们最后将每一行相加的时候得到的结果也会越大，也就是每一列中如果 0 的个数小于 1 的数则进行一次操作，否则不进行操作


### 解答
				
	class Solution(object):
	    def matrixScore(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        # 行数
	        M = len(grid)
	        # 列数
	        N = len(grid[0])
	        # 记录每一列 1 的个数
	        count = [0] * N
	        # 每一行中如果第一个元素为 0 ，则执行反转数字操作
	        for i in range(M):
	            row = grid[i]
	            # 如果第一个元素为 0 ，则进行反转操作
	            if row[0] == 0:
	                for j in range(N):
	                    if row[j] == 0:
	                        row[j] = 1
	                        count[j] += 1
	                    else:
	                        row[j] = 0
	            # 否则只进行每一列 1 的个数的统计
	            else:
	                for j in range(N):
	                    if row[j] == 1:
	                        count[j] += 1
	        # 每一列中如果 1 的个数小于 0 ，则执行反转数字操作
	        for j in range(N):
	            if count[j] < M//2+M%2:
	                for i in range(M):
	                    if grid[i][j] == 0:
	                        grid[i][j] = 1
	                    else:
	                        grid[i][j] = 0
	        result = 0
	        # 将每一行表示的二进制相加
	        for row in grid:
	            tmp = int(''.join([str(x) for x in row]),2)
	            result += tmp
	        return result

            	      
			
### 运行结果

	
	Runtime: 24 ms, faster than 77.42% of Python online submissions for Score After Flipping Matrix.
	Memory Usage: 13.5 MB, less than 19.35% of Python online submissions for Score After Flipping Matrix.
	
### 解析


将上述过程简化一下，使用异或运算和 zip 函数可以省去很多代码。

### 解答



	class Solution(object):
	    def matrixScore(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        for row in grid:
	            if row[0] == 1:
	                continue
	            for i in range(0, len(row)):
	                row[i] ^= 1
	        grid = [list(row) for row in zip(*grid)]
	        for row in grid:
	            if row.count(0) > len(row) // 2:
	                for i in range(0, len(row)):
	                    row[i] ^= 1
	        res = 0
	        for row in zip(*grid):
	            res += int(''.join(str(char) for char in row), 2)
	        return res

### 运行结果

	Runtime: 16 ms, faster than 96.77% of Python online submissions for Score After Flipping Matrix.
	Memory Usage: 13.4 MB, less than 70.97% of Python online submissions for Score After Flipping Matrix.
	
	
### 解析

本题很适合贪心优化算法求解，代码会大幅缩减，直接看 [高手分析](https://leetcode.com/problems/score-after-flipping-matrix/discuss/843685/Python-3-or-Greedy-and-Optimization-(5-lines)-or-Explanation)

### 解答

	class Solution(object):
	    def matrixScore(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        m, n, ans = len(grid), len(grid[0]), 0
	        for c in range(n):
	            col = sum(grid[r][c] == grid[r][0] for r in range(m))
	            ans += max(col, m-col) * 2 ** (n-1-c)
	        return ans
	        
### 运行结果

	Runtime: 20 ms, faster than 87.10% of Python online submissions for Score After Flipping Matrix.
	Memory Usage: 13.6 MB, less than 19.35% of Python online submissions for Score After Flipping Matrix.


原题链接：https://leetcode.com/problems/score-after-flipping-matrix/



您的支持是我最大的动力
