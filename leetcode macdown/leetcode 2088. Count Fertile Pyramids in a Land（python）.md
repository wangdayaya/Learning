leetcode  2088. Count Fertile Pyramids in a Land（python）

### 描述


A farmer has a rectangular grid of land with m rows and n columns that can be divided into unit cells. Each cell is either fertile (represented by a 1) or barren (represented by a 0). All cells outside the grid are considered barren.

A pyramidal plot of land can be defined as a set of cells with the following criteria:

* The number of cells in the set has to be greater than 1 and all cells must be fertile.
* The apex of a pyramid is the topmost cell of the pyramid. The height of a pyramid is the number of rows it covers. Let (r, c) be the apex of the pyramid, and its height be h. Then, the plot comprises of cells (i, j) where r <= i <= r + h - 1 and c - (i - r) <= j <= c + (i - r).

An inverse pyramidal plot of land can be defined as a set of cells with similar criteria:

* The number of cells in the set has to be greater than 1 and all cells must be fertile.
* The apex of an inverse pyramid is the bottommost cell of the inverse pyramid. The height of an inverse pyramid is the number of rows it covers. Let (r, c) be the apex of the pyramid, and its height be h. Then, the plot comprises of cells (i, j) where r - h + 1 <= i <= r and c - (r - i) <= j <= c + (r - i).

Some examples of valid and invalid pyramidal (and inverse pyramidal) plots are shown below. Black cells indicate fertile cells.


Given a 0-indexed m x n binary matrix grid representing the farmland, return the total number of pyramidal and inverse pyramidal plots that can be found in grid.


Example 1:

![](https://assets.leetcode.com/uploads/2021/10/23/exa12.png)

	Input: grid = [[0,1,1,0],[1,1,1,1]]
	Output: 2
	Explanation:
	The 2 possible pyramidal plots are shown in blue and red respectively.
	There are no inverse pyramidal plots in this grid. 
	Hence total number of pyramidal and inverse pyramidal plots is 2 + 0 = 2.

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/10/23/exa22.png)

	Input: grid = [[1,1,1],[1,1,1]]
	Output: 2
	Explanation:
	The pyramidal plot is shown in blue, and the inverse pyramidal plot is shown in red. 
	Hence the total number of plots is 1 + 1 = 2.


Example 3:

![](https://assets.leetcode.com/uploads/2021/10/23/eg3.png)

	Input: grid = [[1,0,1],[0,0,0],[1,0,1]]
	Output: 0
	Explanation:
	There are no pyramidal or inverse pyramidal plots in the grid.

	
Example 4:


![](https://assets.leetcode.com/uploads/2021/10/23/eg42.png)

	Input: grid = [[1,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[0,1,0,0,1]]
	Output: 13
	Explanation:
	There are 7 pyramidal plots, 3 of which are shown in the 2nd and 3rd figures.
	There are 6 inverse pyramidal plots, 2 of which are shown in the last figure.
	The total number of plots is 7 + 6 = 13.

Note:

	m == grid.length
	n == grid[i].length
	1 <= m, n <= 1000
	1 <= m * n <= 10^5
	grid[i][j] is either 0 or 1.



### 解析

根据题意，一个农民有一个长方形的 mxn 的土地网格可以分成单位格。每个格子都是肥沃的（用 1 表示）或贫瘠的（用 0 表示）。网格外的所有单元格都被认为是贫瘠的。一个金字塔形地块可以定义为一组具有以下标准的单元格：

* 集合中的格子数必须大于 1，并且所有细胞都必须是肥沃的。
* 金字塔的顶点是金字塔的最顶部单元格。金字塔的高度是它覆盖的行数。令 (r, c) 为金字塔的顶点，其高度为 h。然后，该图由单元格 (i, j) 组成，其中 r <= i <= r + h - 1 且 c - (i - r) <= j <= c + (i - r)。

土地的倒金字塔地块可以定义为一组具有类似标准的单元格：

* 集合中的格子数必须大于 1，并且所有细胞都必须是肥沃的。
* 倒金字塔的顶点是倒金字塔最底部的单元格。倒金字塔的高度是它覆盖的行数。令 (r, c) 为金字塔的顶点，其高度为 h。然后，该图由单元格 (i, j) 组成，其中 r - h + 1 <= i <= r 和 c - (r - i) <= j <= c + (r - i)。图中显示了一些有效和无效金字塔（和倒金字塔）图的示例。黑色细胞表示可育细胞。

![](https://assets.leetcode.com/uploads/2021/11/08/image.png)

给定一个表示农田的 0 索引 m x n 二元矩阵 grid ，返回可以在网格中找到的金字塔形和反金字塔形图的总数。

一看到这中数数的题目，基本上第一反应用动态规划基本上方向没啥问题。这道题的关键就是找出使用动态规划的规律，金字塔形状的农田有上下两种形态，只要解决其中的正金字塔基本就能同样解决倒金字塔。

如果以金字塔的塔尖作为特征，那么光是遍历完所有的格子的时间复杂度为 O(MxN) ，作为塔尖又要向下找 n 层来找合法的 n 层金字塔，那么时间复杂度至少要 O(MxNxM) ，基本上时会超时的。那么我们反过来用金字塔的底座的中间点作为特征，如果某个格子 X 是三层金字塔的底层中点，其一定也是一个二层金字塔的底层中点，这样计算量为大幅下降。

想要直到某格子 X 是多长底层的中点，可以找该点左边和右边的格子较小数得知该金字塔底层的半径 L ，其次我们知道如果该格子 X 是底层中点，那么 X 上面的点肯定至少是半径为 L-1 的金字塔的底层中点，所以定义 dp[i][j] 为以该点为底层中点的正金字塔数量，dp_r[i][j] 为以该点为顶层中点的倒金字塔数量，动态规划规则为，正金字塔：

	dp[i][j] = min(min(left[i][j], right[i][j]), dp[i-1][j]+1)
	
或者，倒金字塔：
	
	dp_r[i][j] = min(min(left[i][j], right[i][j]), dp_r[i+1][j]+1)


### 解答
				
	class Solution(object):
	    def countPyramids(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        M,N = len(grid),len(grid[0])
	        left = [[0]*N for _ in range(M)]
	        right = [[0]*N for _ in range(M)]
	        dp = [[0]*N for _ in range(M)]
	        dp_r = [[0]*N for _ in range(M)]
	        
	        for i in range(M):
	            count = 0
	            for j in range(N):
	                if grid[i][j] == 0:
	                    count = 0
	                else:
	                    count += 1
	                left[i][j] = count
	                
	        for i in range(M):
	            count = 0
	            for j in range(N-1,-1,-1):
	                if grid[i][j] == 0:
	                    count = 0
	                else:
	                    count += 1
	                right[i][j] = count
	            
	        result = 0
	        for i in range(M):
	            for j in range(N):
	                if grid[i][j] == 0: continue
	                if i == 0:
	                    dp[i][j] = 1
	                else:
	                    dp[i][j] = min(min(left[i][j], right[i][j]), dp[i-1][j]+1)
	                result += dp[i][j] - 1
	                
	        for i in range(M-1,-1,-1):
	            for j in range(N):
	                if grid[i][j] == 0: continue
	                if i == M-1:
	                    dp_r[i][j] = 1
	                else:
	                    dp_r[i][j] = min(min(left[i][j], right[i][j]), dp_r[i+1][j]+1)
	                result += dp_r[i][j] - 1
	                
	        return result
            	      
			
### 运行结果

	Runtime: 1304 ms, faster than 67.60% of Python online submissions for Count Fertile Pyramids in a Land.
	Memory Usage: 21.9 MB, less than 14.08% of Python online submissions for Count Fertile Pyramids in a Land.

### 解析

另外一种动态规划，定义 dp[i][j] 表示以 (i, j) 为顶点的最大金字塔的层数。动态规划规则如下，对于正金字塔：

	dp[i][j] = min(dp[i + 1][j - 1], dp[i + 1][j + 1]) + 1
	
对于倒金字塔：

	dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j + 1]) + 1
	
### 解答
	class Solution(object):
	    def countPyramids(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        M, N, dp, result = len(grid), len(grid[0]), copy.deepcopy(grid), 0
	        # triangle
	        for i in range(M - 2, -1, -1):
	            for j in range(1, N - 1):
	                if dp[i][j] > 0 and dp[i + 1][j] > 0:
	                    dp[i][j] = min(dp[i + 1][j - 1], dp[i + 1][j + 1]) + 1
	                    result += dp[i][j] - 1
	        # inverted triangle
	        dp = grid
	        for i in range(1, M):
	            for j in range(1, N - 1):
	                if dp[i][j] > 0 and dp[i - 1][j] > 0:
	                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j + 1]) + 1
	                    result += dp[i][j] - 1
	        return result
	        
	        
### 运行结果

	Runtime: 1268 ms, faster than 73.24% of Python online submissions for Count Fertile Pyramids in a Land.
	Memory Usage: 16.2 MB, less than 53.52% of Python online submissions for Count Fertile Pyramids in a Land.  

原题链接：https://leetcode.com/problems/count-fertile-pyramids-in-a-land/



您的支持是我最大的动力
