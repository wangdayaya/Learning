leetcode 2245. Maximum Trailing Zeros in a Cornered Path （python）

这道题是第 289 场 leetcode 周赛的第三题，难度为 Medium ，主要考察的是就是对字符串和列表的基本操作。


### 描述


You are given a 2D integer array grid of size m x n, where each cell contains a positive integer.

A cornered path is defined as a set of adjacent cells with at most one turn. More specifically, the path should exclusively move either horizontally or vertically up to the turn (if there is one), without returning to a previously visited cell. After the turn, the path will then move exclusively in the alternate direction: move vertically if it moved horizontally, and vice versa, also without returning to a previously visited cell.

The product of a path is defined as the product of all the values in the path.

Return the maximum number of trailing zeros in the product of a cornered path found in grid.

Note:

* Horizontal movement means moving in either the left or right direction.
* Vertical movement means moving in either the up or down direction.


Example 1:


![](https://assets.leetcode.com/uploads/2022/03/23/ex1new2.jpg)

	Input: grid = [[23,17,15,3,20],[8,1,20,27,11],[9,4,6,2,21],[40,9,1,10,6],[22,7,4,5,3]]
	Output: 3
	Explanation: The grid on the left shows a valid cornered path.
	It has a product of 15 * 20 * 6 * 1 * 10 = 18000 which has 3 trailing zeros.
	It can be shown that this is the maximum trailing zeros in the product of a cornered path.
	
	The grid in the middle is not a cornered path as it has more than one turn.
	The grid on the right is not a cornered path as it requires a return to a previously visited cell.
	
Example 2:

![](https://assets.leetcode.com/uploads/2022/03/25/ex2.jpg)

	Input: grid = [[4,3,2],[7,6,1],[8,8,8]]
	Output: 0
	Explanation: The grid is shown in the figure above.
	There are no cornered paths in the grid that result in a product with a trailing zero.





Note:


* 	m == grid.length
* 	n == grid[i].length
* 	1 <= m, n <= 10^5
* 	1 <= m * n <= 10^5
* 	1 <= grid[i][j] <= 1000

### 解析

根据题意，给定一个大小为 m x n 的 2D 整数数组 grid ，其中每个单元格包含一个正整数。

拐角路径定义为一组最多有一个转弯的相邻单元。 更具体地说，其实路径就是一条形状为“一” 的路径，或者形状为“L” 的路径，或者形状为“｜” 的路径，但是“一” 、“L” 和“｜” 可能是任意大小，也可能是任意方向的，当 “L”形状中的某一边缩短为零也就退化成了其他两种。路径的乘积定义为路径中所有值的乘积。返回在网格中找到的拐角路径的乘积结果可能包含的最多个末尾 0 的个数。

根据上面的推理我们知道一共有四种形态，如图所示

![](https://assets.leetcode.com/users/images/b91a94f3-ca21-49d7-af85-662f80023fa0_1650169235.6731899.png)

我们只需要对每一个格子进行下面的运算然后比较大小取最大值结果 result 即可

![](https://assets.leetcode.com/users/images/20d53765-2b44-455b-90d4-c44d94c7388d_1650169239.717305.png)

运算清楚之后，难点就剩下了怎么找乘积“末尾 0 ”的个数，我们知道，10 = 2\* 5 ，所以我们将一个数字尽量拆分成若干个 2 和 5 的乘积，那么 2 和 5 中较少的出现个数就是 10 出现的个数，也就是乘积“末尾 0 ”的个数。如 1500 = 3\*2\*2\*5\*5\*5 ，其中 2 出现了 2 次， 5 出现了 3 次，那么两者出现次数的较小值为 2 次，所以相乘之后 10 出现的个数为 2 次，也就是 1500 末尾有 2 个 0 。

我们只需要定义一个二维数组 left 存储每一行中每个格子从左到右的 2/5 出现的累积次数，定义一个二维数字 top 存储每一列中每个格子从上到下的 2/5 出现的累积次数，最后进行所有格子的遍历，计算结果取最大值返回即可。

因为限制条件中提到 m \* n 最大为 10^5 ，所以时间复杂度为 O(N) ，空间复杂度为 O(N) 。

本解答过程参考的是这位大佬的，如果我没有解释清楚，请移步观看大佬本人的解释：https://leetcode.com/problems/maximum-trailing-zeros-in-a-cornered-path/discuss/1955607/Python3-Explanation-with-pictures-prefix-sum.


### 解答
				

	class Solution(object):
	    def maxTrailingZeros(self, grid):
	        """
	        :type grid: List[List[int]]
	        :rtype: int
	        """
	        m = len(grid)
	        n = len(grid[0])
	        left = [[[0,0] for _ in range(n)] for _ in range(m)]
	        top = [[[0,0] for _ in range(n)] for _ in range(m)]
	
	        def helper(n):
	            a,b = 0,0
	            while n%2 == 0:
	                n //= 2
	                a += 1
	            while n%5 == 0:
	                n //= 5
	                b += 1
	            return [a,b]
	
	        for i in range(m):
	            for j in range(n):
	                if j == 0:
	                    left[i][j] = helper(grid[i][j])
	                else:
	                    a, b = helper(grid[i][j])
	                    left[i][j][0] = left[i][j - 1][0] + a
	                    left[i][j][1] = left[i][j - 1][1] + b
	
	        for j in range(n):
	            for i in range(m):
	                if i == 0:
	                    top[i][j] = helper(grid[i][j])
	                else:
	                    a,b = helper(grid[i][j])
	                    top[i][j][0] = top[i-1][j][0] + a
	                    top[i][j][1] = top[i-1][j][1] + b
	
	        result = 0
	        for i in range(m):
	            for j in range(n):
	                a, b = top[m-1][j]
	                c, d = left[i][n-1]
	                x, y = helper(grid[i][j])
	                e, f = top[i][j]
	                g, h = left[i][j]
	                result = max(result, min([e+g-x, f+h-y]))
	                result = max(result, min([c-g+e, d-h+f]))
	                result = max(result, min([a-e+g, b-f+h]))
	                result = max(result, min([a-e+c-g+x, b-f+d-h+y]))
	
	        return result
            	      
			
### 运行结果


	54 / 54 test cases passed.
	Status: Accepted
	Runtime: 4905 ms
	Memory Usage: 87.6 MB


### 原题链接


https://leetcode.com/contest/weekly-contest-289/problems/maximum-trailing-zeros-in-a-cornered-path/


您的支持是我最大的动力
