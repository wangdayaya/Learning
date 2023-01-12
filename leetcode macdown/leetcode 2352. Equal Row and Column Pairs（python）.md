leetcode  2352. Equal Row and Column Pairs（python）




### 描述


Given a 0-indexed n x n integer matrix grid, return the number of pairs (Ri, Cj) such that row Ri and column Cj are equal.

A row and column pair is considered equal if they contain the same elements in the same order (i.e. an equal array).


Example 1:


![](https://assets.leetcode.com/uploads/2022/06/01/ex1.jpg)
	
	Input: grid = [[3,2,1],[1,7,6],[2,7,7]]
	Output: 1
	Explanation: There is 1 equal row and column pair:
	- (Row 2, Column 1): [2,7,7]
	
Example 2:

![](https://assets.leetcode.com/uploads/2022/06/01/ex2.jpg)

	
	Input: grid = [[3,1,2,2],[1,4,4,5],[2,4,2,2],[2,4,2,2]]
	Output: 3
	Explanation: There are 3 equal row and column pairs:
	- (Row 0, Column 0): [3,1,2,2]
	- (Row 2, Column 2): [2,4,2,2]
	- (Row 3, Column 2): [2,4,2,2]





Note:

	n == grid.length == grid[i].length
	1 <= n <= 200
	1 <= grid[i][j] <= 10^5


### 解析

根据题意，给定一个 0 索引的 n x n 整数矩阵网格 grid ，返回 (Ri, Cj) 对的数量，使得行 Ri 和列 Cj 相等。如果行和列对包含相同顺序的相同元素（即相等的数组），则认为它们是相等的。

其实这道题不用太复杂的解法，因为限制条件中 n 最长为 200 ，所以我们可以直接使用暴力的方法，对比每一行和每一列时候是否相等，如果相等则把 (Ri, Cj)  进行计数，最后返回统计的结果即可。

但是我们还是可以进行简化，我们将每一行进行计数得到 counter ，然后我们遍历每一列 c，如果有列和行 r 相等，那么就将 counter[r] 加入结果中，遍历完所有列返回结果即可。

时间复杂度为 O(N^2) ，空间复杂度为 O(N\*N) 。


### 解答

	class Solution:
	    def equalPairs(self, grid: List[List[int]]) -> int:
        	counter = collections.Counter(tuple(r) for r in grid)
    		return sum(counter[c] for c in zip(*grid))

### 运行结果

	69 / 69 test cases passed.
	Status: Accepted
	Runtime: 674 ms
	Memory Usage: 18.1 MB

### 原题链接

	https://leetcode.com/contest/weekly-contest-303/problems/equal-row-and-column-pairs/


您的支持是我最大的动力
