leetcode 1473. Paint House III （python）




### 描述

There is a row of m houses in a small city, each house must be painted with one of the n colors (labeled from 1 to n), some houses that have been painted last summer should not be painted again. A neighborhood is a maximal group of continuous houses that are painted with the same color.

* For example: houses = [1,2,2,3,3,2,1,1] contains 5 neighborhoods [{1}, {2,2}, {3,3}, {2}, {1,1}].

Given an array houses, an m x n matrix cost and an integer target where:

* houses[i]: is the color of the house i, and 0 if the house is not painted yet.
* cost[i][j]: is the cost of paint the house i with the color j + 1.

Return the minimum cost of painting all the remaining houses in such a way that there are exactly target neighborhoods. If it is not possible, return -1.



Example 1:


	Input: houses = [0,0,0,0,0], cost = [[1,10],[10,1],[10,1],[1,10],[5,1]], m = 5, n = 2, target = 3
	Output: 9
	Explanation: Paint houses of this way [1,2,2,1,1]
	This array contains target = 3 neighborhoods, [{1}, {2,2}, {1,1}].
	Cost of paint all houses (1 + 1 + 1 + 1 + 5) = 9.
	
Example 2:

	Input: houses = [0,2,1,2,0], cost = [[1,10],[10,1],[10,1],[1,10],[5,1]], m = 5, n = 2, target = 3
	Output: 11
	Explanation: Some houses are already painted, Paint the houses of this way [2,2,1,2,2]
	This array contains target = 3 neighborhoods, [{2,2}, {1}, {2,2}]. 
	Cost of paint the first and last house (10 + 1) = 11.


Example 3:


	Input: houses = [3,1,2,3], cost = [[1,1,1],[1,1,1],[1,1,1],[1,1,1]], m = 4, n = 3, target = 3
	Output: -1
	Explanation: Houses are already painted with a total of 4 neighborhoods [{3},{1},{2},{3}] different of target = 3.


Note:


	m == houses.length == cost.length
	n == cost[i].length
	1 <= m <= 100
	1 <= n <= 20
	1 <= target <= m
	0 <= houses[i] <= n
	1 <= cost[i][j] <= 10^4

### 解析

根据题意，一个小城市有一排 m 栋房子，每栋房子都必须涂上 n 种颜色中的一种（从 1 到 n 标出），有些去年夏天已经粉刷过的房子不应该再粉刷了。 一个社区是一组最大的连续房屋，它们涂有相同的颜色。

* 例如：houses = [1,2,2,3,3,2,1,1] 包含 5 个街区 [{1}, {2,2}, {3,3}, {2}, {1, 1}]。

给定一个数组 house ，一个 m x n 矩阵 cost 和一个整数 target，其中：

* houses[i]：是房子 i 的颜色，如果房子还没有涂则为 0 。
* cost[i][j]：是用颜色 j + 1 粉刷房子 i 的成本。

返回恰好粉刷出 target 个社区的最低成本。 如果不可能，则返回 -1 。

根据限制条件中的描述我们知道，m 最大为 100 ， n 最大为 20 ，我们直接使用记忆化的 DFS 来解题，根据题意写代码即可，index 是房子的索引，target 是代划分的街区个数，preColor 是前一个房子的颜色，我们分情况讨论即可：

* 如果 houses[index] 不为 0 ，也就是之前被粉刷过，我们直接根据和前一个房子是否相同来确认是否形成新的区
* 如果 houses[index] 为 0  ，也就是需要粉刷的房子，我们直接根据和前一个房子是否相等来确认是否形成新的区

使用动态规划也是同样的道理，只不过代码会显得比较长，但是状态转换之间的逻辑会更好懂一点。定义了一个三维的矩阵 dp[i][j][k] ，表示在前 i 个房子形成 j 个区的时候第 i 个房子的颜色为 k ，具体看注释即可。

时间复杂度为 O(m\*n^2\*t)，空间复杂度为 O(m\*n\*t)。

### 解答

	class Solution:
	    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
	        @lru_cache(None)
	        def dfs(index, target, preColor):
	            if target == -1 or index + target > m:
	                return float("inf")
	            if index == m:
	                return 0
	            if houses[index] != 0:
	                return dfs(index + 1, target if houses[index] == preColor else target - 1, houses[index])
	            else:
	                tmp = []
	                for j in range(n):
	                    a = dfs(index + 1, target if j + 1 == preColor else target - 1, j + 1)
	                    b = cost[index][j]
	                    tmp.append(a + b)
	                return min(tmp)
	        ans = dfs(0, target, -1)
	        return ans if ans != float("inf") else -1


### 运行结果

	Runtime: 618 ms, faster than 88.57% of Python3 online submissions for Paint House III.
	Memory Usage: 20.2 MB, less than 56.86% of Python3 online submissions for Paint House III.

### 解答

	class Solution(object):
	    def minCost(self, houses, cost, m, n, target):
	        houses = [0] + houses
	        cost.insert(0, [0]*n)
	        INF = float("inf")
	        dp = [[[INF] * (n + 1) for _ in range(target + 1)] for _ in range(m + 1)]
	
	        for k in range(n+1):
	            dp[0][0][k] = 0
	
	        for i in range(1, m + 1):
	            if houses[i] != 0: # 房子已经在去年被染色
	                for j in range(1, target+1):
	                    k = houses[i]
	                    for kk in range(1, n+1):
	                        if kk == k: # 和前一个房子颜色一样
	                            dp[i][j][k] = min(dp[i][j][k], dp[i-1][j][kk])
	                        else: # 和前一个房子颜色不一样
	                            dp[i][j][k] = min(dp[i][j][k], dp[i-1][j-1][kk])
	            else: # 房子还没有染色
	                for j in range(1, target+1):
	                    for k in range(1, n+1):
	                        for kk in range(1, n+1):
	                            if kk == k: # 当前房子选用染色 k ，和前一个房子颜色一样
	                                dp[i][j][k] = min(dp[i][j][k], dp[i - 1][j][kk] + cost[i][k-1])
	                            else: # 当前房子选用染色 k ，和前一个房子颜色不一样
	                                dp[i][j][k] = min(dp[i][j][k], dp[i - 1][j-1][kk] + cost[i][k-1])
	
	
	        res = INF
	        for i in range(1, n + 1): # 找最小的结果
	            res = min(res, dp[m][target][i])
	        return -1 if res == INF else res

### 运行结果

	Runtime: 6401 ms, faster than 5.20% of Python online submissions for Paint House III.
	Memory Usage: 18.2 MB, less than 54.74% of Python online submissions for Paint House III.
### 原题链接

https://leetcode.com/problems/paint-house-iii/


您的支持是我最大的动力
