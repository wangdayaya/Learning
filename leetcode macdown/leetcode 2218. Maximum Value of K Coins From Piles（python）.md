leetcode 2218. Maximum Value of K Coins From Piles （python）




### 描述

There are n piles of coins on a table. Each pile consists of a positive number of coins of assorted denominations.

In one move, you can choose any coin on top of any pile, remove it, and add it to your wallet.

Given a list piles, where piles[i] is a list of integers denoting the composition of the i<sub>th</sub> pile from top to bottom, and a positive integer k, return the maximum total value of coins you can have in your wallet if you choose exactly k coins optimally.



Example 1:

![](https://assets.leetcode.com/uploads/2019/11/09/e1.png)


	Input: piles = [[1,100,3],[7,8,9]], k = 2
	Output: 101
	Explanation:
	The above diagram shows the different ways we can choose k coins.
	The maximum total we can obtain is 101.

	
Example 2:

	Input: piles = [[100],[100],[100],[100],[100],[100],[1,1,1,1,1,1,700]], k = 7
	Output: 706
	Explanation:
	The maximum total can be obtained if we choose all coins from the last pile.





Note:


	n == piles.length
	1 <= n <= 1000
	1 <= piles[i][j] <= 10^5
	1 <= k <= sum(piles[i].length) <= 2000

### 解析

根据题意，桌子上有 n 堆硬币。 每堆由正数个各种面额的硬币组成。我们只能操作一次，可以选择任何一堆硬币上的硬币，将其取出并将其添加到钱包中。

给定一个列表 piles，其中 piles[i] 是一个整数列表，表示从上到下的第 i 堆硬币的组成，以及一个正整数 k，返回钱包中可以拥有的最大硬币总值 k 个硬币。

这道题很明显使用动态规划来做的，我们假定 DP[i][j] 表示在前 i 个罐子里取 j 个硬币的最大总值，那么 DP[i][j] 的动态变化关系就是 ：

	dp[i][j] = max(dp[i][j], dp[i-1][j-t]+preSum[i][t])
	
其中假如第 i 个罐子要取 t 个硬币，那么 dp[i-1][j-t] 就是前 i-1 个罐子中取了 j-t 个硬币的最大总值加上当前罐子前 t 个硬币的总值，可以和 dp[i][j] 比较取较大值，这就是当前 dp[i][j] 的值。当然在动态规划过程中需要特别注意的就是注意边界条件，当 i 为 0 的时候 dp 值都为 0 。最后只需要返回 dp[N][k-1] 即可， N 是 piles 长度。

时间复杂度为 O(K \* sum(piles[i].length) ) ，别看代码上是三层循环，其实就是遍历了一次 k ，还遍历了一次 piles 中所有的元素，基本在 10^6 数量级，刚刚没有超时，空间复杂度为 O(N\*K) 。


### 解答
				

	class Solution(object):
	    def maxValueOfCoins(self, piles, k):
	        """
	        :type piles: List[List[int]]
	        :type k: int
	        :rtype: int
	        """
	        dp = [ [0]*2001 for _ in range(1001)]
	        preSum = collections.defaultdict(list)
	        N = len(piles)
	        for i in range(N):
	            total = 0
	            preSum[i].append(total)
	            for j in range(len(piles[i])):
	                total += piles[i][j]
	                preSum[i].append(total)
	
	        for i in range(N):
	            for j in range(k+1):
	                for t in range(min(j, len(piles[i])) + 1):
	                    current = (0 if i==0 else dp[i-1][j-t]) + preSum[i][t]
	                    dp[i][j] = max(dp[i][j], current)
	        return dp[N-1][k]
            	      
			
### 运行结果

	Runtime: 8252 ms, faster than 18.87% of Python online submissions for Maximum Value of K Coins From Piles.
	Memory Usage: 53.6 MB, less than 30.19% of Python online submissions for Maximum Value of K Coins From Piles.


### 原题链接


https://leetcode.com/contest/weekly-contest-286/problems/maximum-value-of-k-coins-from-piles/


您的支持是我最大的动力
