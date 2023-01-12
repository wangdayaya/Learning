leetcode  322. Coin Change（python）




### 描述

You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.



Example 1:

	Input: coins = [1,2,5], amount = 11
	Output: 3
	Explanation: 11 = 5 + 5 + 1


	
Example 2:

	Input: coins = [2], amount = 3
	Output: -1	


Example 3:

	Input: coins = [1], amount = 0
	Output: 0

	

Note:

	1 <= coins.length <= 12
	1 <= coins[i] <= 2^31 - 1
	0 <= amount <= 10^4



### 解析


根据题意，给定一个整数数组 coins ，表示不同面额的硬币种类，又给定一个整数 amount 表示总金额。返回弥补的最少硬币数量以达到 amount 。 如果 amount 不能由任何硬币组合组成，则返回 -1 。可以假设拥有无限数量的各种硬币。

像这种用最少的硬币组成 amount 的题目，一眼就认出这是考察动态规划，解题方法要用到 DFS + 备忘录的套路（其实是因为这几天的每日一题都是在考察这类解法），我们就按照最朴素的逻辑，假如像例子一中要求 amount 为 11 ，那么我们只要求出 amount 为 10 的最少硬币个数加 1 ，或者求出 amount 为 9 的最少硬币个数加 1 ，我们可以看出这些子问题之间是互相独立的，我们定义函数 dp(n) 表示要凑出总金额为 n 需要的最少的硬币个数，动态方程如下：

	0,   n=0
	-1,   n<0
	min(dp(n-c)+1),   c 为不通的面额硬币

如果可以拼凑出 amount 肯定是最少数量的硬币，如果不能那就说明无法拼凑出来直接返回 -1 即可。这里为了实现记忆化还需要用到 python 注解来记住出现过的 dp(n) 结果，否则会超时。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答
				

	class Solution:
	    def coinChange(self, coins: List[int], amount: int) -> int:
	        @functools.lru_cache(None)
	        def dp(n):
	            if n == 0: return 0
	            if n < 0: return -1
	            result = float('INF')
	            for c in coins:
	                r = dp(n - c)
	                if r == -1: continue
	                result = min(result, 1 + r)
	            return result if result!=float('INF') else -1
	
	        return dp(amount)
            	      
			
### 运行结果

	Runtime: 2049 ms, faster than 43.78% of Python3 online submissions for Coin Change.
	Memory Usage: 35.1 MB, less than 10.50% of Python3 online submissions for Coin Change.


### 原题链接



https://leetcode.com/problems/coin-change/

您的支持是我最大的动力
