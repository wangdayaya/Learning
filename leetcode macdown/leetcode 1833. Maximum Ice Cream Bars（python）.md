leetcode  1833. Maximum Ice Cream Bars（python）




### 描述

It is a sweltering summer day, and a boy wants to buy some ice cream bars. At the store, there are n ice cream bars. You are given an array costs of length n, where costs[i] is the price of the i<sup>th</sup> ice cream bar in coins. The boy initially has coins coins to spend, and he wants to buy as many ice cream bars as possible.  Return the maximum number of ice cream bars the boy can buy with coins coins. Note: The boy can buy the ice cream bars in any order.



Example 1:

	Input: costs = [1,3,2,4,1], coins = 7
	Output: 4
	Explanation: The boy can buy ice cream bars at indices 0,1,2,4 for a total price of 1 + 3 + 2 + 1 = 7.

	
Example 2:

	Input: costs = [10,6,8,7,7,8], coins = 5
	Output: 0
	Explanation: The boy cannot afford any of the ice cream bars.


Example 3:

	Input: costs = [1,6,3,1,2,5], coins = 20
	Output: 6
	Explanation: The boy can buy all the ice cream bars for a total price of 1 + 6 + 3 + 1 + 2 + 5 = 18.



Note:


* costs.length == n
* 1 <= n <= 10^5
* 1 <= costs[i] <= 10^5
* 1 <= coins <= 10^8

### 解析

根据题意，在一个闷热的夏日，一个男孩想买一些冰淇淋棒。在商店里有 n 个冰淇淋棒。给定一个长度为 n 的数组 costs ，其中 cost[i] 是第 i 个冰淇淋棒的价格。男孩有硬币可以花，他想买尽可能多的冰淇淋棒。 计算男孩可以用硬币购买冰淇淋棒的最大数量。注意：男孩可以按任何顺序购买冰淇淋棒。

这道题虽然是中等难度，但是实际上看完题目我们可以发现是一道简单题目，我们只需要使用贪心思想解题即可。

想要冰淇淋数目最大化，我们可以从价钱最便宜的冰淇淋开始买起，现将 costs 按照升序排列，然后我们不断遍历每个冰淇淋的价钱 c ，只要男孩手里的钱大于等于当前冰淇凌的价钱，男孩就可以花钱买这个冰淇凌，我们就可以将计数器 result 加一，当男孩钱不够的时候，只需要返回计数器 result 即可。

时间复杂度为 O(NlogN + N) ，N 是 costs 的长度，这里耗时主要在排序和遍历，空间复杂度为 O(1) ，没有使用额外的空间。

### 解答

	class Solution:
	    def maxIceCream(self, costs: List[int], coins: int) -> int:
	        costs.sort()
	        result = 0
	        for c in costs:
	            if coins >= c:
	                result += 1
	                coins -= c
	            else:
	                break
	        return result

### 运行结果

	Runtime 828 ms ，Beats 96.30%
	Memory 27.9 MB ，Beats 90.96%

### 原题链接

https://leetcode.com/problems/maximum-ice-cream-bars/description/


您的支持是我最大的动力
