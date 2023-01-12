leetcode 2240. Number of Ways to Buy Pens and Pencils （python）

这道题是第 76 场 leetcode 双周赛的第二题，难度为 Medium ，主要考察的是数学中的解方程问题


### 描述

You are given an integer total indicating the amount of money you have. You are also given two integers cost1 and cost2 indicating the price of a pen and pencil respectively. You can spend part or all of your money to buy multiple quantities (or none) of each kind of writing utensil.

Return the number of distinct ways you can buy some number of pens and pencils.



Example 1:

	Input: total = 20, cost1 = 10, cost2 = 5
	Output: 9
	Explanation: The price of a pen is 10 and the price of a pencil is 5.
	- If you buy 0 pens, you can buy 0, 1, 2, 3, or 4 pencils.
	- If you buy 1 pen, you can buy 0, 1, or 2 pencils.
	- If you buy 2 pens, you cannot buy any pencils.
	The total number of ways to buy pens and pencils is 5 + 3 + 1 = 9.

	
Example 2:

	Input: total = 5, cost1 = 10, cost2 = 10
	Output: 1
	Explanation: The price of both pens and pencils are 10, which cost more than total, so you cannot buy any writing utensils. Therefore, there is only 1 way: buy 0 pens and 0 pencils.




Note:



	1 <= total, cost1, cost2 <= 10^6
	
### 解析

根据题意，给定一个整数 total ，表示你有多少钱。 还给出了两个整数 cost1 和 cost2 ，分别表示钢笔和铅笔的价格。 问我们可以花费部分或全部资金购买多种（或不购买）每种书写用具。返回购买钢笔和铅笔的不同方案的数量。

这道题的题意很符合我们日常生活中购买东西的案例，很有意思，其实我觉得没什么考察的点，就是一个数学解方程的问题。

我们为了方便计算，找出较大的价格 mx_cost 和较小的价格 mn_cost ，然后我们可以得到最多可以买  mx_nums  = total // mx_cost 个较大价格的商品，然后我们遍历 mx_nums+1 次（因为可以买 0 个商品），第 i 次的遍历中我们可以计算在已知购买了 i 个较大价格商品后，再买较小价格商品的方案数量，将其加入结果 result 中，遍历结束返回 result 即可。

时间复杂度为 O(total // max(cost1, cost2)) ，空间复杂度为 O(1)。


### 解答
				

	class Solution(object):
	    def waysToBuyPensPencils(self, total, cost1, cost2):
	        """
	        :type total: int
	        :type cost1: int
	        :type cost2: int
	        :rtype: int
	        """
	        mn_cost = min(cost1,cost2)
	        mx_cost = max(cost1,cost2)
	        mx_nums = total // mx_cost
	        result = 0
	        for i in range(mx_nums+1):
	            result += 1 + (total - i * mx_cost) // mn_cost
	        return result
	            
            	      
			
### 运行结果

	
	

	145 / 145 test cases passed.
	Status: Accepted
	Runtime: 336 ms
	Memory Usage: 44.9 MB


### 原题链接



https://leetcode.com/contest/biweekly-contest-76/problems/number-of-ways-to-buy-pens-and-pencils/


您的支持是我最大的动力
