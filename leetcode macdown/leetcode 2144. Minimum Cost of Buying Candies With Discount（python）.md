leetcode  2144. Minimum Cost of Buying Candies With Discount（python）


### 前言

本来是在和老弟打游戏，到了晚上 10 点半突然闹钟响起来了，是我提醒自己第 70 场双周赛，结果着急地关掉游戏，有点对不起队友，等我整出电脑，都浪费了 5 分钟了，这是 Biweekly Contest 70 的第一题，难度 Easy ，感觉考察的是贪心和数组排序。


### 描述


A shop is selling candies at a discount. For every two candies sold, the shop gives a third candy for free.

The customer can choose any candy to take away for free as long as the cost of the chosen candy is less than or equal to the minimum cost of the two candies bought.

* For example, if there are 4 candies with costs 1, 2, 3, and 4, and the customer buys candies with costs 2 and 3, they can take the candy with cost 1 for free, but not the candy with cost 4.

Given a 0-indexed integer array cost, where cost[i] denotes the cost of the i<sub>th</sub> candy, return the minimum cost of buying all the candies.


Example 1:

	Input: cost = [1,2,3]
	Output: 5
	Explanation: We buy the candies with costs 2 and 3, and take the candy with cost 1 for free.
	The total cost of buying all candies is 2 + 3 = 5. This is the only way we can buy the candies.
	Note that we cannot buy candies with costs 1 and 3, and then take the candy with cost 2 for free.
	The cost of the free candy has to be less than or equal to the minimum cost of the purchased candies.

	
Example 2:

	Input: cost = [6,5,7,9,2,2]
	Output: 23
	Explanation: The way in which we can get the minimum cost is described below:
	- Buy candies with costs 9 and 7
	- Take the candy with cost 6 for free
	- We buy candies with costs 5 and 2
	- Take the last remaining candy with cost 2 for free
	Hence, the minimum cost to buy all candies is 9 + 7 + 5 + 2 = 23.


Example 3:


	Input: cost = [5,5]
	Output: 10
	Explanation: Since there are only 2 candies, we buy both of them. There is not a third candy we can take for free.
	Hence, the minimum cost to buy all candies is 5 + 5 = 10.
	




Note:

	1 <= cost.length <= 100
	1 <= cost[i] <= 100



### 解析

根据题意，一家商店在促销糖果，每售出两颗糖果，商店将免费赠送第三颗糖果。顾客可以选择任何糖果免费带走，但是要求所选糖果的价格小于或等于购买的两颗糖果的最低价格。

* 例如，如果有 4 个价格为 1、2、3 和 4 的糖果，客户购买价格为 2 和 3 的糖果，他们可以免费获得价格为 1 的糖果，但不能获得价格为 4 的糖果。

给定一个索引为 0 的整数数组 cost ，其中 cost[i] 表示第 i 个糖果的成本，返回购买所有糖果的最低成本。

这道题尽管考察的是排序，但是还透露出了一丝丝的贪心，因为我们想要最低的成本，那就尽量免费获得价格高的糖果，而想要获得价格高的糖果，就要买大于等于免费获得糖果的价格的糖果，所以最朴素的想法就是将 cost 的数组升序排序，我们只要从后向前的顺序购买糖果，每两次 cost.pop(-1) 买到糖果的花销计入 result ，然后再免费获得一个 cost.pop(-1) 的糖果，一直到 cost 为空，这样我们就能做到花费最少的钱买所有的糖果。

总的来说还是比较简单的，时间复杂度就是 O(n) ，空间复杂度 O(1) 。

其实当时我在做第一题的时候就明显感觉出来这次的双周赛题目不简单，后面的几道果然是验证了我的想法。


### 解答
				

	class Solution(object):
	    def minimumCost(self, cost):
	        """
	        :type cost: List[int]
	        :rtype: int
	        """
	        cost.sort()
	        result = 0
	        count = 0
	        while cost:
	            result += cost.pop(-1)
	            count += 1
	            if count == 2 and cost:
	                cost.pop(-1)
	                count = 0
	        return result

            	      
			
### 运行结果


	192 / 192 test cases passed.
	Status: Accepted
	Runtime: 28 ms
	Memory Usage: 13.6 MB


### 原题链接


https://leetcode.com/contest/biweekly-contest-70/problems/minimum-cost-of-buying-candies-with-discount/


您的支持是我最大的动力
