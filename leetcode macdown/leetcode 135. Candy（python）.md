leetcode  135. Candy（python）




### 描述

There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.

You are giving candies to these children subjected to the following requirements:

* Each child must have at least one candy.
* Children with a higher rating get more candies than their neighbors.

Return the minimum number of candies you need to have to distribute the candies to the children.

 



Example 1:


	Input: ratings = [1,0,2]
	Output: 5
	Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.
	
Example 2:

	Input: ratings = [1,2,2]
	Output: 4
	Explanation: You can allocate to the first, second and third child with 1, 2, 1 candies respectively.
	The third child gets 1 candy because it satisfies the above two conditions.





Note:


	n == ratings.length
	1 <= n <= 2 * 104
	0 <= ratings[i] <= 2 * 10^4

### 解析

根据题意，有 n 个孩子排成一排。 每个孩子都被分配了一个在整数数组 ratings 中给出的分数。向这些儿童提供糖果，但需满足以下要求：

* 每个孩子必须至少有一个糖果。
* 分数较高的孩子比他们的相邻的孩子得到更多的糖果。

返回将糖果分发给孩子所需的最少糖果数量。

这道题考查的是贪心思想，我们如果只是从左往右去根据规则分糖果，会造成顾前失后的问题，因为每个孩子都有相邻的两个孩子，要想符合规则，我们可以进行两个方向的贪心：

* 先从左往右去给每个孩子分配糖果，如果 ratings[i] > ratings[i - 1] 那么第 i 个孩子的糖果至少比第 i-1 个孩子糖果多一个，这样就能让每个孩子满足全局分数高的右边的孩子糖果比左边孩子低。
* 然后从右往左去给每个孩子分配糖果，如果  ratings[i] > ratings[i + 1] ，那么第 i 个孩子的糖果有两种选择，第一种是此时第 i+1 个孩子的糖果数加一，第二种是第 i 个孩子的糖果数（上面已经从左往右分了一次），取两者的最大值可以满足题意，即第 i 个孩子的糖果即大于左边也大于右边。
* 最后求和即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答
				

	class Solution(object):
	    def candy(self, ratings):
	        """
	        :type ratings: List[int]
	        :rtype: int
	        """
	        L = [1] * len(ratings)
	        for i in range(1, len(ratings)):
	            if ratings[i] > ratings[i - 1]:
	                L[i] = L[i - 1] + 1
	        for j in range(len(ratings) - 2, -1, -1):
	            if ratings[j] > ratings[j + 1]:
	                L[j] = max(L[j], L[j + 1] + 1)
	        return sum(L)
### 运行结果


	Runtime: 136 ms, faster than 83.42% of Python online submissions for Candy.
	Memory Usage: 15.3 MB, less than 84.46% of Python online submissions for Candy.

### 原题链接

https://leetcode.com/problems/candy/


您的支持是我最大的动力
