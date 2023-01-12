leetcode  2285. Maximum Total Importance of Roads（python）



### 描述

You are given an integer n denoting the number of cities in a country. The cities are numbered from 0 to n - 1.

You are also given a 2D integer array roads where roads[i] = [ai, bi] denotes that there exists a bidirectional road connecting cities ai and bi.

You need to assign each city with an integer value from 1 to n, where each value can only be used once. The importance of a road is then defined as the sum of the values of the two cities it connects.

Return the maximum total importance of all roads possible after assigning the values optimally.



Example 1:

![](https://assets.leetcode.com/uploads/2022/04/07/ex1drawio.png)

	Input: n = 5, roads = [[0,1],[1,2],[2,3],[0,2],[1,3],[2,4]]
	Output: 43
	Explanation: The figure above shows the country and the assigned values of [2,4,5,3,1].
	- The road (0,1) has an importance of 2 + 4 = 6.
	- The road (1,2) has an importance of 4 + 5 = 9.
	- The road (2,3) has an importance of 5 + 3 = 8.
	- The road (0,2) has an importance of 2 + 5 = 7.
	- The road (1,3) has an importance of 4 + 3 = 7.
	- The road (2,4) has an importance of 5 + 1 = 6.
	The total importance of all roads is 6 + 9 + 8 + 7 + 7 + 6 = 43.
	It can be shown that we cannot obtain a greater total importance than 43.

	
Example 2:


![](https://assets.leetcode.com/uploads/2022/04/07/ex2drawio.png)

	Input: n = 5, roads = [[0,3],[2,4],[1,3]]
	Output: 20
	Explanation: The figure above shows the country and the assigned values of [4,3,2,5,1].
	- The road (0,3) has an importance of 4 + 5 = 9.
	- The road (2,4) has an importance of 2 + 1 = 3.
	- The road (1,3) has an importance of 3 + 5 = 8.
	The total importance of all roads is 9 + 3 + 8 = 20.
	It can be shown that we cannot obtain a greater total importance than 20.




Note:

	2 <= n <= 5 * 10^4
	1 <= roads.length <= 5 * 10^4
	roads[i].length == 2
	0 <= ai, bi <= n - 1
	ai != bi
	There are no duplicate roads.


### 解析


根据题意，给定一个整数 n ，表示一个国家的城市数量。 城市的编号从 0 到 n - 1 。还给定二维整数数组 road ，其中 road[i] = [ai, bi] 表示存在一条连接城市 ai 和 bi 的双向道路。

题目需要我们为每个城市分配一个从 1 到 n 的整数值，其中每个值只能使用一次。 然后将道路的权重为它连接的两个城市的价值之和。返回所有道路可能的最大权重和。

这道题表面上看感觉是在考察图相关的问题，其实本质上就是个贪心+排序问题，要想使得最终的所有道路的权重和最大，根据朴素的贪心思想，那肯定是把较大的权重尽量赋予图中较高的度的城市节点，较高的度的城市节点自然就是拥有较多的连接道路的城市，经过这样排序的城市，然后依次从高到低给他们赋予权重即可，这样安排得到的所有的道路的权重和最大。

时间复杂度为 O(NlogN) ，空间复杂度为 O(N) 。


### 解答
				
	class Solution(object):
	    def maximumImportance(self, n, roads):
	        """
	        :type n: int
	        :type roads: List[List[int]]
	        :rtype: int
	        """
	        graph = collections.defaultdict(int)
	        for a,b in roads:
	            graph[a] += 1
	            graph[b] += 1
	        graph = sorted(graph.items(),  key=lambda x:x[1], reverse=True)
	        weight = collections.defaultdict(int)
	        for i,(node, num) in enumerate(graph):
	            weight[node] = n-i
	        result = 0
	        for a,b in roads:
	            result += weight[a] + weight[b]
	        return result
            	      
			
### 运行结果


	58 / 58 test cases passed.
	Status: Accepted
	Runtime: 1977 ms
	Memory Usage: 44.3 MB

### 原题链接

https://leetcode.com/contest/biweekly-contest-79/problems/maximum-total-importance-of-roads/

您的支持是我最大的动力
