leetcode  2242. Maximum Score of a Node Sequence（python）

这道题是第 76 场 leetcode 双周赛的第四题，难度为 Hard ，主要考察的是贪心思想


### 描述

There is an undirected graph with n nodes, numbered from 0 to n - 1. You are given a 0-indexed integer array scores of length n where scores[i] denotes the score of node i. You are also given a 2D integer array edges where edges[i] = [ai, bi] denotes that there exists an undirected edge connecting nodes ai and bi.A node sequence is valid if it meets the following conditions:

* There is an edge connecting every pair of adjacent nodes in the sequence.
* No node appears more than once in the sequence.

The score of a node sequence is defined as the sum of the scores of the nodes in the sequence.Return the maximum score of a valid node sequence with a length of 4. If no such sequence exists, return -1.



Example 1:

![](https://assets.leetcode.com/uploads/2022/04/15/ex1new3.png)
	
	Input: scores = [5,2,9,8,4], edges = [[0,1],[1,2],[2,3],[0,2],[1,3],[2,4]]
	Output: 24
	Explanation: The figure above shows the graph and the chosen node sequence [0,1,2,3].
	The score of the node sequence is 5 + 2 + 9 + 8 = 24.
	It can be shown that no other node sequence has a score of more than 24.
	Note that the sequences [3,1,2,0] and [1,0,2,3] are also valid and have a score of 24.
	The sequence [0,3,2,4] is not valid since no edge connects nodes 0 and 3.

	
Example 2:


![](https://assets.leetcode.com/uploads/2022/03/17/ex2.png)

	Input: scores = [9,20,6,4,11,12], edges = [[0,3],[5,3],[2,4],[1,3]]
	Output: -1
	Explanation: The figure above shows the graph.
	There are no valid node sequences of length 4, so we return -1.




Note:


* 	n == scores.length
* 	4 <= n <= 5 * 10^4
* 	1 <= scores[i] <= 10^8
* 	0 <= edges.length <= 5 * 10^4
* 	edges[i].length == 2
* 	0 <= ai, bi <= n - 1
* 	ai != bi
* 	There are no duplicate edges.

### 解析

根据题意，有一个 n 个节点的无向图，索引编号从 0 到 n - 1 。给定一个长度为 n 的 0  索引整数数组 score ，其中 scores[i] 表示节点 i 的分数。还给出一个二维整数数组 edges ，其中 edges[i] = [ai, bi] 表示存在连接节点 ai 和 bi 的无向边。

在满足题意的情况下，我们要找出任意 4 个节点相连的序列，序列的得分定义为序列中所有节点的得分之和。返回长度为 4 的有效节点序列的最大分数。如果不存在这样的序列，则返回 -1 。

这道题我们可以把题意进行简化，无向图中所有的边我们已经知道了，我们可以将每个节点与其相连的节点都存入字典 graph 中。

因为我们已经知道了所有的边，所以我们可以得到每条边的两端 b 、c 的分数，然后我们只需要找 b 前的一个新的节点，c 后的一个新的节点，这样我们就有了一个四节点的序列 a + b + c + d 的分数和，遍历所有的边，进行最大值 result 的计算即可。

如果是按照上面的算法，本身没有错，但是可能会超时，因为一个节点可能与 n 个节点相连，但是根据我们简化之后的逻辑我们发现，在 [a,b,c,d] 中不管哪个节点想要成四节点序列，肯定要和三个不一样的节点有关系，而且我们又是需要最大分值的节点，所以我们对 graph 中的列表进行裁剪，每个节点只留下最多三个节点且是三个分数最大的节点。这样的时间复杂度刚好能通过。

排序的时间复杂度为 O(ElogE)，遍历的时间复杂度为 O( E\*9)，因为两个节点各自最多有 3 个相邻边，组合数有 9 种 ，所以时间复杂度总共为 O(ElogE) ，E 是边数。

空间复杂度为 O（N * 3），所以总共为 O(N)， N 是节点数。



### 解答
				

	class Solution(object):
	    def maximumScore(self, scores, edges):
	        """
	        :type scores: List[int]
	        :type edges: List[List[int]]
	        :rtype: int
	        """
	        graph = collections.defaultdict(list)
	        for a, b in edges:
	            graph[a].append([scores[b], b])
	            graph[b].append([scores[a], a])
	        for k in graph:
	            graph[k] = heapq.nlargest(3, graph[k])
	        result = 0
	        for b, c in edges:
	            for (a_score, a), (d_score, d) in product(graph[b], graph[c]):
	                if a not in [b, c] and d not in [b, c] and a != d:
	                    s = a_score + scores[b] + scores[c] + d_score
	                    result = max(result, s)
	        return -1 if result == 0 else result
            	      
			
### 运行结果



	63 / 63 test cases passed.
	Status: Accepted
	Runtime: 2048 ms
	Memory Usage: 46.6 MB

### 原题链接



https://leetcode.com/contest/biweekly-contest-76/problems/maximum-score-of-a-node-sequence/


您的支持是我最大的动力
