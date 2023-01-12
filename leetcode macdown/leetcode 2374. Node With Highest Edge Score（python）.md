leetcode  2374. Node With Highest Edge Score（python）




### 描述

You are given a directed graph with n nodes labeled from 0 to n - 1, where each node has exactly one outgoing edge. The graph is represented by a given 0-indexed integer array edges of length n, where edges[i] indicates that there is a directed edge from node i to node edges[i]. The edge score of a node i is defined as the sum of the labels of all the nodes that have an edge pointing to i.

Return the node with the highest edge score. If multiple nodes have the same edge score, return the node with the smallest index.



Example 1:


![](https://assets.leetcode.com/uploads/2022/06/20/image-20220620195403-1.png)

	Input: edges = [1,0,0,0,0,7,7,5]
	Output: 7
	Explanation:
	- The nodes 1, 2, 3 and 4 have an edge pointing to node 0. The edge score of node 0 is 1 + 2 + 3 + 4 = 10.
	- The node 0 has an edge pointing to node 1. The edge score of node 1 is 0.
	- The node 7 has an edge pointing to node 5. The edge score of node 5 is 7.
	- The nodes 5 and 6 have an edge pointing to node 7. The edge score of node 7 is 5 + 6 = 11.
	Node 7 has the highest edge score so return 7.
	
Example 2:

![](https://assets.leetcode.com/uploads/2022/06/20/image-20220620200212-3.png)

	Input: edges = [2,0,0,2]
	Output: 0
	Explanation:
	- The nodes 1 and 2 have an edge pointing to node 0. The edge score of node 0 is 1 + 2 = 3.
	- The nodes 0 and 3 have an edge pointing to node 2. The edge score of node 2 is 0 + 3 = 3.
	Nodes 0 and 2 both have an edge score of 3. Since node 0 has a smaller index, we return 0.




Note:

* n == edges.length
* 2 <= n <= 10^5
* 0 <= edges[i] < n
* edges[i] != i


### 解析
根据题意，给定一个有向图，其中 n 个节点标记为从 0 到 n - 1，其中每个节点恰好有一个出边。 该图由长度为 n 的给定 0 索引整数数组 edges 表示，其中 edges[i] 表示从节点 i 到节点 edges[i] 存在有向边。 节点 i 的边缘得分定义为所有具有指向 i 的边的节点的标签的总和。返回具有最高边缘分数的节点。 如果多个节点具有相同的边缘分数，则返回索引最小的节点。

其实这道题看起来是个图，但是真的和图算法一点关系都没有，我们只需根据 edges 将每个节点周围指向它的其他周围节点找到并计算标签和，然后根据此标签和进行排序即可，取和最大并且节点标签最小的节点标签返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答
	class Solution(object):
	    def edgeScore(self, edges):
	        """
	        :type edges: List[int]
	        :rtype: int
	        """
	        d = collections.defaultdict(int)
	        for s,t in enumerate(edges):
	            d[t] += s
	        mx = max(d.values())
	        for i in range(len(edges)):
	            if d[i] == mx:
	                return i
	

### 运行结果

	
	118 / 118 test cases passed.
	Status: Accepted
	Runtime: 1643 ms
	Memory Usage: 33.9 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-306/problems/node-with-highest-edge-score/

您的支持是我最大的动力
