leetcode  2359. Find Closest Node to Given Two Nodes（python）




### 描述

You are given a directed graph of n nodes numbered from 0 to n - 1, where each node has at most one outgoing edge. The graph is represented with a given 0-indexed array edges of size n, indicating that there is a directed edge from node i to node edges[i]. If there is no outgoing edge from i, then edges[i] == -1.

You are also given two integers node1 and node2. Return the index of the node that can be reached from both node1 and node2, such that the maximum between the distance from node1 to that node, and from node2 to that node is minimized. If there are multiple answers, return the node with the smallest index, and if no possible answer exists, return -1.

Note that edges may contain cycles.



Example 1:

![](https://assets.leetcode.com/uploads/2022/06/07/graph4drawio-2.png)

	Input: edges = [2,2,3,-1], node1 = 0, node2 = 1
	Output: 2
	Explanation: The distance from node 0 to node 2 is 1, and the distance from node 1 to node 2 is 1.
	The maximum of those two distances is 1. It can be proven that we cannot get a node with a smaller maximum distance than 1, so we return node 2.

	
Example 2:

![](https://assets.leetcode.com/uploads/2022/06/07/graph4drawio-4.png)

	Input: edges = [1,2,-1], node1 = 0, node2 = 2
	Output: 2
	Explanation: The distance from node 0 to node 2 is 2, and the distance from node 2 to itself is 0.
	The maximum of those two distances is 2. It can be proven that we cannot get a node with a smaller maximum distance than 2, so we return node 2.





Note:

	n == edges.length
	2 <= n <= 10^5
	-1 <= edges[i] < n
	edges[i] != i
	0 <= node1, node2 < n


### 解析

根据题意，给定一个从 0 到 n - 1 编号的 n 个节点的有向图，其中每个节点最多有一个出边。 该图用给定的大小为 n 的 0 索引数组 edges 表示，表示从节点 i 到节点 edges[i] 有一条有向边。 如果没有出边，则 edges[i] == -1。

还给定两个整数 node1 和 node2。 返回从 node1 和 node2 都可以到达的节点的索引，使得从 node1 到该节点的距离与从 node2 到该节点的距离之间的最大值最小化。 如果有多个答案，则返回索引最小的节点，如果不存在可能的答案，则返回 -1 。需要注意的是，边可能包含循环。

我们定义 BFS 函数，分别找出 node1 和 node2 到可达节点的距离列表 L1 和 L2 ，如果不存在可达距离则为 inf ，这两个列表长度一样，所以直接遍历对应的元素，找出最小的最大值的索引，这个索引就是题目中要找的节点索引。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答

	class Solution:
	    def closestMeetingNode(self, edges: List[int], node1: int, node2: int) -> int:
	        def bfs(node):
	            stack = [node]
	            L = [inf] * N
	            L[node] = 0
	            visited = set()
	            visited.add(node)
	            while stack:
	                for i in range(len(stack)):
	                    cur = stack.pop(0)
	                    nxt = edges[cur]
	                    if nxt == -1 or nxt in visited:
	                        break
	                    else:
	                        L[nxt] = L[cur] + 1
	                        stack.append(nxt)
	                        visited.add(nxt)
	            return L
	
	        N = len(edges)
	        L1 = bfs(node1)
	        L2 = bfs(node2)
	        result = -1
	        tmp = inf
	        for i in range(N):
	            if max(L1[i], L2[i]) < tmp:
	                tmp = max(L1[i], L2[i])
	                result = i
	        return result


### 运行结果

	Runtime: 1389 ms, faster than 100.00% of Python3 online submissions for Find Closest Node to Given Two Nodes.
	Memory Usage: 34.1 MB, less than 20.00% of Python3 online submissions for Find Closest Node to Given Two Nodes.

### 原题链接

	https://leetcode.com/contest/weekly-contest-304/problems/find-closest-node-to-given-two-nodes/


您的支持是我最大的动力
