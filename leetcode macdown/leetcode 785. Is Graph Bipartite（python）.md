leetcode  785. Is Graph Bipartite?（python）




### 描述


There is an undirected graph with n nodes, where each node is numbered between 0 and n - 1. You are given a 2D array graph, where graph[u] is an array of nodes that node u is adjacent to. More formally, for each v in graph[u], there is an undirected edge between node u and node v. The graph has the following properties:

* There are no self-edges (graph[u] does not contain u).
* There are no parallel edges (graph[u] does not contain duplicate values).
* If v is in graph[u], then u is in graph[v] (the graph is undirected).
* The graph may not be connected, meaning there may be two nodes u and v such that there is no path between them.

A graph is bipartite if the nodes can be partitioned into two independent sets A and B such that every edge in the graph connects a node in set A and a node in set B.

Return true if and only if it is bipartite.


Example 1:

![](https://assets.leetcode.com/uploads/2020/10/21/bi2.jpg)

	Input: graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
	Output: false
	Explanation: There is no way to partition the nodes into two independent sets such that every edge connects a node in one and a node in the other.

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/10/21/bi1.jpg)

	Input: graph = [[1,3],[0,2],[1,3],[0,2]]
	Output: true
	Explanation: We can partition the nodes into two sets: {0, 2} and {1, 3}.






Note:

	graph.length == n
	1 <= n <= 100
	0 <= graph[u].length < n
	0 <= graph[u][i] <= n - 1
	graph[u] does not contain u.
	All the values of graph[u] are unique.
	If graph[u] contains v, then graph[v] contains u.


### 解析


根据题意，有一个具有 n 个节点的无向图，其中每个节点的编号在 0 和 n - 1 之间。给定一个二维数组图，其中 graph[u] 是节点 u 相邻的节点数组。 也就是说，对于 graph[u] 中的每个 v ，在节点 u 和节点 v 之间都有一条无向边。该图具有以下属性：

* 没有自连边（也就是 graph[u] 不包含 u）。
* 没有平行边（graph[u] 不包含重复值）。
* 如果 v 在 graph[u] 中，那么 u 在 graph[v] 中（图是无向的）。
* 该图可能有节点未连接，如可能有两个节点 u 和 v，因此它们之间没有路径。

如果节点可以划分为两个独立的集合 A 和 B，则图是二分图，使得图中的每条边都连接集合 A 中的节点和集合 B 中的节点。 当且仅当它是二分图的时候返回 True 。

### 解答
				

	class Solution(object):
	    def isBipartite(self, graph):
	        """
	        :type graph: List[List[int]]
	        :rtype: bool
	        """
	        n, colored = len(graph), {}
	        for i in range(n):
	            if i not in colored and graph[i]:
	                colored[i] = 1
	                q = collections.deque([i])
	                while q:
	                    cur = q.popleft()
	                    for nb in graph[cur]:
	                        if nb not in colored:
	                            colored[nb] = -colored[cur]
	                            q.append(nb)
	                        elif colored[nb] == colored[cur]:
	                            return False
	        return True
        
            	      
			
### 运行结果


	Runtime: 169 ms, faster than 59.63% of Python online submissions for Is Graph Bipartite?.
	Memory Usage: 13.8 MB, less than 74.25% of Python online submissions for Is Graph Bipartite?.

### 原题链接



https://leetcode.com/problems/is-graph-bipartite/


您的支持是我最大的动力
