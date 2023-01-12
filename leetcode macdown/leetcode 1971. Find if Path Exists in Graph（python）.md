leetcode  1971. Find if Path Exists in Graph（python）

### 描述

There is a bi-directional graph with n vertices, where each vertex is labeled from 0 to n - 1 (inclusive). The edges in the graph are represented as a 2D integer array edges, where each edges[i] = [u<sub>i</sub>, v<sub>i</sub>] denotes a bi-directional edge between vertex u<sub>i</sub> and vertex v<sub>i</sub>. Every vertex pair is connected by at most one edge, and no vertex has an edge to itself.

You want to determine if there is a valid path that exists from vertex start to vertex end.

Given edges and the integers n, start, and end, return true if there is a valid path from start to end, or false otherwise.



Example 1:

![](https://assets.leetcode.com/uploads/2021/08/14/validpath-ex1.png)

	Input: n = 3, edges = [[0,1],[1,2],[2,0]], start = 0, end = 2
	Output: true
	Explanation: There are two paths from vertex 0 to vertex 2:
	- 0 → 1 → 2
	- 0 → 2


	
Example 2:

![](https://assets.leetcode.com/uploads/2021/08/14/validpath-ex2.png)

	Input: n = 6, edges = [[0,1],[0,2],[3,5],[5,4],[4,3]], start = 0, end = 5
	Output: false
	Explanation: There is no path from vertex 0 to vertex 5.


Note:

* 	1 <= n <= 2 * 10^5
* 	0 <= edges.length <= 2 * 10^5
* 	edges[i].length == 2
* 	0 <= u<sub>i</sub>, v<sub>i</sub> <= n - 1
* 	u<sub>i</sub> != v<sub>i</sub>
* 	0 <= start, end <= n - 1
* 	There are no duplicate edges.
* 	There are no self edges.



### 解析


根据题意，给出一个具有 n 个顶点的双向图，其中每个顶点标记为 0 到 n-1 中的某个数字。 图中的边表示为二维整数数组边，其中每个 edges[i] = [u<sub>i</sub>, v<sub>i</sub>] 表示顶点 u<sub>i</sub>  和顶点 v<sub>i</sub> 之间的双向边。 每个顶点对最多由一条边连接，并且没有顶点与自身相连的边。题目要求确定是否存在从顶点 start 到顶点 end 的有效路径。给定边和整数 n、start 和 end ，如果从开始到结束存在有效路径，则返回 true，否则返回 false。

直接使用 BFS 思路：

* 如果 start 等于 end  直接返回 True （我没想到这个边界情况，导致提交错了一次！！）
* 将所有的 edges 中的顶点情况都存在一个二维列表  graph ，graph 长度为 n 
* 初始化一个长度为 n 的全为 False 的 visited 列表，表示某个顶点是否被访问过，如果被访问过则变为 True
* 初始化一个队列 stack ，存放没有访问过的且可能是 end 的顶点，加入一个顶点 start
* 当 stack 不为空的时候 while 循环，取出 stack 的第一个元素为 key ，将 visited[key] 置为 True 表示被访问过，如果 end 在 graph[key] 中出现过直接返回 True ，否则将 graph[key] 中的没有访问过的元素加入到 stack 中，继续循环
* 循环结束没有找到合法路径，直接返回 False


### 解答
				

	class Solution(object):
	    def validPath(self, n, edges, start, end):
	        """
	        :type n: int
	        :type edges: List[List[int]]
	        :type start: int
	        :type end: int
	        :rtype: bool
	        """
	        if start == end: return True
	        graph = [[] for i in range(n)]
	        for s,e in edges:
	            graph[s].append(e)
	            graph[e].append(s)
	        visited = [False for _ in range(n)]
	        stack = [start]
	        while stack:
	            key = stack.pop(0)
	            visited[key] = True
	            if end in graph[key]:
	                return True
	            for vertex in graph[key]:
	                if not visited[vertex]:
	                    stack.append(vertex)
	        return False
            	      
			
### 运行结果

	Runtime: 1612 ms, faster than 97.22% of Python online submissions for Find if Path Exists in Graph.
	Memory Usage: 110.9 MB, less than 89.03% of Python online submissions for Find if Path Exists in Graph.


原题链接：https://leetcode.com/problems/find-if-path-exists-in-graph/



您的支持是我最大的动力
