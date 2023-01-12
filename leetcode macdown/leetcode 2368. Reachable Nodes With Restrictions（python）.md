leetcode  2368. Reachable Nodes With Restrictions（python）




### 描述

There is an undirected tree with n nodes labeled from 0 to n - 1 and n - 1 edges. You are given a 2D integer array edges of length n - 1 where edges[i] = [a<sub>i</sub>, b<sub>i</sub>] indicates that there is an edge between nodes a<sub>i</sub> and b<sub>i</sub> in the tree. You are also given an integer array restricted which represents restricted nodes.

Return the maximum number of nodes you can reach from node 0 without visiting a restricted node. Note that node 0 will not be a restricted node.



Example 1:

![](https://assets.leetcode.com/uploads/2022/06/15/ex1drawio.png)

	Input: n = 7, edges = [[0,1],[1,2],[3,1],[4,0],[0,5],[5,6]], restricted = [4,5]
	Output: 4
	Explanation: The diagram above shows the tree.
	We have that [0,1,2,3] are the only nodes that can be reached from node 0 without visiting a restricted node.

	
Example 2:

![](https://assets.leetcode.com/uploads/2022/06/15/ex2drawio.png)


	Input: n = 7, edges = [[0,1],[0,2],[0,5],[0,4],[3,2],[6,5]], restricted = [4,2,1]
	Output: 3
	Explanation: The diagram above shows the tree.
	We have that [0,5,6] are the only nodes that can be reached from node 0 without visiting a restricted node.



Note:


* 2 <= n <= 10^5
* edges.length == n - 1
* edges[i].length == 2
* 0 <= a<sub>i</sub>, b<sub>i</sub> < n
* a<sub>i</sub>!= b<sub>i</sub>
* edges represents a valid tree.
* 1 <= restricted.length < n
* 1 <= restricted[i] < n
* All the values of restricted are unique.

### 解析

根据题意，有一棵无向树，其 n 个节点标记为从 0 到 n - 1 和并且有 n - 1 条边。 给定一个长度为 n - 1 的二维整数数组 edges ，其中 edges[i] = [a<sub>i</sub>, b<sub>i</sub>] 表示在树中有一条节点 a< sub>i</sub> 到 b<sub>i</sub> 的边。 给定一个整数数组 restricted ，它表示受限节点。返回可以从节点 0 开始可以到达的所有节点数，而无需访问受限节点。 请注意，节点 0 不受限节。

其实这道题考察的就是 BFS ，我们首先要遍历所有的 edges ，将节点之间的边保存到字典 d 中，然后我们定义一个集合来保存可以访问的节点集合 result ，然后我们使用 BFS 的思路从节点 0 开始不断去访问非 restricted 的节点，并不断保存入 result 中，遍历结束我们返回 result 得长度即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def reachableNodes(self, n, edges, restricted):
	        """
	        :type n: int
	        :type edges: List[List[int]]
	        :type restricted: List[int]
	        :rtype: int
	        """
	        d = collections.defaultdict(list)
	        for s,t in edges:
	            d[s].append(t)
	            d[t].append(s)
	        restricted = set(restricted)
	        result = set()
	        result.add(0)
	        stack = [0]
	        while stack:
	            cur = stack.pop(0)
	            for sub in d[cur]:
	                if sub not in result and sub not in restricted:
	                    result.add(sub)
	                    stack.append(sub)
	        return len(result)

### 运行结果

	
	62 / 62 test cases passed.
	Status: Accepted
	Runtime: 1518 ms
	Memory Usage: 74.3 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-305/problems/reachable-nodes-with-restrictions/

您的支持是我最大的动力
