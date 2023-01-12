leetcode  2316. Count Unreachable Pairs of Nodes in an Undirected Graph（python）



### 描述


You are given an integer n. There is an undirected graph with n nodes, numbered from 0 to n - 1. You are given a 2D integer array edges where edges[i] = [ai, bi] denotes that there exists an undirected edge connecting nodes ai and bi. Return the number of pairs of different nodes that are unreachable from each other.


Example 1:

![](https://assets.leetcode.com/uploads/2022/05/05/tc-3.png)

	Input: n = 3, edges = [[0,1],[0,2],[1,2]]
	Output: 0
	Explanation: There are no pairs of nodes that are unreachable from each other. Therefore, we return 0.

	
Example 2:


![](https://assets.leetcode.com/uploads/2022/05/05/tc-2.png)

	Input: n = 7, edges = [[0,2],[0,5],[2,4],[1,6],[5,4]]
	Output: 14
	Explanation: There are 14 pairs of nodes that are unreachable from each other:
	[[0,1],[0,3],[0,6],[1,2],[1,3],[1,4],[1,5],[2,3],[2,6],[3,4],[3,5],[3,6],[4,6],[5,6]].
	Therefore, we return 14.




Note:

* 1 <= n <= 10^5
* 0 <= edges.length <= 2 * 10^5
* edges[i].length == 2
* 0 <= ai, bi < n
* ai != bi
* There are no repeated edges.


### 解析

根据题意，给定一个整数 n 。 有一个具有 n 个节点的无向图，编号从 0 到 n - 1 。给定一个 2D 整数数组边，其中 edges[i] = [ai, bi]  表示存在连接节点 ai 和 bi 的无向边。返回彼此不可达的不同节点对的数量。

读完这道题，我们就能发现，这道题考查的就是并查集，我们通过并查集中的 union 去将相连的节点放到一块，通过 find 取寻找相同的“父节点”，然后得到相连接点的组中的节点个数，然后在进行两两相乘的计算并求和就能找到所有不相连的节点对数量，最后记得除 2 即可。


强烈推荐看这位[大神的并查集讲解](https://www.bilibili.com/video/BV13t411v7Fs?p=1&vd_source=66ea1dd09047312f5bc02b99f5652ac6)，声音好听，知识点讲解非常容易懂。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答
				

	class Solution(object):
	    def __init__(self):
	        self.father = []
	        self.rank = []
	
	    def countPairs(self, n, edges):
	        for i in range(n):
	            self.father.append(i)
	            self.rank.append(1)
	
	        for x,y in edges:
	            self.union(x,y)
	        c = collections.Counter([self.find(i) for i in range(n)])
	        L = list(c.values())
	        result = 0
	        if not L:
	            return result
	        N = len(L)
	        for i in range(N):
	            result += L[i] * (n - L[i])
	        return result // 2
	
	    def find(self, x):
	        if self.father[x] != x:
	            self.father[x] = self.find(self.father[x])
	        return self.father[x]
	
	    def union(self, x, y):
	        x_root = self.find(x)
	        y_root = self.find(y)
	        if x_root != y_root:
	            if self.rank[x_root] > self.rank[y_root]:
	                self.father[y_root] = x_root
	            elif self.rank[y_root] > self.rank[x_root]:
	                self.father[x_root] = y_root
	            else:
	                self.father[y_root] = x_root
	                self.rank[x_root] += 1
            	      
			
### 运行结果


	64 / 64 test cases passed.
	Status: Accepted
	Runtime: 2145 ms
	Memory Usage: 82.4 MB



### 原题链接

https://leetcode.com/contest/biweekly-contest-81/problems/count-unreachable-pairs-of-nodes-in-an-undirected-graph/

您的支持是我最大的动力
