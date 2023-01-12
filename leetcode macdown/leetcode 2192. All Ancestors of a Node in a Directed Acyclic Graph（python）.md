leetcode  2192. All Ancestors of a Node in a Directed Acyclic Graph（python）


### 前言

这是 Biweekly Contest 73 的第三题，主要考察的是对有向无环图的遍历和 BFS ，难度为  Medium 。

### 描述


You are given a positive integer n representing the number of nodes of a Directed Acyclic Graph (DAG). The nodes are numbered from 0 to n - 1 (inclusive).

You are also given a 2D integer array edges, where edges[i] = [from<sup>i</sup>, to<sup>i</sup>] denotes that there is a unidirectional edge from from<sup>i</sup> to to<sup>i</sup> in the graph.

Return a list answer, where answer[i] is the list of ancestors of the i<sup>th</sup> node, sorted in ascending order.

A node u is an ancestor of another node v if u can reach v via a set of edges.


Example 1:


![](https://assets.leetcode.com/uploads/2019/12/12/e1.png)

	Input: n = 8, edgeList = [[0,3],[0,4],[1,3],[2,4],[2,7],[3,5],[3,6],[3,7],[4,6]]
	Output: [[],[],[],[0,1],[0,2],[0,1,3],[0,1,2,3,4],[0,1,2,3]]
	Explanation:
	The above diagram represents the input graph.
	- Nodes 0, 1, and 2 do not have any ancestors.
	- Node 3 has two ancestors 0 and 1.
	- Node 4 has two ancestors 0 and 2.
	- Node 5 has three ancestors 0, 1, and 3.
	- Node 6 has five ancestors 0, 1, 2, 3, and 4.
	- Node 7 has four ancestors 0, 1, 2, and 3.




Note:

* 1 <= n <= 1000
* 0 <= edges.length <= min(2000, n * (n - 1) / 2)
* edges[i].length == 2
* 0 <= from<sup>i</sup>, to<sup>i</sup> <= n - 1
* from<sup>i</sup> != to<sup>i</sup>
* There are no duplicate edges.
* The graph is directed and acyclic.


### 解析

根据题意，给定一个正整数 n，表示有向无环图 (DAG) 的节点数。 节点编号从 0 到 n - 1 。另外给出了一个二维整数数组 edges，其中 edges[i] = [from<sup>i</sup>, to<sup>i</sup>] 表示图中存在从from<sup>i</sup> 到 to<sup>i</sup> 的单向边。

返回一个列表 answer ，其中 answer[i] 是第 i 个节点的祖先列表，将祖先列表按照升序排序。另外题目还给出了祖先的定义，如果 u 可以通过一条边到达 v，则节点 u 是另一个节点 v 的祖先。

这道题不难，首先我们通过上面的祖先定义，通过遍历 edges 中的元素来找出字典 d ，d 表示 {子节点:[初始祖先列表]} ，然后我们遍历 0 到 n-1 中的每个索引 i ，表示我们要找索引 i 节点的祖先。如果 i 不存在 d 中，说明没有祖先，所以直接在结果列表 result 中追加一个空列表即可，然后进行下一轮循环去找下一个节点的祖先。如果 i 在 d 中，我们就使用 BFS 来对其所能找到的祖先进行遍历，将所有的祖先都放进集合 ancestors 中，再对其进行升序排序后将其追加到 result 中，继续下一个节点的祖先查找。所有节点的祖先都找完之后返回 result 即可。

时间复杂度为 O(N^2) ，空间复杂度为 O(N^2)。

其实这道题还可以用 DFS 来进行解题，大家可以尝试一下，这里就不赘述了。
 

### 解答
				
	class Solution(object):
	    def getAncestors(self, n, edges):
	        """
	        :type n: int
	        :type edges: List[List[int]]
	        :rtype: List[List[int]]
	        """
	        d = {}
	        for a,b in edges:
	            d[b] = d.get(b, []) + [a]
	        result = []
	        for i in range(n):
	            if i not in d:
	                result.append([])
	                continue
	            ancestors = set(d[i])
	            stack = copy.deepcopy(d[i])
	            while stack:
	                node = stack.pop(0)
	                if node in d:
	                    for n in d[node]:
	                        if n not in ancestors:
	                            ancestors.add(n)
	                            stack.append(n)
	            result.append(sorted(ancestors))
	        return result
        
            	      
			
### 运行结果


	80 / 80 test cases passed.
	Status: Accepted
	Runtime: 1827 ms
	Memory Usage: 28.6 MB

### 原题链接



https://leetcode.com/contest/biweekly-contest-73/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/


您的支持是我最大的动力
