leetcode  797. All Paths From Source to Target（python）




### 描述


Given a directed acyclic graph (DAG) of n nodes labeled from 0 to n - 1, find all possible paths from node 0 to node n - 1 and return them in any order. The graph is given as follows: graph[i] is a list of all nodes you can visit from node i (i.e., there is a directed edge from node i to node graph[i][j]).


Example 1:

![](https://assets.leetcode.com/uploads/2020/09/28/all_1.jpg)

	Input: graph = [[1,2],[3],[3],[]]
	Output: [[0,1,3],[0,2,3]]
	Explanation: There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/09/28/all_2.jpg)

	Input: graph = [[4,3,1],[3,2,4],[3],[4],[]]
	Output: [[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]




Note:


* 	n == graph.length
* 	2 <= n <= 15
* 	0 <= graph[i][j] < n
* 	graph[i][j] != i (i.e., there will be no self-loops).
* 	All the elements of graph[i] are unique.
* 	The input graph is guaranteed to be a DAG.



### 解析

根据题意，给定一个由 n 个标记为从 0 到 n - 1 的节点的有向无环图 （DAG），找到从节点 0 到节点 n - 1 的所有可能路径，并按任意顺序返回它们。

其实这道题就是考察 DFS ，直接使用 DFS 从 0 开始到最后一个标签的节点，把所有可能的路线走一遍然后把路线存入结果 resut 中就好了。

时间复杂度为 O(N\*2^N) ，因为 DAG 中有 N 个节点，最坏情况下每一个节点都连接一个编号比它大的节点，此时路径数的量级为 O(2^N)，每条路径长度最长为 O(N)，空间复杂度为 O(N) ，主要是递归栈的开销。

### 解答

	class Solution(object):
	    def allPathsSourceTarget(self, graph):
	        """
	        :type graph: List[List[int]]
	        :rtype: List[List[int]]
	        """
	        result = []
	        N = len(graph)
	        def dfs(node, L):
	            if node == N-1:
	                result.append(L)
	                return
	            for n in graph[node]:
	                dfs(n, L+[n])
	        dfs(0, [0])
	        return result
	        

### 运行结果

	Runtime 63 ms，Beats 99.25%
	Memory 14.8 MB，Beats 45.48%

### 原题链接

	https://leetcode.com/problems/all-paths-from-source-to-target/description/

您的支持是我最大的动力
