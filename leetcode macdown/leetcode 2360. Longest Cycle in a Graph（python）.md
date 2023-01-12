leetcode 2360. Longest Cycle in a Graph （python）




### 描述


You are given a directed graph of n nodes numbered from 0 to n - 1, where each node has at most one outgoing edge.

The graph is represented with a given 0-indexed array edges of size n, indicating that there is a directed edge from node i to node edges[i]. If there is no outgoing edge from node i, then edges[i] == -1.

Return the length of the longest cycle in the graph. If no cycle exists, return -1.

A cycle is a path that starts and ends at the same node.


Example 1:

![](https://assets.leetcode.com/uploads/2022/06/08/graph4drawio-5.png)

	Input: edges = [3,3,4,2,3]
	Output: 3
	Explanation: The longest cycle in the graph is the cycle: 2 -> 4 -> 3 -> 2.
	The length of this cycle is 3, so 3 is returned.

	
Example 2:

![](https://assets.leetcode.com/uploads/2022/06/07/graph4drawio-1.png)

	Input: edges = [2,-1,3,1]
	Output: -1
	Explanation: There are no cycles in this graph.








Note:

	n == edges.length
	2 <= n <= 10^5
	-1 <= edges[i] < n
	edges[i] != i



### 解析

根据题意，给定一个从 0 到 n - 1 编号的 n 个节点的有向图，其中每个节点最多有一个出边。 该图用给定的大小为 n 的 0 索引数组 edges 来表示，表示从节点 i 到节点 edges[i] 有一条有向边。 如果没有出边，则 edges[i] == -1。该图用给定的大小为 n 的 0 索引数组 edges 表示，表示从节点 i 到节点 edges[i] 有一条有向边。 如果没有出边，则 edges[i] == -1。

返回图中最长环的长度。 如果不存在循环，则返回-1。环是在同一节点开始和结束的路径。

我们先用字典 d 找出路每个节点的下一个节点，定义 memo 来记录已经遍历过的节点，然后使用 BFS 来依次从 0 节点开始找可能存在的环的长度，并不断更新最后的结果 result ，遍历完所有的节点之后返回 result 即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答

	class Solution(object):
	    def longestCycle(self, edges):
	        """
	        :type edges: List[int]
	        :rtype: int
	        """
	        d = collections.defaultdict(list)
	        for i, val in enumerate(edges):
	            if val != -1:
	                d[i].append(val)
	
	        memo = dict()
	        result = -1
	
	        for i in range(len(edges)):
	            if i in memo:
	                continue
	            memo[i] = (i, 0)  #  destination: (starting, distance)
	            stack = [(i, 0)]
	            while stack:
	                for _ in range(len(stack)):
	                    curr, dis = stack.pop(0)
	                    for nxt in d[curr]:
	                        if nxt in memo:
	                            if memo[nxt][0] == i:
	                                result = max(result, dis - memo[nxt][1] + 1)
	                        else:
	                            memo[nxt] = (i, dis + 1)
	                            stack.append((nxt, dis + 1))
	        return result

### 运行结果


	72 / 72 test cases passed.
	Status: Accepted
	Runtime: 2761 ms
	Memory Usage: 63 MB

### 原题链接

https://assets.leetcode.com/uploads/2022/06/08/graph4drawio-5.png


您的支持是我最大的动力
