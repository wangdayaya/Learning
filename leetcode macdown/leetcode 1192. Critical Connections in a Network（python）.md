leetcode  1192. Critical Connections in a Network（python）




### 描述

There are n servers numbered from 0 to n - 1 connected by undirected server-to-server connections forming a network where connections[i] = [ai, bi] represents a connection between servers ai and bi. Any server can reach other servers directly or indirectly through the network.

A critical connection is a connection that, if removed, will make some servers unable to reach some other server.

Return all critical connections in the network in any order.





Example 1:


![](https://assets.leetcode.com/uploads/2019/09/03/1537_ex1_2.png)

	Input: n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]
	Output: [[1,3]]
	Explanation: [[3,1]] is also accepted.
	
Example 2:

	Input: n = 2, connections = [[0,1]]
	Output: [[0,1]]






Note:


	2 <= n <= 105
	n - 1 <= connections.length <= 10^5
	0 <= ai, bi <= n - 1
	ai != bi
	There are no repeated connections.

### 解析

根据题意，有 n 个服务器，编号从 0 到 n - 1，通过定义好的 connections 进行服务器到服务器双向连接形成一个网络，其中 connections[i] = [ai, bi] 表示服务器 ai 和 bi 之间的连接。 任何服务器都可以通过网络直接或间接访问其他服务器。

如果删除一个关键连接会使某些服务器无法访问其他服务器。以任意顺序返回网络中的所有关键连接。

其实题目中的关键链接就是图中的边，删除了某个关键的边，会导致图中所有的节点不能全部连通，这个边就是关键边或者关键连接。其实这是图中的经典问题，如果如何求一个无向图中的割边，我推荐这个介绍相关割边和算法的[博客](https://www.cnblogs.com/nullzx/p/7968110.html)，有不懂的可以看一下。

时间复杂度为 O(E)，空间复杂度为 O(E) 。



### 解答
				
	class Solution(object):
	    def criticalConnections(self, n, connections):
	        """
	        :type n: int
	        :type connections: List[List[int]]
	        :rtype: List[List[int]]
	        """
	        graph = defaultdict(list)
	        for v in connections:
	            graph[v[0]].append(v[1])
	            graph[v[1]].append(v[0])
	            
	        dfn = [None for i in range(n)]
	        low = [None for i in range(n)]
	        
	        cur = 0
	        start = 0
	        res = []
	        self.cur = 0
	       
	        def dfs(node,parent):
	            if dfn[node] is None:
	                dfn[node] = self.cur
	                low[node] = self.cur
	                self.cur+=1
	                for n in graph[node]:
	                    if dfn[n] is None:
	                        dfs(n,node)
	                    
	                if parent is not None:
	                    l = min([low[i] for i in graph[node] if i!=parent]+[low[node]])
	                else:
	                    l = min(low[i] for i in graph[node]+[low[node]])
	                low[node] = l
	                
	        dfs(0,None)
	        
	        for v in connections:
	            if low[v[0]]>dfn[v[1]] or low[v[1]]>dfn[v[0]]:
	                res.append(v)
	        return res

            	      
			
### 运行结果

	Runtime: 3010 ms, faster than 44.00% of Python online submissions for Critical Connections in a Network.
	Memory Usage: 74.7 MB, less than 84.00% of Python online submissions for Critical Connections in a Network.




### 原题链接


https://leetcode.com/problems/critical-connections-in-a-network/


您的支持是我最大的动力
