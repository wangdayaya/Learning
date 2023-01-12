leetcode 743. Network Delay Time （python）




### 描述

You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.

We will send a signal from a given node k. Return the time it takes for all the n nodes to receive the signal. If it is impossible for all the n nodes to receive the signal, return -1.





Example 1:

![](https://assets.leetcode.com/uploads/2019/05/23/931_example_1.png)

	Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
	Output: 2

	
Example 2:

	Input: times = [[1,2,1]], n = 2, k = 1
	Output: 1


Example 3:

	Input: times = [[1,2,1]], n = 2, k = 2
	Output: -1

	



Note:

	1 <= k <= n <= 100
	1 <= times.length <= 6000
	times[i].length == 3
	1 <= ui, vi <= n
	ui != vi
	0 <= wi <= 100
	All the pairs (ui, vi) are unique. (i.e., no multiple edges.)


### 解析

根据题意，给定一个由 n 个节点组成的网络，从 1 到 n 标记。 还给定了有向图 times ， times[i] = (ui, vi, wi) 的旅行时间列表，其中 ui 是源节点，vi 是目标节点，wi 是信号所需的时间。

需要将从给定节点 k 发送一个信号。 返回所有 n 个节点接收到信号所需的时间。 如果不可能所有的 n 个节点都接收到信号，则返回 -1 。


这道题是一道很经典的考察 Dijkstra 算法的模版题，如果这道题会做，类似的应该都能解决，其实 Dijkstra 算法本质上就是 BFS + 优先队列的问题，因为正常的 BFS 其实就是权重为 1 的遍历过程，而图中的权重明显不一定是 1 ，所以需要加入优先队列中不断对从源点到目标节点的最短距离进行排序，具体的过程我都写到代码注释里面了，应该很好理解，如果有不懂的，可以参考这位[大佬的讲解](https://www.bilibili.com/video/BV12f4y1z7PM?spm_id_from=333.999.0.0)，非常通俗易懂。

时间复杂度为 O(ElogE) ，空间复杂度为 O(E) ，E 是边的数量。

### 解答
				

	class Solution(object):
	    def networkDelayTime(self, times, N, K):
	        """
	        :type times: List[List[int]]
	        :type n: int
	        :type k: int
	        :rtype: int
	        """
	        
	        graph = collections.defaultdict(list) # 构造图
	        for u, v, t in times: 
	            graph[u].append([v, t])
	        queue = [[0, K]] # 优先队列，要以第一个元素也就是权重排序
	        visited = [False] * (N + 1) # 判断某个节点是否已经访问过
	        result = 0
	        while queue: # 遍历优先队列
	            weight, target = heapq.heappop(queue) # 弹出权重值最小的元素
	            if visited[target]: continue # 如果已经访问过就不再重复访问
	            visited[target] = True # 设置表示 target 节点已经访问过
	            result = max(result, weight) # 因为弹出的节点已经是经过排序的最小值，所以当到达目标 target 的时候，就是源点到 target 的最短距离
	            for v, t in graph[target]: # 更新后续相邻的节点及其权重并加入到优先队列中
	                heapq.heappush(queue, [t + weight, v])
	        for i in range(1, N + 1): # 如果有节点不可达，直接返回 -1
	            if not visited[i]:
	                return -1
	        return result
            	      
			
### 运行结果


	Runtime: 537 ms, faster than 42.88% of Python online submissions for Network Delay Time.
	Memory Usage: 15.9 MB, less than 44.52% of Python online submissions for Network Delay Time.

### 原题链接



https://leetcode.com/problems/network-delay-time/


您的支持是我最大的动力
