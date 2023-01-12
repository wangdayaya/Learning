leetcode  2065. Maximum Path Quality of a Graph（python）

### 描述


There is an undirected graph with n nodes numbered from 0 to n - 1 (inclusive). You are given a 0-indexed integer array values where values[i] is the value of the i<sub>th</sub> node. You are also given a 0-indexed 2D integer array edges, where each edges[j] = [u<sub>j</sub>, v<sub>j</sub>, time<sub>j</sub>] indicates that there is an undirected edge between the nodes u<sub>j</sub> and v<sub>j</sub>, and it takes time<sub>j</sub> seconds to travel between the two nodes. Finally, you are given an integer maxTime.

A valid path in the graph is any path that starts at node 0, ends at node 0, and takes at most maxTime seconds to complete. You may visit the same node multiple times. The quality of a valid path is the sum of the values of the unique nodes visited in the path (each node's value is added at most once to the sum).

Return the maximum quality of a valid path.

Note: There are at most four edges connected to each node.


Example 1:

![](https://assets.leetcode.com/uploads/2021/10/19/ex1drawio.png)

	Input: values = [0,32,10,43], edges = [[0,1,10],[1,2,15],[0,3,10]], maxTime = 49
	Output: 75
	Explanation:
	One possible path is 0 -> 1 -> 0 -> 3 -> 0. The total time taken is 10 + 10 + 10 + 10 = 40 <= 49.
	The nodes visited are 0, 1, and 3, giving a maximal path quality of 0 + 32 + 43 = 75.

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/10/19/ex2drawio.png)

	Input: values = [5,10,15,20], edges = [[0,1,10],[1,2,10],[0,3,10]], maxTime = 30
	Output: 25
	Explanation:
	One possible path is 0 -> 3 -> 0. The total time taken is 10 + 10 = 20 <= 30.
	The nodes visited are 0 and 3, giving a maximal path quality of 5 + 20 = 25.


Example 3:

![](https://assets.leetcode.com/uploads/2021/10/19/ex31drawio.png)

	Input: values = [1,2,3,4], edges = [[0,1,10],[1,2,11],[2,3,12],[1,3,13]], maxTime = 50
	Output: 7
	Explanation:
	One possible path is 0 -> 1 -> 3 -> 1 -> 0. The total time taken is 10 + 13 + 13 + 10 = 46 <= 50.
	The nodes visited are 0, 1, and 3, giving a maximal path quality of 1 + 2 + 4 = 7.

	
Example 4:

![](https://assets.leetcode.com/uploads/2021/10/21/ex4drawio.png)

	Input: values = [0,1,2], edges = [[1,2,10]], maxTime = 10
	Output: 0
	Explanation: 
	The only path is 0. The total time taken is 0.
	The only node visited is 0, giving a maximal path quality of 0.

	



Note:

* n == values.length
* 1 <= n <= 1000
* 0 <= values[i] <= 10^8
* 0 <= edges.length <= 2000
* edges[j].length == 3
* 0 <= u<sub>j</sub> < v<sub>j</sub> <= n - 1
* 10 <= time<sub>j</sub>, maxTime <= 100
* All the pairs [u<sub>j</sub>, v<sub>j</sub>] are unique.
* There are at most four edges connected to each node.
* The graph may not be connected.



### 解析


根据题意，有一个无向图有 n 个节点，编号从 0 到 n - 1（含）。 给定一个 0 索引的整数数组 values ，其中 values[i] 是第 i 个节点的值。 还给出了一个 0 索引的二维整数数组 edges ，其中每个 edges[j] = [u<sub>j</sub>, v<sub>j</sub>, time<sub>j</sub>] 表示节点 u<sub>j</sub> 和 v<sub>j</sub> 之间存在一条无向边，并且在节点之间移动需要 time<sub>j</sub> 秒。 最后给出了一个整数 maxTime 。

无向图中的有效路径是从节点 0 开始、在节点 0 结束，并且最多消耗 maxTime 秒来完成的任何路径。 可以多次访问同一个节点。 有效路径的质量是路径中访问的唯一节点的值的总和（每个节点的值最多添加一次）。返回有效路径的最高质量。注意：最多有四个边连接到每个节点。

这道题其实仔细想想就会发现很难按照常规的图算法解题，结合限制条件 10 <= time<sub>j</sub>, maxTime <= 100 和  There are at most four edges connected to each node 我们其实可以取巧解题，因为 time<sub>j</sub> 最小为 10 ，maxTime 最大为 100 ，说明最多可以走 10 步，而且每个节点最多有四个节点相连，那就是 4^10 条路径，只要暴力找出所有的路径，然后找有效路径的最大质量就可以。

### 解答
				

	class Solution(object):
	    def __init__(self):
	        self.result = 0
	        self.visited = [0] * 1000
	        self.d = {i: [] for i in range(1000)}
	
	    def maximalPathQuality(self, values, edges, maxTime):
	        """
	        :type values: List[int]
	        :type edges: List[List[int]]
	        :type maxTime: int
	        :rtype: int
	        """
	        n = len(values)
	        for x, y, v in edges:
	            self.d[x].append([y, v])
	            self.d[y].append([x, v])
	
	        self.visited[0] = 1
	        self.dfs(0, values[0], 0, maxTime, values)
	        return self.result
	
	    def dfs(self, current_node, totalValue, totalTime, maxTime, values):
	        if totalTime > maxTime: return
	        if current_node == 0:
	            self.result = max(totalValue, self.result)
	        for node, time in self.d[current_node]:
	            self.visited[node] += 1
	            self.dfs(node, totalValue + (values[node] if self.visited[node] == 1 else 0), totalTime + time, maxTime, values)
	            self.visited[node] -= 1
            	      
			
### 运行结果

	Runtime: 5348 ms, faster than 7.04% of Python online submissions for Maximum Path Quality of a Graph.
	Memory Usage: 14.1 MB, less than 100.00% of Python online submissions for Maximum Path Quality of a Graph.


原题链接：https://leetcode.com/problems/maximum-path-quality-of-a-graph/



您的支持是我最大的动力
