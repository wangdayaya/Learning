leetcode 1627. Graph Connectivity With Threshold （python）

### 描述



We have n cities labeled from 1 to n. Two different cities with labels x and y are directly connected by a bidirectional road if and only if x and y share a common divisor strictly greater than some threshold. More formally, cities with labels x and y have a road between them if there exists an integer z such that all of the following are true:

* x % z == 0,
* y % z == 0, and
* z > threshold.

Given the two integers, n and threshold, and an array of queries, you must determine for each queries[i] = [a<sub>i</sub>, b<sub>i</sub>] if cities a<sub>i</sub> and b<sub>i</sub> are connected directly or indirectly. (i.e. there is some path between them).

Return an array answer, where answer.length == queries.length and answer[i] is true if for the i<sup>th</sup> query, there is a path between a<sub>i</sub> and b<sub>i</sub>, or answer[i] is false if there is no path.

Example 1:

![](https://assets.leetcode.com/uploads/2020/10/09/ex1.jpg)

	Input: n = 6, threshold = 2, queries = [[1,4],[2,5],[3,6]]
	Output: [false,false,true]
	Explanation: The divisors for each number:
	1:   1
	2:   1, 2
	3:   1, 3
	4:   1, 2, 4
	5:   1, 5
	6:   1, 2, 3, 6
	Using the underlined divisors above the threshold, only cities 3 and 6 share a common divisor, so they are the
	only ones directly connected. The result of each query:
	[1,4]   1 is not connected to 4
	[2,5]   2 is not connected to 5
	[3,6]   3 is connected to 6 through path 3--6

	
Example 2:

![](https://assets.leetcode.com/uploads/2020/10/10/tmp.jpg)

	Input: n = 6, threshold = 0, queries = [[4,5],[3,4],[3,2],[2,6],[1,3]]
	Output: [true,true,true,true,true]
	Explanation: The divisors for each number are the same as the previous example. However, since the threshold is 0,
	all divisors can be used. Since all numbers share 1 as a divisor, all cities are connected.



Example 3:

![](https://assets.leetcode.com/uploads/2020/10/17/ex3.jpg)

	Input: n = 5, threshold = 1, queries = [[4,5],[4,5],[3,2],[2,3],[3,4]]
	Output: [false,false,false,false,false]
	Explanation: Only cities 2 and 4 share a common divisor 2 which is strictly greater than the threshold 1, so they are the only ones directly connected.
	Please notice that there can be multiple queries for the same pair of nodes [x, y], and that the query [x, y] is equivalent to the query [y, x].



Note:


* 2 <= n <= 10^4
* 0 <= threshold <= n
* 1 <= queries.length <= 10^5
* queries[i].length == 2
* 1 <= a<sub>i</sub>, b<sub>i</sub> <= cities
* a<sub>i</sub> != b<sub>i</sub>

### 解析

根据题意，给定 n 个城市，从 1 到 n 标记。 当且仅当 x 和 y 共享一个严格大于某个 threshold 的公约数时，标签为 x 和 y 的两个不同城市通过一条双向道路直接相连。 如果存在一个整数 z 使得以下所有条件都为真，则标签为 x 和 y 的城市之间有一条道路：

* x % z == 0
* y % z == 0
* z > threshold

给定两个整数 n 和 threshold ，以及一个数组 queries ，针对每个 queries[i] = [a<sub>i</sub>, b<sub>i</sub>] ，确认城市 a<sub>i</sub> 和 b<sub>i</sub> 是否相连接，不管直接还是间接。 返回一个数组 answer ，其中 answer.length == queries.length 并且 answer[i] 如果对于第 i 个查询，如果城市之间存在路径，则  answer[i] 为 true，如果没有路径，则 answer[i] 为 false。

本题其实考察的是快读选择两个城市进行合并，如果用暴力解法，先遍历 x 城市再遍历 y 城市，在判断是否相通，时间复杂度太高了。换一个思路就是遍历大于 threshold 的公约数 t ，然后在小于等于 n 范围内找 t 的倍数，只要是 t 的倍数的城市都可以合并到一块，给这些城市赋予最小的城市 id 作为他们的祖先。最后便利 queries 中的每一对城市，如果他们的祖先相同说明就是相通的，否则说明不相通。

### 解答
				
	
	class Solution(object):
	    def areConnected(self, n, threshold, queries):
	        """
	        :type n: int
	        :type threshold: int
	        :type queries: List[List[int]]
	        :rtype: List[bool]
	        """
	        father = {i:i for i in range(1, n+1)}
	        visited = {i:0 for i in range(1, n+1)}
	        for t in range(threshold+1, n+1):
	            if visited[t]:continue
	            for x in range(t, n+1, t):
	                visited[x] = 1
	                if self.getFather(x,father) != self.getFather(t,father): self.union(x, t,father)
	        result = []
	        for query in queries:
	            result.append(self.getFather(query[0],father)==self.getFather(query[1],father))
	        return result
	    
	    def union(self,a,b,father):
	        a = father[a]
	        b = father[b]
	        if a<b: father[b] =  a
	        else: father[a] = b
	
	
	    def getFather(self,x,father):
	        if father[x] != x: 
	            father[x] = self.getFather(father[x],father)
	        return father[x]
	
	   
	

            	      
			
### 运行结果


	Runtime: 824 ms, faster than 100.00% of Python online submissions for Graph Connectivity With Threshold.
	Memory Usage: 51.4 MB, less than 100.00% of Python online submissions for Graph Connectivity With Threshold.


原题链接：https://leetcode.com/problems/graph-connectivity-with-threshold/



您的支持是我最大的动力
