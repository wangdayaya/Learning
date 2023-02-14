leetcode  2477. Minimum Fuel Cost to Report to the Capital（python）




### 描述


There is a tree (i.e., a connected, undirected graph with no cycles) structure country network consisting of n cities numbered from 0 to n - 1 and exactly n - 1 roads. The capital city is city 0. You are given a 2D integer array roads where roads[i] = [a<sub>i</sub>, b<sub>i</sub>] denotes that there exists a bidirectional road connecting cities a<sub>i</sub> and b<sub>i</sub>.

There is a meeting for the representatives of each city. The meeting is in the capital city. There is a car in each city. You are given an integer seats that indicates the number of seats in each car. A representative can use the car in their city to travel or change the car and ride with another representative. The cost of traveling between two cities is one liter of fuel. Return the minimum number of liters of fuel to reach the capital city.


Example 1:

![](https://assets.leetcode.com/uploads/2022/09/22/a4c380025e3ff0c379525e96a7d63a3.png)

	Input: roads = [[0,1],[0,2],[0,3]], seats = 5
	Output: 3
	Explanation: 
	- Representative1 goes directly to the capital with 1 liter of fuel.
	- Representative2 goes directly to the capital with 1 liter of fuel.
	- Representative3 goes directly to the capital with 1 liter of fuel.
	It costs 3 liters of fuel at minimum. 
	It can be proven that 3 is the minimum number of liters of fuel needed.

	
Example 2:


![](https://assets.leetcode.com/uploads/2022/11/16/2.png)

	Input: roads = [[3,1],[3,2],[1,0],[0,4],[0,5],[4,6]], seats = 2
	Output: 7
	Explanation: 
	- Representative2 goes directly to city 3 with 1 liter of fuel.
	- Representative2 and representative3 go together to city 1 with 1 liter of fuel.
	- Representative2 and representative3 go together to the capital with 1 liter of fuel.
	- Representative1 goes directly to the capital with 1 liter of fuel.
	- Representative5 goes directly to the capital with 1 liter of fuel.
	- Representative6 goes directly to city 4 with 1 liter of fuel.
	- Representative4 and representative6 go together to the capital with 1 liter of fuel.
	It costs 7 liters of fuel at minimum. 
	It can be proven that 7 is the minimum number of liters of fuel needed.

Example 3:

![](https://assets.leetcode.com/uploads/2022/09/27/efcf7f7be6830b8763639cfd01b690a.png)

	Input: roads = [], seats = 1
	Output: 0
	Explanation: No representatives need to travel to the capital city.



Note:

* 1 <= n <= 10^5
* roads.length == n - 1
* roads[i].length == 2
* 0 <= a<sub>i</sub>, b<sub>i</sub> < n
* a<sub>i</sub> != b<sub>i</sub>
* roads represents a valid tree.
* 1 <= seats <= 10^5


### 解析

根据题意，有一个树结构的国家网络，由 n 个城市组成，编号从 0 到 n-1 ，正好 n-1 条道路。首都是城市 0 ，给定一个二维整数数组 roads，其中 roads[i] = [a<sub>i</sub>, b<sub>i</sub>] 表示存在连接城市 a<sub>i</sub> 和 b<sub></sub> <sub>i</sub> 的双向道路。

每个城市的代表都要参加一个会议，会议在首都举行。每个城市都有一辆车，给定一个整数 seats ，指示每辆车的座位数。代表可以在他们的城市使用汽车旅行，或更换汽车与另一名代表一起乘车。在两个城市之间旅行的费用是一升燃料，返回到达首都的最低燃料升数。


这道题考查图和贪心的结合点，我们先构建出邻接图 g ，然后从 0 开始进行递归，使用递归函数来计算已经达到该节点所消耗的燃料数 ，我们经过观察例子总结出来到达此刻节点的消耗的燃料数是其子树节点个数除 seats 向上取整的结果，我们将此值加入到结果 result 中，经过递归将所有子树消耗的燃料不断累加起来就是最后的结果。

N 为节点个数，时间复杂度为 O(N) ，空间复杂度为 O(N)。

### 解答

	class Solution:
	    def minimumFuelCost(self, roads: List[List[int]], seats: int) -> int:
	        n = len(roads) + 1
	        if n == 1:
	            return 0
	        result = 0
	        g = [[] for _ in range(len(roads) + 1)]
	        for x, y in roads:
	            g[x].append(y)
	            g[y].append(x)
	        def dfs(curNode: int, father: int) -> int:
	            nonlocal result
	            nodeNum = 1
	            for son in g[curNode]:
	                if son != father:
	                    nodeNum += dfs(son, curNode)
	            if curNode:
	                result += math.ceil(nodeNum/seats)
	            return nodeNum
	        dfs(0, -1)
	        return result

### 运行结果

	Runtime 1861 ms，Beats 93.88%
	Memory 150.4 MB，Beats 87.85%

### 原题链接

	https://leetcode.com/problems/minimum-fuel-cost-to-report-to-the-capital/description/


您的支持是我最大的动力
