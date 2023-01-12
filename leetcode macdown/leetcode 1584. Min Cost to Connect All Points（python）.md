leetcode  1584. Min Cost to Connect All Points（python）




### 描述

You are given an array points representing integer coordinates of some points on a 2D-plane, where points[i] = [xi, yi].

The cost of connecting two points [xi, yi] and [xj, yj] is the manhattan distance between them: |xi - xj| + |yi - yj|, where |val| denotes the absolute value of val.

Return the minimum cost to make all points connected. All points are connected if there is exactly one simple path between any two points.



Example 1:

![](https://assets.leetcode.com/uploads/2020/08/26/d.png)

	Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
	Output: 20
	Explanation: 
	
![](https://assets.leetcode.com/uploads/2020/08/26/c.png)
	
	We can connect the points as shown above to get the minimum cost of 20.
	Notice that there is a unique path between every pair of points.

	
Example 2:

	Input: points = [[3,12],[-2,5],[-4,1]]
	Output: 18







Note:

	1 <= points.length <= 1000
	-10^6 <= xi, yi <= 10^6
	All pairs (xi, yi) are distinct.


### 解析


根据题意，给定一个数组 points 表示 2D 平面上某些点的整数坐标，其中 points[i] = [xi, yi]。 连接两个点 [xi, yi] 和 [xj, yj] 的成本是它们之间的曼哈顿距离：|xi - xj| + |yi - yj|，其中 |val| 表示 val 的绝对值。 返回使所有点连接的最小成本。

这道题其实考察的是 Kruskal 算法，因为要让平面中两个点之间的距离最短，且任意两点之间只有一条路径，最后的连接值尽可能小，能够满足任意两点之间有且仅有一条权重路径的只有最小生成树，而最小生成树有一个经典解法，就是 Kruskal 算法 ，这是一个贪心算法。

### 解答
				
	class Solution(object):
	    def minCostConnectPoints(self, points):
	        """
	        :type points: List[List[int]]
	        :rtype: int
	        """
	        dist = lambda x, y: abs(points[x][0] - points[y][0]) + abs(points[x][1] - points[y][1])
	
	        n = len(points)
	        dsu = DisjointSetUnion(n)
	        edges = list()
	
	        for i in range(n):
	            for j in range(i + 1, n):
	                edges.append((dist(i, j), i, j))
	        
	        edges.sort()
	        
	        ret, num = 0, 1
	        for length, x, y in edges:
	            if dsu.unionSet(x, y):
	                ret += length
	                num += 1
	                if num == n:
	                    break
	        
	        return ret
	
	class DisjointSetUnion:
	    def __init__(self, n):
	        self.n = n
	        self.rank = [1] * n
	        self.f = list(range(n))
	    
	    def find(self, x):
	        if self.f[x] == x:
	            return x
	        self.f[x] = self.find(self.f[x])
	        return self.f[x]
	    
	    def unionSet(self, x, y):
	        fx, fy = self.find(x), self.find(y)
	        if fx == fy:
	            return False
	
	        if self.rank[fx] < self.rank[fy]:
	            fx, fy = fy, fx
	        
	        self.rank[fx] += self.rank[fy]
	        self.f[fy] = fx
	        return True



        

            	      
			
### 运行结果

	Runtime: 3026 ms, faster than 59.86% of Python online submissions for Min Cost to Connect All Points.
	Memory Usage: 80.9 MB, less than 68.03% of Python online submissions for Min Cost to Connect All Points.



### 原题链接

https://leetcode.com/problems/min-cost-to-connect-all-points/


您的支持是我最大的动力
