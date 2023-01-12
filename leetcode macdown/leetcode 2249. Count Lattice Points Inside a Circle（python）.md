leetcode  2249. Count Lattice Points Inside a Circle（python）




### 描述

Given a 2D integer array circles where circles[i] = [xi, yi, ri] represents the center (xi, yi) and radius ri of the ith circle drawn on a grid, return the number of lattice points that are present inside at least one circle.

Note:

* A lattice point is a point with integer coordinates.
* Points that lie on the circumference of a circle are also considered to be inside it.
 



Example 1:

![](https://assets.leetcode.com/uploads/2022/03/02/exa-11.png)

	Input: circles = [[2,2,1]]
	Output: 5
	Explanation:
	The figure above shows the given circle.
	The lattice points present inside the circle are (1, 2), (2, 1), (2, 2), (2, 3), and (3, 2) and are shown in green.
	Other points such as (1, 1) and (1, 3), which are shown in red, are not considered inside the circle.
	Hence, the number of lattice points present inside at least one circle is 5.

	
Example 2:

![](https://assets.leetcode.com/uploads/2022/03/02/exa-22.png)

	Input: circles = [[2,2,2],[3,4,1]]
	Output: 16
	Explanation:
	The figure above shows the given circles.
	There are exactly 16 lattice points which are present inside at least one circle. 
	Some of them are (0, 2), (2, 0), (2, 4), (3, 2), and (4, 4).





Note:

	1 <= circles.length <= 200
	circles[i].length == 3
	1 <= xi, yi <= 100
	1 <= ri <= min(xi, yi)


### 解析


根据题意，给定一个 2D 整数数组 circles，其中 circles[i] = [xi, yi, ri] 表示在网格上绘制的第 i 个圆的中心为 (xi, yi) 半径为 ri ，返回在所有 circles 范围内的点的个数。这里需要注意的是点出现在圆的边上，也算范围内的点。

这道题只要读完题目就想起了当年中学时候的几何题，一摸一样，不过当年是手算，现在使用代码算，原理没有什么区别，考察的就是数学的解方程。

思路很简单，我们遍历所有的圆 circles[i] 的坐标和半径 x , y ,  r ，然后在该圆的范围内进行两重循环：

* 第一层循环找出所有 [x-r, x+r] 范围内的横坐标 i
* 第二层循环找出所有 [y-r, y+r] 范围内的纵坐标 j

对于每一个点 [i,j] , 根据勾股定理计算，只要距离 [x,y] 的长度小于等于 r ，那么就将这个点加入结果集合 result 中，遍历结束返回 result 的长度即可。

因为圆的个数最多为 200 个，而且 i 和 j 的范围也都为 200 ，所以进行 200 ^ 3 刚好在时间复杂度的 AC 范围内，时间复杂度为 O(circles.length \* 200 \* 200) ，空间复杂度为 O(circles.length \* 200 \* 200) 。

### 解答
				
	class Solution(object):
	    def countLatticePoints(self, circles):
	        """
	        :type circles: List[List[int]]
	        :rtype: int
	        """
	        result = set()
	        for x,y,r in circles: 
	            for i in range(x-r, x+r+1):
	                for j in range(y-r, y+r+1):
	                    if abs(i-x) * abs(i-x) + abs(j-y) * abs(j-y) <= r * r:
	                        result.add((i,j))
	        return len(result)

            	      
			
### 运行结果


	60 / 60 test cases passed.
	Status: Accepted
	Runtime: 2827 ms
	Memory Usage: 18.5 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-290/problems/count-lattice-points-inside-a-circle/


您的支持是我最大的动力
