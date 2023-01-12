leetcode  1828. Queries on Number of Points Inside a Circle（python）

### 描述



You are given an array points where points[i] = [x<sub>i</sub>, y<sub>i</sub>] is the coordinates of the i<sup>th</sup> point on a 2D plane. Multiple points can have the same coordinates.

You are also given an array queries where queries[j] = [x<sub>j</sub>, y<sub>j</sub>, r<sub>j</sub>] describes a circle centered at (x<sub>j</sub>, y<sub>j</sub>) with a radius of r<sub>j</sub>.

For each query queries[j], compute the number of points inside the j<sup>th</sup> circle. Points on the border of the circle are considered inside.

Return an array answer, where answer[j] is the answer to the j<sup>th</sup> query.

Example 1:

![](https://assets.leetcode.com/uploads/2021/03/25/chrome_2021-03-25_22-34-16.png)

	Input: points = [[1,3],[3,3],[5,3],[2,2]], queries = [[2,3,1],[4,3,1],[1,1,2]]
	Output: [3,2,2]
	Explanation: The points and circles are shown above.
	queries[0] is the green circle, queries[1] is the red circle, and queries[2] is the blue circle.
	
Example 2:

![](https://assets.leetcode.com/uploads/2021/03/25/chrome_2021-03-25_22-42-07.png)

	Input: points = [[1,1],[2,2],[3,3],[4,4],[5,5]], queries = [[1,2,2],[2,2,2],[4,3,2],[4,3,3]]
	Output: [2,3,2,4]
	Explanation: The points and circles are shown above.
	queries[0] is green, queries[1] is red, queries[2] is blue, and queries[3] is purple.




Note:

* 1 <= points.length <= 500
* points[i].length == 2
* 0 <= x​​​​​​i, y​​​​​​i <= 500
* 1 <= queries.length <= 500
* queries[j].length == 3
* 0 <= x<sub>j</sub>, y<sub>j</sub> <= 500
* 1 <= r<sub>j</sub> <= 500
* All coordinates are integers.

**Follow up: Could you find the answer for each query in better complexity than O(n)?**

### 解析

根据题意，就是给出一个点列表 points ，然后又给出一组圆列表，每个圆元素有三个值，前两个是表示中心点为位置，第三个表示的是圆半径，计算每个每个圆可以囊括 points 中的点的个数，并且按照圆出现的顺序，在列表中显示每个圆能囊括的点的个数，其中点如果刚好在边上也算有效点。思路比较简单：

* 初始化一个结果列表 result ，长度为圆的个数
* 两层循环，一层循环点列表 points ，第二层循环圆列表 queries ，然后判断点到第 i 个圆的圆心的距离小于等于半径，即该点属于该圆的有效点，将 result[i] 加一
* 遍历结束，返回 result 即可





### 解答
				

	class Solution(object):
	    def countPoints(self, points, queries):
	        """
	        :type points: List[List[int]]
	        :type queries: List[List[int]]
	        :rtype: List[int]
	        """
	        result = [0] * len(queries)
	        for point in points:
	            x = point[0]
	            y = point[1]
	            for i,query in enumerate(queries):
	                xx = query[0]
	                yy = query[1]
	                r = query[2]
	                if abs(xx-x)**2+abs(yy-y)**2<=r**2:
	                    result[i] += 1
	        return result
            	      
			
### 运行结果


	Runtime: 1160 ms, faster than 35.35% of Python online submissions for Queries on Number of Points Inside a Circle.
	Memory Usage: 13.8 MB, less than 49.30% of Python online submissions for Queries on Number of Points Inside a Circle.


### 思考
看题目末尾的一行字：**Follow up: Could you find the answer for each query in better complexity than O(n)?**，这意思是要我们将时间复杂度降低到 O(n) ，这个还真有点难，相当于只能一次遍历就出结果，暂时没有思路。

原题链接：https://leetcode.com/problems/queries-on-number-of-points-inside-a-circle/



您的支持是我最大的动力
