leetcode 2250. Count Number of Rectangles Containing Each Point （python）




### 描述


You are given a 2D integer array rectangles where rectangles[i] = [li, hi] indicates that ith rectangle has a length of li and a height of hi. You are also given a 2D integer array points where points[j] = [xj, yj] is a point with coordinates (xj, yj).

The ith rectangle has its bottom-left corner point at the coordinates (0, 0) and its top-right corner point at (li, hi).

Return an integer array count of length points.length where count[j] is the number of rectangles that contain the jth point.

The ith rectangle contains the jth point if 0 <= xj <= li and 0 <= yj <= hi. Note that points that lie on the edges of a rectangle are also considered to be contained by that rectangle.


Example 1:

![](https://assets.leetcode.com/uploads/2022/03/02/example1.png)

	Input: rectangles = [[1,2],[2,3],[2,5]], points = [[2,1],[1,4]]
	Output: [2,1]
	Explanation: 
	The first rectangle contains no points.
	The second rectangle contains only the point (2, 1).
	The third rectangle contains the points (2, 1) and (1, 4).
	The number of rectangles that contain the point (2, 1) is 2.
	The number of rectangles that contain the point (1, 4) is 1.
	Therefore, we return [2, 1].

	
Example 2:

![](https://assets.leetcode.com/uploads/2022/03/02/example2.png)

	Input: rectangles = [[1,1],[2,2],[3,3]], points = [[1,3],[1,1]]
	Output: [1,3]
	Explanation:
	The first rectangle contains only the point (1, 1).
	The second rectangle contains only the point (1, 1).
	The third rectangle contains the points (1, 3) and (1, 1).
	The number of rectangles that contain the point (1, 3) is 1.
	The number of rectangles that contain the point (1, 1) is 3.
	Therefore, we return [1, 3].






Note:

	1 <= rectangles.length, points.length <= 5 * 10^4
	rectangles[i].length == points[j].length == 2
	1 <= li, xj <= 10^9
	1 <= hi, yj <= 100
	All the rectangles are unique.
	All the points are unique.


### 解析

根据题意，给定一个二维整数数组 rectangles，其中 rectangles[i] = [li, hi] 表示第 i 个矩形的长度为 li，高度为 hi。 还给出一个二维整数数组 points，其中 points[j] = [xj, yj] 是坐标为 (xj, yj) 的点。每个矩形左下角点都在在坐标 (0, 0) 处，右上角点在 (li, hi) 处。

返回长度为 points.length 的整数数组 cuont ，其中 count[i] 表示包含了第 i 个点的矩形的数量。这里需要注意的是位于矩形边缘的点也被认为包含在该矩形中。

读完题之后我们看限制条件发现很多变量都是万级别以上的，单位唯独一个 hi 和 yj 最大为 100 ，所以要想满足时间复杂度，肯定要从这里找突破口。

接下来我们找规律，一个点 (x,y) 如果被包含一个矩形 [l,h] 中，那么肯定 l>=x 且 h>=y ，所以我们对于每个点，只需要找满足这个条件的矩形个数即可。

因为上面的规律，我们发现找合理的 y 比较容易，我们先找合理的 y 再找合理的 x 。所以我们在这里使用一个二维列表 yy 存储当矩形高度为 y 的时候，对应的 x 列表，并将 x 列表进行升序排列。然后我们遍历 points 中所有的点，对于每个点，我们已经知道它的位置为 [x,y] ，那么有效的矩阵的 y 肯定是在 [y,100] 的范围内，所以我们遍历  [y,100] 的范围，然后我们将每个 yy[i] 中大于等于 x 的元素个数进行累加得到 cur ，将 cur 存入结果列表 reuslt 中，继续进行下一个点的计算即可。最后返回 result 即可。

因为 points 最长为 5 * 10^4 ，而 y 最大为 100 ，然后虽然是两重循环，其实时间复杂度最大为 5 * 10^6 可以看作是 O(N) ，而且每个循环还有二分查找的操作 ，所以总的时间复杂度为 O(NlogN)，空间复杂度为 O(N) 。

### 解答
				

	class Solution(object):
	    def countRectangles(self, rectangles, points):
	        """
	        :type rectangles: List[List[int]]
	        :type points: List[List[int]]
	        :rtype: List[int]
	        """
	        yy = [[] for _ in range(101)]
	        for x,y in rectangles:
	            yy[y].append(x)
	        for row in yy:
	            row.sort()
	        result = []
	        for x,y in points:
	            cur = 0
	            for i in range(y, 101):
	                row = yy[i]
	                j = bisect.bisect_left(row, x)
	                cur += len(row)-j
	            result.append(cur)
	        return result
            	      
			
### 运行结果



	47 / 47 test cases passed.
	Status: Accepted
	Runtime: 3253 ms
	Memory Usage: 37.5 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-290/problems/count-number-of-rectangles-containing-each-point/

您的支持是我最大的动力
