leetcode  1232. Check If It Is a Straight Line（python）

### 描述



You are given an array coordinates, coordinates[i] = [x, y], where [x, y] represents the coordinate of a point. Check if these points make a straight line in the XY plane.



Example 1:

![](https://assets.leetcode.com/uploads/2019/10/15/untitled-diagram-2.jpg)


	Input: coordinates = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
	Output: true
	
Example 2:


![](https://assets.leetcode.com/uploads/2019/10/09/untitled-diagram-1.jpg)

	Input: coordinates = [[1,1],[2,2],[3,4],[4,5],[5,6],[7,7]]
	Output: false




Note:

	2 <= coordinates.length <= 1000
	coordinates[i].length == 2
	-10^4 <= coordinates[i][0], coordinates[i][1] <= 10^4
	coordinates contains no duplicate point.


### 解析

根据题意，就是判断坐标系中给出的点是不是都在一条直线上，如果 coordinates 中只有两个点直接返回 True ，因为平面中的两个点肯定在一条直线上。

如果有多个点，先计算出前两个点组成直线的斜率 p ，然后从第三个点开始遍历，计算它与第二点组成的直线的斜率是否和 p 相等，如果不等则直接返回 False ；

接着判断第四个点与第三个点组成的直线的斜率是否和 p 相等，如果不等则直接返回 False ，以此类推，如果遍历结束都在一条直线上返回 True 。

【注意】计算斜率的时候，当两个点的 y 都相等的时候斜率为 0，当两个点的 x 相等的时候斜率为一个大数，其他情况直接出斜率即可。另外程序中判断小数是否相等需要用到 abs(a-b) > 0.0001 。

### 解答
				

	class Solution(object):
	    def checkStraightLine(self, coordinates):
	        """
	        :type coordinates: List[List[int]]
	        :rtype: bool
	        """
	        if len(coordinates)==2:
	            return True
	        def cal_slope(p1, p2):
	            x, y = p1[0], p1[1]
	            x_, y_ = p2[0], p2[1]
	            deltay = y_ - y
	            deltax = x_ - x
	            if deltax == 0:
	                return float("inf")
	            elif deltay == 0:
	                return 0
	            else:
	                return deltay/deltax
	        p = cal_slope(coordinates[0], coordinates[1])
	        for i in range(2,len(coordinates)):
	            tmp = cal_slope(coordinates[i-1], coordinates[i])
	            if abs(tmp- p) > 0.0001:
	                return False
	        return True
            	      
			
### 运行结果

	Runtime: 32 ms, faster than 100.00% of Python online submissions for Check If It Is a Straight Line.
	Memory Usage: 13.8 MB, less than 73.10% of Python online submissions for Check If It Is a Straight Line.



原题链接：https://leetcode.com/problems/check-if-it-is-a-straight-line/



您的支持是我最大的动力
