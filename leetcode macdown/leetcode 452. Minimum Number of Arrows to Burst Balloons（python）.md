leetcode 452. Minimum Number of Arrows to Burst Balloons （python）

### 每日经典

《赠李白》 ——杜甫（唐）

秋来相顾尚飘蓬，未就丹砂愧葛洪。

痛饮狂歌空度日，飞扬跋扈为谁雄。

### 描述


There are some spherical balloons taped onto a flat wall that represents the XY-plane. The balloons are represented as a 2D integer array points where points[i] = [x<sub>start</sub>, x<sub>end</sub>] denotes a balloon whose horizontal diameter stretches between x<sub>start</sub> and x<sub>end</sub>. You do not know the exact y-coordinates of the balloons.

Arrows can be shot up directly vertically (in the positive y-direction) from different points along the x-axis. A balloon with x<sub>start</sub> and x<sub>end</sub> is burst by an arrow shot at x if x<sub>start</sub> <= x <= x<sub>end</sub>. There is no limit to the number of arrows that can be shot. A shot arrow keeps traveling up infinitely, bursting any balloons in its path.

Given the array points, return the minimum number of arrows that must be shot to burst all balloons.




Example 1:

	Input: points = [[10,16],[2,8],[1,6],[7,12]]
	Output: 2
	Explanation: The balloons can be burst by 2 arrows:
	- Shoot an arrow at x = 6, bursting the balloons [2,8] and [1,6].
	- Shoot an arrow at x = 11, bursting the balloons [10,16] and [7,12].
	
Example 2:

	Input: points = [[1,2],[3,4],[5,6],[7,8]]
	Output: 4
	Explanation: One arrow needs to be shot for each balloon for a total of 4 arrows.


Example 3:


	Input: points = [[1,2],[2,3],[3,4],[4,5]]
	Output: 2
	Explanation: The balloons can be burst by 2 arrows:
	- Shoot an arrow at x = 2, bursting the balloons [1,2] and [2,3].
	- Shoot an arrow at x = 4, bursting the balloons [3,4] and [4,5].





Note:

* 	1 <= points.length <= 10^5
* 	points[i].length == 2
* 	-2^31 <= x<sub>start</sub> < x<sub>end</sub> <= 2^31 - 1


### 解析

根据题意，有一些球形气球贴在代表 XY 平面的平面墙上。 气球表示为 2D 整数数组点，其中 points[i] = [x<sub>start</sub>, x<sub>end</sub>] 表示直径横跨在 x<sub>start</sub> 和 x<sub>end</sub> 的气球，但是不需要知道气球的确切 y 坐标。

可以沿某个 x 轴的点向 y 方向射箭，如果 x<sub>start</sub> <= x <= x<sub>end</sub>，则在 x<sub>start</sub> 和 x<sub>end</sub> 范围内的气球会被箭头爆裂。 射箭不断向上移动，将其路径上的任何气球爆裂。给定数组 points ，返回使所有气球爆裂必须射出的最小箭头数。

其实思路很简单，只要是范围有重合的气球，那射爆他们只需要一箭就够了，所以先对 points 进行升序排序，然后使用贪心的思想尽量多融合范围内的气球，定义左边界 mn 和右边界 mx ，然后从左往右遍历 points ，只要新的范围 [a,b] 保证 !(a > mx or b < mn) ，说明还能继续融合，否则尽量能把范围重合的气球一箭射爆，也就是 result 加一，然后更新新的 mn 和 mx ，直到遍历结束，因为最后不管怎么样还需要一箭，所以 result 加一返回即可。

### 解答
				

	class Solution(object):
	    def findMinArrowShots(self, points):
	        """
	        :type points: List[List[int]]
	        :rtype: int
	        """
	        points.sort()
	        stack = [points[0]]
	        result = 0
	        mn = points[0][0]
	        mx = points[0][1]
	        for a, b in points[1:]:
	            if a > mx or b < mn:
	                result += 1
	                mn = a
	                mx = b
	            mn = max(mn, a)
	            mx = min(mx, b)
	        return result+1
	                
	                
	        
			
### 运行结果

	
	Runtime: 1543 ms, faster than 31.97% of Python online submissions for Minimum Number of Arrows to Burst Balloons.
	Memory Usage: 61.2 MB, less than 62.08% of Python online submissions for Minimum Number of Arrows to Burst Balloons.


原题链接：https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/



您的支持是我最大的动力
