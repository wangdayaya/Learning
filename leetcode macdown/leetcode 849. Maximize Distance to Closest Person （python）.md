

### 每日经典

《声声慢·寻寻觅觅》 ——李清照(宋) 


寻寻觅觅，冷冷清清，凄凄惨惨戚戚。乍暖还寒时候，最难将息。三杯两盏淡酒，怎敌他、晚来风急！雁过也，正伤心，却是旧时相识。

满地黄花堆积，憔悴损，如今有谁堪摘？守着窗儿，独自怎生得黑！梧桐更兼细雨，到黄昏、点点滴滴。这次第，怎一个愁字了得！

### 描述

You are given an array representing a row of seats where seats[i] = 1 represents a person sitting in the i<sup>th</sup> seat, and seats[i] = 0 represents that the i<sup>th</sup> seat is empty (0-indexed).

There is at least one empty seat, and at least one person sitting.

Alex wants to sit in the seat such that the distance between him and the closest person to him is maximized. 

Return that maximum distance to the closest person.




Example 1:

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/14c1d2dc4ab641a8a5921a3c2e850b85~tplv-k3u1fbpfcp-zoom-1.image)

	Input: seats = [1,0,0,0,1,0,1]
	Output: 2
	Explanation: 
	If Alex sits in the second open seat (i.e. seats[2]), then the closest person has distance 2.
	If Alex sits in any other open seat, the closest person has distance 1.
	Thus, the maximum distance to the closest person is 2.
	
	
Example 2:

	
	Input: seats = [1,0,0,0]
	Output: 3
	Explanation: 
	If Alex sits in the last seat (i.e. seats[3]), the closest person is 3 seats away.
	This is the maximum distance possible, so the answer is 3.

Example 3:

	Input: seats = [0,1]
	Output: 1
	
	



Note:

	2 <= seats.length <= 2 * 10^4
	seats[i] is 0 or 1.
	At least one seat is empty.
	At least one seat is occupied.

### 解析

根据题意，给定一个数组 seats 表示一排座位，其中 seat[i] = 1 表示坐在第 i 个座位上的人，而 seat[i] = 0 表示第 i 个座位是空的。这排座位至少有一个空座位，至少有一个人坐着。有一个名叫 Alex 的人想坐在座位上，使他和离他最近的人之间的距离最大化，返回离他最近的人的最大距离。

其实这个题按照我们最朴素的想法就能解决，我们找每个位置的左边最近有人位置的距离和右边最近诱人位置距离，比较取较小值即可，但是比较完每个位置之后的结果 result 取最大值。关键就在怎么找每个位置的距离左边和右边有人为之的距离。我们定义两个列表 L 和 R ，L[i] 表示索引 i 的位置距离其左边最近有人的距离，R[i] 表示索引 i 的位置距离其右边最近有人的距离。将这两个列表计算出来之后我们再次遍历 seats 比较 result = max(result, min(L[i], R[i])) ，遍历结束返回 result 即可。



### 解答
				
	class Solution(object):
	    def maxDistToClosest(self, seats):
	        """
	        :type seats: List[int]
	        :rtype: int
	        """
	        N = len(seats)
	        L = [float('inf')] * N
	        R = [float('inf')] * N
	        
	        for i, seat in enumerate(seats):
	            if seat == 1:
	                L[i] = 0
	            elif i>0:
	                L[i] = L[i-1] + 1
	        
	        for j in range(N-1, -1, -1):
	            if seats[j] == 1:
	                R[j] = 0
	            elif j<N-1:
	                R[j] = R[j+1] + 1
	                
	        result = 0
	        for i,seat in enumerate(seats):
	            result = max(result, min(L[i], R[i]))
	            
	        return result		
### 运行结果

	Runtime: 149 ms, faster than 44.00% of Python online submissions for Maximize Distance to Closest Person.
	Memory Usage: 14.9 MB, less than 6.29% of Python online submissions for Maximize Distance to Closest Person.


原题链接：https://leetcode.com/problems/maximize-distance-to-closest-person/


您的支持是我最大的动力
