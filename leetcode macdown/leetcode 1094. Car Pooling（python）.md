leetcode 1094. Car Pooling （python）

### 每日经典

《题都城南庄》 ——崔护（唐）

去年今日此门中，人面桃花相映红。

人面不知何处去，桃花依旧笑春风。

### 描述



There is a car with capacity empty seats. The vehicle only drives east (i.e., it cannot turn around and drive west).

You are given the integer capacity and an array trips where trip[i] = [numPassengers<sub>i</sub>, from<sub>i</sub>, to<sub>i</sub>] indicates that the i<sub>th</sub> trip has numPassengers<sub>i</sub> passengers and the locations to pick them up and drop them off are from<sub>i</sub> and to<sub>i</sub> respectively. The locations are given as the number of kilometers due east from the car's initial location.

Return true if it is possible to pick up and drop off all passengers for all the given trips, or false otherwise.

Example 1:

	Input: trips = [[2,1,5],[3,3,7]], capacity = 4
	Output: false

	
Example 2:

	
	Input: trips = [[2,1,5],[3,3,7]], capacity = 5
	Output: true




Note:

* 	1 <= trips.length <= 1000
* 	trips[i].length == 3
* 	1 <= numPassengers<sub>i</sub> <= 100
* 	0 <= from<sub>i</sub> < to<sub>i</sub> <= 1000
* 	1 <= capacity <= 10^5


### 解析

根据题意，有一辆容纳 capacity 个人的空座的汽车， 车辆只能向东行驶（即不能掉头向西行驶）。给定一个整数 capacity 和一个数组 trips，其中 trip[i] = [numPassengers<sub>i</sub>, from<sub>i</sub>, to<sub>i</sub>] 表示第 i 次行程有 numPassengers<sub>i</sub> 乘客，接送他们的位置分别是 from<sub>i</sub> 和 to<sub>i</sub>。 这些位置以汽车初始位置向东的公里距离给出。如果可以接送所有乘客，则返回 true，否则返回 false 。


题意很简单， 思路也很简单，就是判断在不同的时间点下车之后人们再上车，人数是否大于 capacity ，这里比较巧妙的是将 trips 这个三元元素列表进行了拆分，转化为一个二元元素列表 L ，里面保存 [from , numPassengers] 表示 from 时刻上车 numPassengers 人或者 [to, -numPassengers] 表示 to 时刻下车 numPassengers 人，我们对 L 按时间进行排序，这样我们在遍历 L 的时候就是按照时刻进行遍历的，再计算经过上下车之后人数是否超过 capacity ，如果超过直接返回 False ，否则返回 True 。

### 解答
				

	class Solution(object):
	    def carPooling(self, trips, capacity):
	        """
	        :type trips: List[List[int]]
	        :type capacity: int
	        :rtype: bool
	        """
	        L = []
	        people = 0
	        for n,s,e in trips:
	            L.append([s, n])
	            L.append([e, -n])
	        L.sort()
	        for position,n in L:
	            people += n
	            if people>capacity:
	                return False
	        return True
	            
            	      
			
### 运行结果

	Runtime: 94 ms, faster than 11.21% of Python online submissions for Car Pooling.
	Memory Usage: 13.8 MB, less than 86.64% of Python online submissions for Car Pooling.


原题链接：https://leetcode.com/problems/car-pooling/



您的支持是我最大的动力
