leetcode  2332. The Latest Time to Catch a Bus（python）




### 描述

You are given a 0-indexed integer array buses of length n, where buses[i] represents the departure time of the ith bus. You are also given a 0-indexed integer array passengers of length m, where passengers[j] represents the arrival time of the jth passenger. All bus departure times are unique. All passenger arrival times are unique.

You are given an integer capacity, which represents the maximum number of passengers that can get on each bus.The passengers will get on the next available bus. You can get on a bus that will depart at x minutes if you arrive at y minutes where y <= x, and the bus is not full. Passengers with the earliest arrival times get on the bus first. Return the latest time you may arrive at the bus station to catch a bus. You cannot arrive at the same time as another passenger.





Example 1:

	Input: buses = [10,20], passengers = [2,17,18,19], capacity = 2
	Output: 16
	Explanation: 
	The 1st bus departs with the 1st passenger. 
	The 2nd bus departs with you and the 2nd passenger.
	Note that you must not arrive at the same time as the passengers, which is why you must arrive before the 2nd passenger to catch the bus.

	
Example 2:


	Input: buses = [20,30,10], passengers = [19,13,26,4,25,11,21], capacity = 2
	Output: 20
	Explanation: 
	The 1st bus departs with the 4th passenger. 
	The 2nd bus departs with the 6th and 2nd passengers.
	The 3rd bus departs with the 1st passenger and you.



Note:


	n == buses.length
	m == passengers.length
	1 <= n, m, capacity <= 10^5
	2 <= buses[i], passengers[i] <= 10^9
	Each element in buses is unique.
	Each element in passengers is unique.

### 解析

根据题意，给定一个长度为 n 的 0 索引整数数组 bus ，其中 bus[i] 表示第 i 个路线的出发时间。 还给定一个长度为 m 的 0 索引整数数组 passengers ，其中 passengers[j] 表示第 j 个乘客的到达候车室时间。 所有巴士发车时间都是独一无二的。 所有乘客的到达时间都是独一无二的。 给定一个整数 capacity ，它代表每辆公共汽车上可以乘坐的最大乘客数量。

乘客将乘坐下一班可用的公共汽车。 如果在 y 分钟到达且 y <= x 并且巴士未满员，则您可以乘坐将在 x 分钟发车的巴士。 到达时间最早的乘客先上车。返回到达巴士站候车室去乘车的最晚时间。 您不能与其他乘客同时到达。

其实这道题看题比较复杂，我们画个直线草图，把人和车的时间都放上去看就比较好理解了，因为都是公交和乘客都是按照时间顺序进行的，所以先将 buses 和 passengers 都进行升序排序，我们就模拟乘客乘车的方式来找出我们可以到候车室的最晚时间，我们从前往后遍历 buses 数组，每个元素代表的是发车时间，所以我们再同时遍历 passengers 中乘客，我们有两种方式来不断更新结果：

* 如果乘客到候车室的时间小于等于当前车的发车时间，并且车还没有满，我们就能让其上车，此时我们可以达到候车室最晚的时间就可能是每个乘客的前一分钟（如果前一分钟没有其他乘客的话）
* 如果发车的时候车还没有坐满，那么我们就可以在发车的时候刚好到候车室直接乘车出发

时间复杂度为 O(NlogN+MlogM) ，空间复杂度为 O(1) 。
### 解答

	class Solution(object):
	    def latestTimeCatchTheBus(self, buses, passengers, capacity):
	        """
	        :type buses: List[int]
	        :type passengers: List[int]
	        :type capacity: int
	        :rtype: int
	        """
	        buses.sort()
	        passengers.sort()
	        s = set(passengers)
	        M = len(passengers)
	        result = 0
	        c = 0
	        j = 0
	        for b in buses:
	            while j < M and c < capacity and passengers[j] <= b:
	                if passengers[j] - 1 not in s:
	                    result = passengers[j] - 1
	                c += 1
	                j += 1
	            if c < capacity and (j==0 or passengers[j-1] < b):
	                result = b
	            c = 0
	        return result


### 运行结果

	50 / 50 test cases passed.
	Status: Accepted
	Runtime: 1079 ms
	Memory Usage: 35.3 MB


### 原题链接

https://leetcode.com/contest/biweekly-contest-82/problems/the-latest-time-to-catch-a-bus/



您的支持是我最大的动力
