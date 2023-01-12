leetcode 1396. Design Underground System （python）

### 描述

An underground railway system is keeping track of customer travel times between different stations. They are using this data to calculate the average time it takes to travel from one station to another.

Implement the UndergroundSystem class:

* void checkIn(int id, string stationName, int t)

		A customer with a card ID equal to id, checks in at the station stationName at time t.
		A customer can only be checked into one place at a time.
* void checkOut(int id, string stationName, int t)

		A customer with a card ID equal to id, checks out from the station stationName at time t.

* double getAverageTime(string startStation, string endStation)

		Returns the average time it takes to travel from startStation to endStation.
		The average time is computed from all the previous traveling times from startStation to endStation that happened directly, meaning a check in at startStation followed by a check out from endStation.
		The time it takes to travel from startStation to endStation may be different from the time it takes to travel from endStation to startStation.
		There will be at least one customer that has traveled from startStation to endStation before getAverageTime is called.

You may assume all calls to the checkIn and checkOut methods are consistent. If a customer checks in at time t<sub>1</sub> then checks out at time t<sub>2</sub>, then t<sub>1</sub> < t<sub>2</sub>. All events happen in chronological order.

Example 1:

	Input
	["UndergroundSystem","checkIn","checkIn","checkIn","checkOut","checkOut","checkOut","getAverageTime","getAverageTime","checkIn","getAverageTime","checkOut","getAverageTime"]
	[[],[45,"Leyton",3],[32,"Paradise",8],[27,"Leyton",10],[45,"Waterloo",15],[27,"Waterloo",20],[32,"Cambridge",22],["Paradise","Cambridge"],["Leyton","Waterloo"],[10,"Leyton",24],["Leyton","Waterloo"],[10,"Waterloo",38],["Leyton","Waterloo"]]
	
	Output
	[null,null,null,null,null,null,null,14.00000,11.00000,null,11.00000,null,12.00000]
	
	Explanation
	UndergroundSystem undergroundSystem = new UndergroundSystem();
	undergroundSystem.checkIn(45, "Leyton", 3);
	undergroundSystem.checkIn(32, "Paradise", 8);
	undergroundSystem.checkIn(27, "Leyton", 10);
	undergroundSystem.checkOut(45, "Waterloo", 15);  // Customer 45 "Leyton" -> "Waterloo" in 15-3 = 12
	undergroundSystem.checkOut(27, "Waterloo", 20);  // Customer 27 "Leyton" -> "Waterloo" in 20-10 = 10
	undergroundSystem.checkOut(32, "Cambridge", 22); // Customer 32 "Paradise" -> "Cambridge" in 22-8 = 14
	undergroundSystem.getAverageTime("Paradise", "Cambridge"); // return 14.00000. One trip "Paradise" -> "Cambridge", (14) / 1 = 14
	undergroundSystem.getAverageTime("Leyton", "Waterloo");    // return 11.00000. Two trips "Leyton" -> "Waterloo", (10 + 12) / 2 = 11
	undergroundSystem.checkIn(10, "Leyton", 24);
	undergroundSystem.getAverageTime("Leyton", "Waterloo");    // return 11.00000
	undergroundSystem.checkOut(10, "Waterloo", 38);  // Customer 10 "Leyton" -> "Waterloo" in 38-24 = 14
	undergroundSystem.getAverageTime("Leyton", "Waterloo");    // return 12.00000. Three trips "Leyton" -> "Waterloo", (10 + 12 + 14) / 3 = 12

	
Example 2:


	Input
	["UndergroundSystem","checkIn","checkOut","getAverageTime","checkIn","checkOut","getAverageTime","checkIn","checkOut","getAverageTime"]
	[[],[10,"Leyton",3],[10,"Paradise",8],["Leyton","Paradise"],[5,"Leyton",10],[5,"Paradise",16],["Leyton","Paradise"],[2,"Leyton",21],[2,"Paradise",30],["Leyton","Paradise"]]
	
	Output
	[null,null,null,5.00000,null,null,5.50000,null,null,6.66667]
	
	Explanation
	UndergroundSystem undergroundSystem = new UndergroundSystem();
	undergroundSystem.checkIn(10, "Leyton", 3);
	undergroundSystem.checkOut(10, "Paradise", 8); // Customer 10 "Leyton" -> "Paradise" in 8-3 = 5
	undergroundSystem.getAverageTime("Leyton", "Paradise"); // return 5.00000, (5) / 1 = 5
	undergroundSystem.checkIn(5, "Leyton", 10);
	undergroundSystem.checkOut(5, "Paradise", 16); // Customer 5 "Leyton" -> "Paradise" in 16-10 = 6
	undergroundSystem.getAverageTime("Leyton", "Paradise"); // return 5.50000, (5 + 6) / 2 = 5.5
	undergroundSystem.checkIn(2, "Leyton", 21);
	undergroundSystem.checkOut(2, "Paradise", 30); // Customer 2 "Leyton" -> "Paradise" in 30-21 = 9
	undergroundSystem.getAverageTime("Leyton", "Paradise"); // return 6.66667, (5 + 6 + 9) / 3 = 6.66667



Note:

* 1 <= id, t <= 10^6
* 1 <= stationName.length, startStation.length, endStation.length <= 10
* All strings consist of uppercase and lowercase English letters and digits.
* There will be at most 2 * 10^4 calls in total to checkIn, checkOut, and getAverageTime.
* Answers within 10<sup>-5</sup> of the actual value will be accepted.


### 解析


根据题意，这道题要使用地下铁路系统跟踪不同车站之间的客户旅行时间。他们使用这些函数来计算从一个站点到另一个站点的平均旅行时间。一个人只能在某个时间只能在一个站点上车，或者一个站点下车，另外计算 【a、b】两个站点的平均旅行时间就是，
在 a 上车的乘客且在 b 下车的乘客的旅程时间总和除人数。

这道题使用两个字典就能解决，字典 user 保存每个旅客的 [进站,进站时间] ，字典 traval 保存所有在进站为 s ，出站为 e 旅程中的乘客各自消耗的时间，这样我们最后在计算进站为 s ，出站为 e 的平均时间时，只需要对 traval[(s, e)] 求和然后除这个区间的总人数即可。

时间复杂度为 O(N)，空间复杂度为 O(N)。
### 解答
				
				

	class UndergroundSystem(object):
	
	    def __init__(self):
	        self.user = collections.defaultdict(list)
	        self.traval = collections.defaultdict(list)
	
	    def checkIn(self, id, s, time):
	        self.user[id] = [s, time]
	        
	
	    def checkOut(self, id, e, time):
	        s, pre_time = self.user[id]
	        self.traval[(s, e)].append(time-pre_time)
	        
	
	    def getAverageTime(self, s, e):
	        tmp = self.traval[(s, e)]
	        return float(sum(tmp))/len(tmp)
			
### 运行结果

	Runtime: 246 ms, faster than 59.15% of Python online submissions for Design Underground System.
	Memory Usage: 27.6 MB, less than 10.98% of Python online submissions for Design Underground System.


###原题链接
https://leetcode.com/problems/design-underground-system/



您的支持是我最大的动力
