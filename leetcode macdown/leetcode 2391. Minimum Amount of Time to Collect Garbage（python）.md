leetcode 2391. Minimum Amount of Time to Collect Garbage （python）




### 描述

You are given a 0-indexed array of strings garbage where garbage[i] represents the assortment of garbage at the i<sup>th</sup> house. garbage[i] consists only of the characters 'M', 'P' and 'G' representing one unit of metal, paper and glass garbage respectively. Picking up one unit of any type of garbage takes 1 minute. You are also given a 0-indexed integer array travel where travel[i] is the number of minutes needed to go from house i to house i + 1.

There are three garbage trucks in the city, each responsible for picking up one type of garbage. Each garbage truck starts at house 0 and must visit each house in order; however, they do not need to visit every house. Only one garbage truck may be used at any given moment. While one truck is driving or picking up garbage, the other two trucks cannot do anything.

Return the minimum number of minutes needed to pick up all the garbage.



Example 1:


	Input: garbage = ["G","P","GP","GG"], travel = [2,4,3]
	Output: 21
	Explanation:
	The paper garbage truck:
	1. Travels from house 0 to house 1
	2. Collects the paper garbage at house 1
	3. Travels from house 1 to house 2
	4. Collects the paper garbage at house 2
	Altogether, it takes 8 minutes to pick up all the paper garbage.
	The glass garbage truck:
	1. Collects the glass garbage at house 0
	2. Travels from house 0 to house 1
	3. Travels from house 1 to house 2
	4. Collects the glass garbage at house 2
	5. Travels from house 2 to house 3
	6. Collects the glass garbage at house 3
	Altogether, it takes 13 minutes to pick up all the glass garbage.
	Since there is no metal garbage, we do not need to consider the metal garbage truck.
	Therefore, it takes a total of 8 + 13 = 21 minutes to collect all the garbage.
	
Example 2:


	Input: garbage = ["MMM","PGM","GP"], travel = [3,10]
	Output: 37
	Explanation:
	The metal garbage truck takes 7 minutes to pick up all the metal garbage.
	The paper garbage truck takes 15 minutes to pick up all the paper garbage.
	The glass garbage truck takes 15 minutes to pick up all the glass garbage.
	It takes a total of 7 + 15 + 15 = 37 minutes to collect all the garbage.




Note:

	2 <= garbage.length <= 10^5
	garbage[i] consists of only the letters 'M', 'P', and 'G'.
	1 <= garbage[i].length <= 10
	travel.length == garbage.length - 1
	1 <= travel[i] <= 100


### 解析

根据题意，给定一个 0 索引的字符串数组 garbage ，其中 garbage[i] 表示第 i<sup>th</sup> 房子的垃圾分类。 garbage[i] 中仅包含字符“M”、“P”和“G”，分别代表一个单位的金属、纸和玻璃。 捡拾一个单位的任何类型的垃圾需要 1 分钟。 另外还给出一个 0 索引整数数组 travel ，其中 travel[i] 是从房子 i 到房子 i + 1 所需的分钟数。

全市共有三辆垃圾车，每辆负责捡拾一种垃圾。 每辆垃圾车从 0 号房屋开始，必须按顺序访问各个房屋； 然而，他们不需要走访每一所房子，只要某个房子后面没有给类型的垃圾，这辆车就立即停止使用不造消耗时间。 任何时候都只能使用一辆垃圾车。 当一辆卡车在开车或捡垃圾时，其他两辆卡车什么也做不了。返回捡起所有垃圾所需的最少分钟数。

其实这道题的描述看起来很长，其实就是考察的是前缀和，每辆车的时间消耗在两个部分，一方面是捡垃圾的时间，而我们所有车子捡垃圾的时间都是一样，所以通过对所有垃圾进行计数可以直接算出来的，另一方面是每辆车在房子之间穿梭的时间，我们只需要知道每辆车最后到达的房子位置，将其之前经过的房子的所有时间加起来即可，这可以通过前缀和求得，最后将两部分时间加起来就是最后的总的消耗时间。

时间复杂度为 O()，空间复杂度为 O()。

### 解答

	class Solution(object):
	    def garbageCollection(self, garbage, travel):
	        """
	        :type garbage: List[str]
	        :type travel: List[int]
	        :rtype: int
	        """
	        result = 0
	        d = collections.defaultdict(int)
	        for i,house in enumerate(garbage):
	            for g in house:
	                d[g] = i
	                result += 1
	        presum = [0]
	        for t in travel:
	            presum.append(presum[-1] + t)
	        for k,v in d.items():
	            result += presum[v]
	        return result


### 运行结果

	
	75 / 75 test cases passed.
	Status: Accepted
	Runtime: 1046 ms
	Memory Usage: 45.8 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-308/problems/minimum-amount-of-time-to-collect-garbage/

您的支持是我最大的动力
