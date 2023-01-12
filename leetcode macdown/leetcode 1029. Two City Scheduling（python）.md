leetcode  1029. Two City Scheduling（python）




### 描述



A company is planning to interview 2n people. Given the array costs where costs[i] = [aCost<sub>i</sub>, bCost<sub>i</sub>], the cost of flying the i<sub>th</sub> person to city a is aCost<sub>i</sub>, and the cost of flying the i<sub>th</sub> person to city b is bCost<sub>i</sub>.

Return the minimum cost to fly every person to a city such that exactly n people arrive in each city.

 

Example 1:

	Input: costs = [[10,20],[30,200],[400,50],[30,20]]
	Output: 110
	Explanation: 
	The first person goes to city A for a cost of 10.
	The second person goes to city A for a cost of 30.
	The third person goes to city B for a cost of 50.
	The fourth person goes to city B for a cost of 20.
	
	The total minimum cost is 10 + 30 + 50 + 20 = 110 to have half the people interviewing in each city.






Note:


* 	2 * n == costs.length
* 	2 <= costs.length <= 100
* 	costs.length is even.
* 	1 <= aCost<sub>i</sub>, bCost<sub>i</sub> <= 1000

### 解析

根据题意，公司计划面试 2n 人。给定一个数组 costs ，其中 costs[i] = [aCost<sub>i</sub>, bCost<sub>i</sub>] ，表示第 i 个人飞往 a 市的费用为 aCost<sub>i</sub> ，飞往 b 市的费用为 bCost<sub>i</sub> 。

返回将每个人都飞到 a 、b 中某座城市的最低费用，要求每个城市得有 n 人抵达。

 这道题明显是考察贪心的思想，我们可以换一个角度思考问题，假如我们先把所有的人都运到 b ，然后选出 N 个人再运到 a ，如果想改变一个人的行程，那么公司将会承担 aCost<sub>i</sub> - bCost<sub>i</sub> 的费用，然后按照升序排序，将前 n 个人都安排飞往 a ，再将后 n 个人飞往 b ，将所有费用相加返回即可。
 
 时间复杂度为 O (NlogN) ， 空间复杂度为 O(1) 。


### 解答
				

	class Solution(object):
	    def twoCitySchedCost(self, costs):
	        N = len(costs)
	        result = 0
	        costs.sort( key = lambda x : x[0] - x[1] )  
	        for i in range(N//2):
	            result += costs[i][0] + costs[i+N//2][1]
	        return result
	            
            	      
			
### 运行结果


	Runtime: 49 ms, faster than 17.26% of Python online submissions for Two City Scheduling.
	Memory Usage: 13.4 MB, less than 57.14% of Python online submissions for Two City Scheduling.

### 原题链接



https://leetcode.com/problems/two-city-scheduling/

您的支持是我最大的动力
