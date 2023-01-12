leetcode  502. IPO（python）

### 描述



Suppose LeetCode will start its IPO soon. In order to sell a good price of its shares to Venture Capital, LeetCode would like to work on some projects to increase its capital before the IPO. Since it has limited resources, it can only finish at most k distinct projects before the IPO. Help LeetCode design the best way to maximize its total capital after finishing at most k distinct projects.

You are given n projects where the ith project has a pure profit profits[i] and a minimum capital of capital[i] is needed to start it.

Initially, you have w capital. When you finish a project, you will obtain its pure profit and the profit will be added to your total capital.

Pick a list of at most k distinct projects from given projects to maximize your final capital, and return the final maximized capital.

The answer is guaranteed to fit in a 32-bit signed integer.



Example 1:

	Input: k = 2, w = 0, profits = [1,2,3], capital = [0,1,1]
	Output: 4
	Explanation: Since your initial capital is 0, you can only start the project indexed 0.
	After finishing it you will obtain profit 1 and your capital becomes 1.
	With capital 1, you can either start the project indexed 1 or the project indexed 2.
	Since you can choose at most 2 projects, you need to finish the project indexed 2 to get the maximum capital.
	Therefore, output the final maximized capital, which is 0 + 1 + 3 = 4.

	
Example 2:

	Input: k = 3, w = 0, profits = [1,2,3], capital = [0,1,2]
	Output: 6





Note:

	1 <= k <= 10^5
	0 <= w <= 10^9
	n == profits.length
	n == capital.length
	1 <= n <= 10^5
	0 <= profits[i] <= 10^4
	0 <= capital[i] <= 10^9


### 解析

根据题意，假设 LeetCode 即将开始 IPO。 为了将其股票卖个好价钱给 VC ， LeetCode 想在 IPO 前做一些项目来增资。 由于资源有限，在 IPO 前最多只能完成 k 个不同的项目。 在最多完成 k 个不同的项目后，帮助 LeetCode 设计可以最大化总资本的最佳方式。

给定 n 个项目，其中第 i 个项目具有纯利润  profits[i] 并且需要最低资本 capital[i] 来启动它。最初有资本 w ，当完成一个项目时，将获得它的纯利润，并将该利润添加到总资本中。从给定项目中选择最多 k 个不同项目的列表，以最大化最终资本，并返回最终最大化的资本。



### 解答
				
	class Solution(object):
	    def findMaximizedCapital(self, k, w, profits, capital):
	        """
	        :type k: int
	        :type w: int
	        :type profits: List[int]
	        :type capital: List[int]
	        :rtype: int
	        """
	        tasks = []
	        for i in range(len(profits)):
	            tasks.append([capital[i], profits[i]])
	        tasks.sort(key=lambda x:x[0])
	        count = 0
	        idx = 0
	        heap = []
	        while count<k:
	            while idx<len(tasks) and tasks[idx][0] <= w:
	                heappush(heap, -tasks[idx][1])
	                idx += 1
	            if not heap:
	                break
	            w += -heappop(heap)
	            count += 1
	        return w
	                

            	      
			
### 运行结果

	Runtime: 1522 ms, faster than 35.42% of Python online submissions for IPO.
	Memory Usage: 41.5 MB, less than 22.92% of Python online submissions for IPO.


原题链接：https://leetcode.com/problems/ipo/



您的支持是我最大的动力
