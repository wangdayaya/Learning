leetcode  1235. Maximum Profit in Job Scheduling（python）

### 描述


We have n jobs, where every job is scheduled to be done from startTime[i] to endTime[i], obtaining a profit of profit[i].

You're given the startTime, endTime and profit arrays, return the maximum profit you can take such that there are no two jobs in the subset with overlapping time range.

If you choose a job that ends at time X you will be able to start another job that starts at time X.




Example 1:

![](https://assets.leetcode.com/uploads/2019/10/10/sample1_1584.png)

	Input: startTime = [1,2,3,3], endTime = [3,4,5,6], profit = [50,10,40,70]
	Output: 120
	Explanation: The subset chosen is the first and fourth job. 
	Time range [1-3]+[3-6] , we get profit of 120 = 50 + 70.


	
Example 2:

![](https://assets.leetcode.com/uploads/2019/10/10/sample22_1584.png)

	Input: startTime = [1,2,3,4,6], endTime = [3,5,10,6,9], profit = [20,20,100,70,60]
	Output: 150
	Explanation: The subset chosen is the first, fourth and fifth job. 
	Profit obtained 150 = 20 + 70 + 60.


Example 3:

![](https://assets.leetcode.com/uploads/2019/10/10/sample3_1584.png)

	Input: startTime = [1,1,1], endTime = [2,3,4], profit = [5,6,4]
	Output: 6

	



Note:

	1 <= startTime.length == endTime.length == profit.length <= 5 * 10^4
	1 <= startTime[i] < endTime[i] <= 10^9
	1 <= profit[i] <= 10^4


### 解析

根据题意，有 n 个工作，其中每个工作都计划从 startTime[i] 到 endTime[i] 完成，获得利润 profit[i]。题目中提供 startTime、endTime 和 profit 三个数组，返回可以获取的最大利润，执行的工作要求没有两个具有重叠时间范围的工作。如果选择在时间 X 结束了一个任务，那么能够在时间 X 开始另一任务。

一般碰到这种区间类型的题目，可以将每个任务按照 [ startTime[i] , endTime[i] ,  profit[i]] 的组合存放入列表 jobs 中，然后按套路将 jobs 按照结束时间来进行升序排序，我们通过找规律可以发现这道题适合动态规划，因为当我选择第 i 件任务的时候，我们只需要知道在这个任务开始之前的最近结束的任务的最大利润  dp[j]  ，然后比较结果值 result 和 dp[j]+jobs[i].profit 取较大值赋给 result 即可，所以我们定义 dp[i] 为 i 时刻结束任务时候的最大利润，动态规划的公式就是：

	dp[i] = dp[j] + jobs[i][2]
	
其中的 j 需要每次从已经遍历过的 endTimes 中寻找。

### 解答
				
	class Solution(object):
	    def jobScheduling(self, startTime, endTime, profit):
	        """
	        :type startTime: List[int]
	        :type endTime: List[int]
	        :type profit: List[int]
	        :rtype: int
	        """
	        N = len(startTime)
	        jobs = []
	        for i in range(N):
	            jobs.append([startTime[i], endTime[i], profit[i]])
	        jobs.sort(key=lambda x: x[1])
	        endTimes = []
	        dp = {}
	        result = 0
	        for i in range(N):
	            idx = bisect.bisect_right(endTimes, jobs[i][0])
	            if idx != 0:
	                cur = max(result, dp[endTimes[idx-1]] + jobs[i][2])
	            else:
	                cur = max(result, jobs[i][2])
	            dp[jobs[i][1]] = cur
	            endTimes.append(jobs[i][1])
	            result = max(result, cur)
	        return result
	        

            	      
			
### 运行结果


	Runtime: 524 ms, faster than 65.63% of Python online submissions for Maximum Profit in Job Scheduling.
	Memory Usage: 30.4 MB, less than 33.26% of Python online submissions for Maximum Profit in Job Scheduling.


原题链接：https://leetcode.com/problems/maximum-profit-in-job-scheduling/



您的支持是我最大的动力
