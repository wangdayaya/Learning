leetcode  2054. Two Best Non-Overlapping Events（python）

### 描述


You are given a 0-indexed 2D integer array of events where events[i] = [startTime<sub>i</sub>, endTime<sub>i</sub>, value<sub>i</sub>]. The i<sub>th</sub> event starts at startTime<sub>i</sub> and ends at endTime<sub>i</sub>, and if you attend this event, you will receive a value of value<sub>i</sub>. You can choose at most two non-overlapping events to attend such that the sum of their values is maximized.

Return this maximum sum.

Note that the start time and end time is inclusive: that is, you cannot attend two events where one of them starts and the other ends at the same time. More specifically, if you attend an event with end time t, the next event must start at or after t + 1.


Example 1:

![](https://assets.leetcode.com/uploads/2021/09/21/picture5.png)

	Input: events = [[1,3,2],[4,5,2],[2,4,3]]
	Output: 4
	Explanation: Choose the green events, 0 and 1 for a sum of 2 + 2 = 4.

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/09/21/picture1.png)

	Input: events = [[1,3,2],[4,5,2],[1,5,5]]
	Output: 5
	Explanation: Choose event 2 for a sum of 5.


Example 3:

![](https://assets.leetcode.com/uploads/2021/09/21/picture3.png)
	
	Input: events = [[1,5,3],[1,5,1],[6,6,5]]
	Output: 8
	Explanation: Choose events 0 and 2 for a sum of 3 + 5 = 8.





Note:

* 	2 <= events.length <= 105
* 	events[i].length == 3
* 	1 <= startTime<sub>i</sub> <= endTime<sub>i</sub> <= 109
* 	1 <= value<sub>i</sub> <= 106



### 解析

根据题意，给定一个 0 索引的 2D 整数数组 events ，其中 events[i] = [startTime<sub>i</sub>, endTime<sub>i</sub>, value<sub>i</sub>] 。 第 i 个事件从 startTime<sub>i</sub> 开始，在 endTime<sub>i</sub> 结束，如果你参加这个事件，你会收到一个 value<sub>i</sub> 的值。 最多可以选择两个不重叠的事件来参加，以使它们的值之和最大化，返回这个最大和。

请注意，活动的开始时刻和结束时刻是不允许覆盖的。即如果参加一个事件的结束时间为 t ，下一个事件的开始时间应该是 t+1 。

其实说白了这个题就是最多找两个区间的 value ，能让其保证最大。区间类的题目因为其自身的特殊性，的常规套路一般都是要排序的贪心，我们这里为了保证前后两个区间不覆盖，可以对 events 中的 endTime<sub>i</sub> 进行升序排序：

* 定义 rollingMax 列表，rollingMax[i] 表示经过排序之后，event 前 i 个（包括当前）事件中已经出现的最大的 value ，直接进行比较最大值即可得到
* 定义 endTimes 列表，存放遍历过的 endTime 
* 遍历 events 中的每个事件的开始时间 a 、结束时间 b 、值 v，因为可能只选择一个区间作为最大值，所以要对该事件的 v 和结果值 result 比较取最大值。然后在 endTimes 中找刚好大于 a 的索引，如果不为 0 那么，就将 rollingMax[idx-1] + v 和结果值 result 比较去最大值，然后将结束时间加入到 endTime 中
* 最后返会结果值 result 

### 解答
				

	class Solution(object):
	    def maxTwoEvents(self, events):
	        """
	        :type events: List[List[int]]
	        :rtype: int
	        """
	        N = len(events)
	        events.sort(key=lambda x:x[1])
	        endTimes = []
	        rollingMax = [0]*N
	        m = 0
	        for i in range(N):
	            m = max(m, events[i][2])
	            rollingMax[i] = m
	        result = 0
	        for i in range(N):
	            a = events[i][0]-1
	            b = events[i][1]
	            v = events[i][2]
	            result = max(result, v)
	            idx = bisect.bisect_right(endTimes, a)
	            if idx!=0:
	                result = max(result, rollingMax[idx-1] + v)
	            endTimes.append(b)
	        return result
	                
	        
	        
	        
            	      
			
### 运行结果

	Runtime: 1676 ms, faster than 53.40% of Python online submissions for Two Best Non-Overlapping Events.
	Memory Usage: 58.3 MB, less than 73.79% of Python online submissions for Two Best Non-Overlapping Events.


原题链接：https://leetcode.com/problems/two-best-non-overlapping-events/



您的支持是我最大的动力
