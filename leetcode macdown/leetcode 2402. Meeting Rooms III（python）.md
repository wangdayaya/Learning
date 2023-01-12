leetcode  2402. Meeting Rooms III（python）




### 描述


You are given an integer n. There are n rooms numbered from 0 to n - 1. You are given a 2D integer array meetings where meetings[i] = [start<sub>i</sub>, end<sub>i</sub>] means that a meeting will be held during the half-closed time interval [start<sub>i</sub>, end<sub>i</sub>). All the values of start<sub>i</sub> are unique. Meetings are allocated to rooms in the following manner:

* Each meeting will take place in the unused room with the lowest number.
* If there are no available rooms, the meeting will be delayed until a room becomes free. The delayed meeting should have the same duration as the original meeting.
* When a room becomes unused, meetings that have an earlier original start time should be given the room.

Return the number of the room that held the most meetings. If there are multiple rooms, return the room with the lowest number. A half-closed interval [a, b) is the interval between a and b including a and not including b.


Example 1:

	Input: n = 2, meetings = [[0,10],[1,5],[2,7],[3,4]]
	Output: 0
	Explanation:
	- At time 0, both rooms are not being used. The first meeting starts in room 0.
	- At time 1, only room 1 is not being used. The second meeting starts in room 1.
	- At time 2, both rooms are being used. The third meeting is delayed.
	- At time 3, both rooms are being used. The fourth meeting is delayed.
	- At time 5, the meeting in room 1 finishes. The third meeting starts in room 1 for the time period [5,10).
	- At time 10, the meetings in both rooms finish. The fourth meeting starts in room 0 for the time period [10,11).
	Both rooms 0 and 1 held 2 meetings, so we return 0. 

	
Example 2:


	Input: n = 3, meetings = [[1,20],[2,10],[3,5],[4,9],[6,8]]
	Output: 1
	Explanation:
	- At time 1, all three rooms are not being used. The first meeting starts in room 0.
	- At time 2, rooms 1 and 2 are not being used. The second meeting starts in room 1.
	- At time 3, only room 2 is not being used. The third meeting starts in room 2.
	- At time 4, all three rooms are being used. The fourth meeting is delayed.
	- At time 5, the meeting in room 2 finishes. The fourth meeting starts in room 2 for the time period [5,10).
	- At time 6, all three rooms are being used. The fifth meeting is delayed.
	- At time 10, the meetings in rooms 1 and 2 finish. The fifth meeting starts in room 1 for the time period [10,12).
	Room 0 held 1 meeting while rooms 1 and 2 each held 2 meetings, so we return 1. 



Note:

* 	1 <= n <= 100
* 	1 <= meetings.length <= 105
* 	meetings[i].length == 2
* 	0 <= start<sub>i</sub> < end<sub>i</sub> <= 5 * 105
* 	All the values of start<sub>i</sub> are unique.


### 解析

根据题意，给定一个整数 n 。 有编号从 0 到 n - 1 的 n 个房间，。给定一个二维整数数组 meeting，其中 meeting[i] = [start<sub>i</sub>, end<sub>i</sub>] 表示会议将在左闭右开的 [start<sub>i</sub>, end<sub>i</sub>) 时间内举办。 start<sub>i</sub> 的所有值都是唯一的。 按以下方式分配到会议室：

* 每次会议将在编码最小的未使用房间举行
* 如果没有可用的房间，会议将延迟到房间空闲为止
* 当会议室闲置时，最先申请过的会议最先使用该会议室

返回举行最多次会议的房间号。 如果有多个房间符合，则返回编号最小的房间。 半闭区间 [a, b) 是包含 a 和不包含 b 的 a 到 b 之间的区间。

像这种题直接使用两个小根堆来解决即可，因为会议室有两种状态——空闲和正在使用，我们就定义两个小根堆 free 和 using ，free 中按照从小到大存储会议室编号，using 中按照（会议的结束时间，正在使用的会议室编号）来按照从小到大存储，因为会议室有“先到先得”的条件，所以我们将 meetings 进行排序，另外定义一个计数器 cnt 来对每个会议室的使用情况进行统计。然后遍历 meetings 

* 首先我们要判断如果 using 中有已经结束的会议，那就将他们弹出来，并将空出来的会议室编号加入到 free 中
* 如果当前没有空闲的会议室，那么我们就将当前的会议往后推迟到 using 中最先结束的会议之后，并更新其结束时间，如果有空闲的会议室直接将 free 中最前面的会议室编号弹出，此时将这个正在进行的会议室加入到 using 中，并给会议室编号在 cnt 中对应的位置加一，表示该会议室使用次数加一

最后我们遍历 cnt ，找出出现次数最多，但是序号大小最小的会议室编号，返回即可。

时间复杂度为 O(mlogm + mlogn + n) ，因为排序为 O(mlogm)  ，堆操作是 O(mlogn) ，最后遍历找答案是 O(n) ，空间复杂度为 O(n) ，m 为 meetings 的长度。


### 解答

	class Solution:
	    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
	        cnt = [0] * n
	        free = [i for i in range(n)] 
	        using = [] 
	        meetings.sort()
	        for start, end in meetings:
	            while using and using[0][0] <= start: 
	                end, i = heapq.heappop(using)
	                heapq.heappush(free, i)
	            if not free:
	                e, i = heapq.heappop(using)
	                end += e - start
	            else:
	                i = heapq.heappop(free)
	            cnt[i] += 1
	            heapq.heappush(using, (end, i))
	        result = 0
	        for i,c in enumerate(cnt):
	            if c > cnt[result]:
	                result = i
	        return result

### 运行结果

	81 / 81 test cases passed.
	Status: Accepted
	Runtime: 1636 ms
	Memory Usage: 60.3 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-309/problems/meeting-rooms-iii/


您的支持是我最大的动力
