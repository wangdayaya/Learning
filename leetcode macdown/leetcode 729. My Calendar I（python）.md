leetcode 729. My Calendar I （python）




### 描述


You are implementing a program to use as your calendar. We can add a new event if adding the event will not cause a double booking. A double booking happens when two events have some non-empty intersection (i.e., some moment is common to both events.). The event can be represented as a pair of integers start and end that represents a booking on the half-open interval [start, end), the range of real numbers x such that start <= x < end.

Implement the MyCalendar class:

* MyCalendar() Initializes the calendar object.
* boolean book(int start, int end) Returns true if the event can be added to the calendar successfully without causing a double booking. Otherwise, return false and do not add the event to the calendar.



Example 1:

	Input
	["MyCalendar", "book", "book", "book"]
	[[], [10, 20], [15, 25], [20, 30]]
	Output
	[null, true, false, true]
	
	Explanation
	MyCalendar myCalendar = new MyCalendar();
	myCalendar.book(10, 20); // return True
	myCalendar.book(15, 25); // return False, It can not be booked because time 15 is already booked by another event.
	myCalendar.book(20, 30); // return True, The event can be booked, as the first event takes every time less than 20, but not including 20.

	




Note:

	0 <= start < end <= 10^9
	At most 1000 calls will be made to book.


### 解析

根据题意，在日历添加事件不。当有两个事件同时预定到相同时刻，就会发生双重预订。 事件可以表示为一对整数 start 和 end，表示半开区间 [start, end) 上的预订，实数范围表示为 start <= x < end 。

实现 MyCalendar 类：

* MyCalendar() 初始化日历对象。
* boolean book(int start, int end) 如果事件可以成功添加到日历中而不会导致重复预订，则返回 true 。 否则，返回 false 并且不要将事件添加到日历中。

因为这道题目的限制条件不是很严格，所以可以直接使用暴力解法，每次在现有的日历安排中一一进行比对看是否能够预定，如果范围没有重合部分则可以将事件加入日历并返回 True ，否则返回 False 。
### 解答

	class MyCalendar(object):
	    def __init__(self):
	        self.L =[]
	    def book(self, start, end):
	        for s, e in self.L:
	            if not (s >= end or start >= e):
	                return False
	        self.L.append((start, end))
	        return True
	        

### 运行结果

	Runtime: 614 ms, faster than 49.56% of Python online submissions for My Calendar I.
	Memory Usage: 14.3 MB, less than 41.23% of Python online submissions for My Calendar I.


### 原题链接


https://leetcode.com/problems/my-calendar-i/

您的支持是我最大的动力
