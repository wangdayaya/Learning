leetcode  2276. Count Integers in Intervals（python）




### 描述

Given an empty set of intervals, implement a data structure that can:

* Add an interval to the set of intervals.
* Count the number of integers that are present in at least one interval.

Implement the CountIntervals class:

* CountIntervals() Initializes the object with an empty set of intervals.
* void add(int left, int right) Adds the interval [left, right] to the set of intervals.
* int count() Returns the number of integers that are present in at least one interval.

Note that an interval [left, right] denotes all the integers x where left <= x <= right.



Example 1:
	
	Input
	["CountIntervals", "add", "add", "count", "add", "count"]
	[[], [2, 3], [7, 10], [], [5, 8], []]
	Output
	[null, null, null, 6, null, 8]
	
	Explanation
	CountIntervals countIntervals = new CountIntervals(); // initialize the object with an empty set of intervals. 
	countIntervals.add(2, 3);  // add [2, 3] to the set of intervals.
	countIntervals.add(7, 10); // add [7, 10] to the set of intervals.
	countIntervals.count();    // return 6
	                           // the integers 2 and 3 are present in the interval [2, 3].
	                           // the integers 7, 8, 9, and 10 are present in the interval [7, 10].
	countIntervals.add(5, 8);  // add [5, 8] to the set of intervals.
	countIntervals.count();    // return 8
	                           // the integers 2 and 3 are present in the interval [2, 3].
	                           // the integers 5 and 6 are present in the interval [5, 8].
	                           // the integers 7 and 8 are present in the intervals [5, 8] and [7, 10].
	                           // the integers 9 and 10 are present in the interval [7, 10].





Note:

	1 <= left <= right <= 10^9
	At most 10^5 calls in total will be made to add and count.
	At least one call will be made to count.

### 解析

根据题意，给定一组空的区间集合，实现一个数据结构，它可以：

* 将间隔添加到间隔集。
* 计算至少一个区间中存在的整数个数。

实现 CountIntervals 类：

* CountIntervals() 用一组空的间隔来初始化对象
* void add(int left, int right) 将区间 [left, right] 添加到区间集合中，这是闭区间
* int count() 返回出现在至少一个区间内的整数个数

比赛的时候我知道这道题要用到二分法，但是最后写的代码又臭又长，后来看了大佬的解释才知道，原来这是在考察珂朵莉树算法，大家可以去看一下[大佬的解释](https://leetcode.cn/problems/count-integers-in-intervals/solution/by-endlesscheng-clk2/)，我这里就不多介绍了。

整体的思路就是在执行 add 的时候，将新来的 [left,right] 区间尽量融入到与其有覆盖交集的更大的区间中，这样就可以将区间集合不断缩小，当然如果没有覆盖的区间，那么我们就在区间集合中新增加一个区间。

时间复杂度为 O(NlogM) 因为一共调用 N 次，每次时间复杂度为 O(logM) ，M 为区间集合数量。空间复杂度为 O(M)，可能会有 M 个区间段 ，N 是调用方法的次数。


### 解答
				

	from sortedcontainers import SortedDict
	class CountIntervals(object):
	    def __init__(self):
	        self.result = 0
	        self.d = SortedDict()
	
	    def add(self, left, right):
	        """
	        :type left: int
	        :type right: int
	        :rtype: None
	        """
	        idx = self.d.bisect_left(left)
	        while idx<len(self.d) and self.d.values()[idx] <= right:
	            r, l = self.d.popitem(idx)
	            left = min(l, left)
	            right = max(r, right)
	            self.result -= r-l+1
	        self.result += right - left + 1
	        self.d[right] = left
	        
	    def count(self):
	        """
	        :rtype: int
	        """
	        return self.result
            	      
			
### 运行结果



	73 / 73 test cases passed.
	Status: Accepted
	Runtime: 1366 ms
	Memory Usage: 54 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-293/problems/count-integers-in-intervals/


您的支持是我最大的动力
