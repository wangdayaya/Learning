leetcode 57. Insert Interval （python）




### 描述

You are given an array of non-overlapping intervals intervals where intervals[i] = [start<sub>i</sub>, end<sub>i</sub>] represent the start and the end of the i<sub>th</sub> interval and intervals is sorted in ascending order by start<sub>i</sub>. You are also given an interval newInterval = [start, end] that represents the start and end of another interval. Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.





Example 1:

	Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
	Output: [[1,5],[6,9]]

	
Example 2:

	Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
	Output: [[1,2],[3,10],[12,16]]
	Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].





Note:

* 0 <= intervals.length <= 10^4
* intervals[i].length == 2
* 0 <= start<sub>i</sub> <= end<sub>i</sub> <= 10^5
* intervals is sorted by start<sub>i</sub> in ascending order.
* newInterval.length == 2
* 0 <= start <= end <= 10^5


### 解析

根据题意，给定一个非重叠区间区间数组，其中 intervals[i] = [start<sub>i</sub>, end<sub>i</sub>] 表示第 i 个区间的开始和结束，intervals 是一个按照区间起始端点升序排序的区间列表。另外给出一个 newInterval = [start, end] ，它表示另一个间隔的开始和结束。将 newInterval 插入到 intervals 中，以便 intervals 仍按 start<sub>i</sub> 升序排序，并且每个间隔中间仍然没有任何重叠间隔，返回最后的结果。

这是一个常见的区间覆盖问题，我们已经知道 intervals 是按照起点排序的区间列表，现在要求将 newInterval 插入 intervals 中之后，将所有覆盖的区间都融合成一个区间，我们可以将 newInterval 先插入 intervals ，并且再次按照区间的起点进行排序，我们遍历现有的所有区间：

当遍历到某个区间时，起点为 left ，终点为 right ，如果该区间与后面相邻的区间有覆盖，我们就持续往后找相邻的区间，并且同时可到达的区间的 right 最大值，直到没有可以覆盖的区间出现，我们将此时的 [left, right] 加入结果即可。

遍历结束返回 result 即为最后的有序无覆盖区间列表结果。

N 为 intervals 长度，时间复杂为 O(NlogN + N) ，主要是消耗在排序和遍历列表，空间复杂度为 O(N) 。




### 解答

	class Solution:
	    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
	        intervals.append(newInterval)
	        intervals.sort()
	        result = []
	        n = len(intervals)
	        i = 0
	        while i < n:
	            left = intervals[i][0]
	            right = intervals[i][1]
	            while i < n - 1 and intervals[i + 1][0] <= right:
	                i = i + 1
	                right = max(intervals[i][1], right)
	            result.append([left, right])
	            i = i + 1
	        return result


### 运行结果

* Runtime 88 ms ,Beats 46.71%
* Memory 7.3 MB ,Beats 47.28%

### 原题链接

	https://leetcode.com/problems/insert-interval/description/


您的支持是我最大的动力
