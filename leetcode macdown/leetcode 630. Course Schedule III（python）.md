leetcode  630. Course Schedule III（python）

### 描述


There are n different online courses numbered from 1 to n. You are given an array courses where courses[i] = [duration<sub>i</sub>, lastDay<sub>i</sub>] indicate that the i<sub>th</sub> course should be taken continuously for duration<sub>i</sub> days and must be finished before or on lastDay<sub>i</sub>.

You will start on the 1<sub>st</sub> day and you cannot take two or more courses simultaneously.

Return the maximum number of courses that you can take.


Example 1:

	Input: courses = [[100,200],[200,1300],[1000,1250],[2000,3200]]
	Output: 3
	Explanation: 
	There are totally 4 courses, but you can take 3 courses at most:
	First, take the 1st course, it costs 100 days so you will finish it on the 100th day, and ready to take the next course on the 101st day.
	Second, take the 3rd course, it costs 1000 days so you will finish it on the 1100th day, and ready to take the next course on the 1101st day. 
	Third, take the 2nd course, it costs 200 days so you will finish it on the 1300th day. 
	The 4th course cannot be taken now, since you will finish it on the 3300th day, which exceeds the closed date.

	
Example 2:

	Input: courses = [[1,2]]
	Output: 1


Example 3:
	
	Input: courses = [[3,2],[4,3]]
	Output: 0


Note:

* 	1 <= courses.length <= 10^4
* 	1 <= duration<sub>i</sub>, lastDay<sub>i</sub> <= 10^4


### 解析


根据题意，有 n 个不同的在线课程，编号从 1 到 n 。 给定一个数组 courses ，其中 courses[i] = [duration<sub>i</sub>, lastDay<sub>i</sub>] 表示第 i 门课程应该在 duration<sub>i</sub> 天中连续学习，并且必须在 lastDay<sub>i</sub> 结束之前完成。从第一天开始学习课程，但是不能同时参加超过两门的课程，返回可以参加的最大课程数。

这道题本质上是考察贪心，我们按照截止时间对课程进行排序，我们可以在某个截止时间之前，从耗时最短的课程开始进行，尽量多学几门课程。假如已经学习了 N 门课程，这时新来了一门课，如果我们能在截止时间之前做完那么学习的课程变成 N+1 ，如果不能我们会将当前的 N+1 门课程中耗时最长的那个课程去掉，当前我们仍然只学习了 N 门课程，但是已经消耗的时间却被我们尽量缩减优化了，这样可以保证我们可以拥有更多的时间学习后面新进来的课程。

### 解答
				
	from sortedcontainers import SortedList
	class Solution(object):
	    def scheduleCourse(self, courses):
	        """
	        :type courses: List[List[int]]
	        :rtype: int
	        """
	        courses.sort(key=lambda x:x[1])
	        L = SortedList([])
	        days = 0
	        for i in range(len(courses)):
	            L.add(courses[i][0])
	            days += courses[i][0]
	            if days>courses[i][1]:
	                days -= L.pop()
	        return len(L)
	        

            	      
			
### 运行结果


	Runtime: 1030 ms, faster than 23.53% of Python online submissions for Course Schedule III.
	Memory Usage: 19.5 MB, less than 9.80% of Python online submissions for Course Schedule III.

### 解析

同样的道理还可以用大顶堆来解题，在速度上略有提升。


### 解答

	class Solution(object):
	    def scheduleCourse(self, courses):
	        """
	        :type courses: List[List[int]]
	        :rtype: int
	        """
	        heap = []
	        days = 0
	        for dur, last in sorted(courses, key=lambda x:x[1]):
	            if dur + days <= last:
	                days += dur
	                heappush(heap, -dur)
	            elif heap and -heap[0]>dur:
	                days += dur+heappop(heap)
	                heappush(heap, -dur)
	        return len(heap)

        

### 运行结果
	
	Runtime: 756 ms, faster than 50.98% of Python online submissions for Course Schedule III.
	Memory Usage: 19.2 MB, less than 29.41% of Python online submissions for Course Schedule III.

原题链接：https://leetcode.com/problems/course-schedule-iii/



您的支持是我最大的动力
