leetcode 207. Course Schedule （python）

### 描述

There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [a<sub>i</sub>, b<sub>i</sub>] indicates that you must take course b<sub>i</sub> first if you want to take course a<sub>i</sub>.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.





Example 1:


	Input: numCourses = 2, prerequisites = [[1,0]]
	Output: true
	Explanation: There are a total of 2 courses to take. 
	To take course 1 you should have finished course 0. So it is possible.
	
Example 2:

	Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
	Output: false
	Explanation: There are a total of 2 courses to take. 
	To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.





Note:

* 	1 <= numCourses <= 10^5
* 	0 <= prerequisites.length <= 5000
* 	prerequisites[i].length == 2
* 	0 <= a<sub>i</sub>, b<sub>i</sub> < numCourses
* 	All the pairs prerequisites[i] are unique.



### 解析


根据题意，必须上 numCourses 门课程，标记为 0 到 numCourses-1 。同时给定了先觉条件 prerequisites[i] = [a<sub>i</sub>, b<sub>i</sub>] ，表示先上课程 b<sub>i</sub> 才能上课程 a<sub>i</sub>。例如 [0,1] 表示，要先上课程 0 您必须先上课程 1 。如果您完成所有课程，则返回 True。 否则返回 False 。

这道题是典型的图论问题，表面是是要在规定了课程先后顺序的情况下判断是否可以把所有课程都完成，但是本质上是要找课程构成的图中是否存在环，如果有环说明不能完成，如果没环说明可以完成。这类题一般都有两种解法，一种是 DFS 另一种是 BFS 。
下面的解法是 DFS 解法。关键在于判断如何有环存在，那就是出现在遍历当前支路中遇到了已经访问过的节点的情况。

我们给其他支路遍历到底的节点标记 1 ，给当前正在遍历的支路中刚刚被访问的节点标记 2 。那么是什么时候标记 1 什么时候标记 2 ？在某条 DFS 的路径上，第一次遇到的节点时候标记 2 ，在回溯返回节点的时候标记 1 ，因为能成功返回的话，说明后续的节点都没有环。

### 解答
				

	class Solution(object):
	    def __init__(self):
	        self.visited = None
	        self.pre = None
	    def canFinish(self, numCourses, prerequisites):
	        """
	        :type numCourses: int
	        :type prerequisites: List[List[int]]
	        :rtype: bool
	        """
	        self.visited = [0]*numCourses
	        self.pre = [[] for _ in range(numCourses)]
	        for x,y in prerequisites:
	            self.pre[x].append(y)
	        for i in range(numCourses):
	            if not self.dfs(i): return False
	        return True
	    
	    def dfs(self, cur):
	        if self.visited[cur] == 1: return True
	        if self.visited[cur] == 2: return False
	        self.visited[cur] = 2
	        for p in self.pre[cur]:
	            if self.dfs(p) == False: return False
	        self.visited[cur] = 1
	        return True
	        
            	      
			
### 运行结果

	Runtime: 68 ms, faster than 98.63% of Python online submissions for Course Schedule.
	Memory Usage: 15.9 MB, less than 57.53% of Python online submissions for Course Schedule.

### 解析

根据题意，我们可以使用 BFS 的方法解题，每次在图中找入度为 0 的节点将其移出，如果最后没有入度为 0 的节点，那么就说明课程都学完了，否则说明有环，课程无法学完。
### 解答
	
	class Solution(object):
	    def __init__(self):
	        self.indegree = None
	        self.pre = None
	    def canFinish(self, numCourses, prerequisites):
	        """
	        :type numCourses: int
	        :type prerequisites: List[List[int]]
	        :rtype: bool
	        """
	        self.indegree = [0]*numCourses
	        self.pre = [[] for _ in range(numCourses)]
	        for x,y in prerequisites:
	            self.pre[x].append(y)
	            self.indegree[y] += 1
	        queue = []
	        count = 0
	        for i in range(numCourses):
	            if self.indegree[i] == 0:
	                queue.append(i)
	                count += 1
	        
	        while queue:
	            cur = queue.pop(0)
	            for p in self.pre[cur]:
	                self.indegree[p]-=1
	                if  self.indegree[p] == 0:
	                    queue.append(p)
	                    count += 1
	        return count == numCourses
	                
	                
### 运行结果            
	
	Runtime: 89 ms, faster than 34.06% of Python online submissions for Course Schedule.
	Memory Usage: 14.9 MB, less than 87.34% of Python online submissions for Course Schedule.



原题链接：https://leetcode.com/problems/course-schedule/



您的支持是我最大的动力
