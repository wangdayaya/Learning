leetcode 210. Course Schedule II （python）

### 描述


There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [a<sub>i</sub>, b<sub>i</sub>] indicates that you must take course b<sub>i</sub> first if you want to take course a<sub>i</sub>.

* For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.


Example 1:


	Input: numCourses = 2, prerequisites = [[1,0]]
	Output: [0,1]
	Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].
	
Example 2:

	Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
	Output: [0,2,1,3]
	Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0.
	So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].


Example 3:

	Input: numCourses = 1, prerequisites = []
	Output: [0]

	



Note:

* 	1 <= numCourses <= 2000
* 	0 <= prerequisites.length <= numCourses * (numCourses - 1)
* 	prerequisites[i].length == 2
* 	0 <= a<sub>i</sub>, b<sub>i</sub> < numCourses
* 	a<sub>i</sub> != b<sub>i</sub>
* 	All the pairs [a<sub>i</sub>, b<sub>i</sub>] are distinct.


### 解析

根据题意，有 numCourses 门课程，标记为从 0 到 numCourses-1 。同时还有先决条件 prerequisites[i] = [a<sub>i</sub>, b<sub>i</sub>]，表示如果您想上 a<sub>i</sub> ，必须先上课程  b<sub>i</sub> 。例如 [0,1] 表示，要想上课程 0 必须先上课程 1 。

返回完成所有课程的顺序。 如果有许多有效答案，则返回任何一个。 如果无法完成所有课程，则返回空数组。

这道题是 [207. Course Schedule](https://leetcode.com/problems/course-schedule/) 的升级版，内容大同小异，可以返回去复习一下。
这里直接用 BFS 的解法进行解题，原理仍然是将入度为 0 的节点放入队列中，之后每次将图中入度为 0 的节点移除，如果入度为 0 的节点个数和图中的节点个数相同，因为是从后向前遍历的所以将经过的节点逆序返回，否则直接返回空列表即可。


### 解答
				
	
	class Solution(object):
	    
	    def findOrder(self, numCourses, prerequisites):
	        """
	        :type numCourses: int
	        :type prerequisites: List[List[int]]
	        :rtype: List[int]
	        """
	        pre = [[] for _ in range(numCourses)]
	        indegree = [0] * numCourses
	        queue = []
	        result = []
	        
	        for x,y in prerequisites:
	            pre[x].append(y)
	            indegree[y] += 1
	        
	        for i in range(numCourses):
	            if indegree[i] == 0:
	                queue.append(i)
	                
	        while queue:
	            cur = queue.pop(0)
	            result.append(cur)
	            for p in pre[cur]:
	                indegree[p] -= 1
	                if indegree[p] == 0:
	                    queue.append(p)
	        
	        return result[::-1] if len(result) == numCourses else []
	  	      
			
### 运行结果

	Runtime: 79 ms, faster than 73.44% of Python online submissions for Course Schedule II.
	Memory Usage: 14.6 MB, less than 99.38% of Python online submissions for Course Schedule II.


### 解析

当然了也可以直接按照学习课程的先后顺序直接返回结果即可，只需要将上面算法中的 pre 改成下面的 nxt ，最后找到的 result 自然就是正向课程顺序。


### 解答

	class Solution(object):
	    
	    def findOrder(self, numCourses, prerequisites):
	        """
	        :type numCourses: int
	        :type prerequisites: List[List[int]]
	        :rtype: List[int]
	        """
	        nxt = [[] for _ in range(numCourses)]
	        indegree = [0] * numCourses
	        queue = []
	        result = []
	        
	        for x,y in prerequisites:
	            nxt[y].append(x)
	            indegree[x] += 1
	        
	        for i in range(numCourses):
	            if indegree[i] == 0:
	                queue.append(i)
	                
	        while queue:
	            cur = queue.pop(0)
	            result.append(cur)
	            for p in nxt[cur]:
	                indegree[p] -= 1
	                if indegree[p] == 0:
	                    queue.append(p)
	        
	        return result if len(result) == numCourses else []
	        
	        
	      
### 运行结果

	
	Runtime: 84 ms, faster than 53.81% of Python online submissions for Course Schedule II.
	Memory Usage: 14.7 MB, less than 97.94% of Python online submissions for Course Schedule II.

原题链接：https://leetcode.com/problems/course-schedule-ii/



您的支持是我最大的动力
