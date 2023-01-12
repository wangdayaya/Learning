leetcode  2050. Parallel Courses III（python）

### 描述

You are given an integer n, which indicates that there are n courses labeled from 1 to n. You are also given a 2D integer array relations where relations[j] = [prevCourse<sub>j</sub>, nextCourse<sub>j</sub>] denotes that course prevCourse<sub>j</sub> has to be completed before course nextCourse<sub>j</sub> (prerequisite relationship). Furthermore, you are given a 0-indexed integer array time where time[i] denotes how many months it takes to complete the (i+1)<sub>th</sub> course.

You must find the minimum number of months needed to complete all the courses following these rules:

* You may start taking a course at any time if the prerequisites are met.

Any number of courses can be taken at the same time.
Return the minimum number of months needed to complete all the courses.

Note: The test cases are generated such that it is possible to complete every course (i.e., the graph is a directed acyclic graph).





Example 1:

![](https://assets.leetcode.com/uploads/2021/10/07/ex1.png)

	Input: n = 3, relations = [[1,3],[2,3]], time = [3,2,5]
	Output: 8
	Explanation: The figure above represents the given graph and the time required to complete each course. 
	We start course 1 and course 2 simultaneously at month 0.
	Course 1 takes 3 months and course 2 takes 2 months to complete respectively.
	Thus, the earliest time we can start course 3 is at month 3, and the total time required is 3 + 5 = 8 months.

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/10/07/ex2.png)

	Input: n = 5, relations = [[1,5],[2,5],[3,5],[3,4],[4,5]], time = [1,2,3,4,5]
	Output: 12
	Explanation: The figure above represents the given graph and the time required to complete each course.
	You can start courses 1, 2, and 3 at month 0.
	You can complete them after 1, 2, and 3 months respectively.
	Course 4 can be taken only after course 3 is completed, i.e., after 3 months. It is completed after 3 + 4 = 7 months.
	Course 5 can be taken only after courses 1, 2, 3, and 4 have been completed, i.e., after max(1,2,3,7) = 7 months.
	Thus, the minimum time needed to complete all the courses is 7 + 5 = 12 months.




Note:

* 1 <= n <= 5 * 10^4
* 0 <= relations.length <= min(n * (n - 1) / 2, 5 * 10^4)
* relations[j].length == 2
* 1 <= prevCourse<sub>j</sub>, nextCourse<sub>j</sub> <= n
* prevCourse<sub>j</sub> != nextCourse<sub>j</sub>
* All the pairs [prevCourse<sub>j</sub>, nextCourse<sub>j</sub>] are unique.
* time.length == n
* 1 <= time[i] <= 10^4
* The given graph is a directed acyclic graph.


### 解析

根据题意，给定一个整数 n ，它表示有 n 个标记为 1 到 n 的课程。 给出了一个二维整数数组 relations ，其中 relations[j] = [prevCourse<sub>j</sub>, nextCourse<sub>j</sub>] 表示课程 prevCourse<sub>j</sub> 必须在课程 nextCourse<sub>j</sub> 之前完成。 还给出一个以 0 为索引的整数数组 time，其中 time[i] 表示完成第 (i+1) 门课程需要多少个月。

必须按照以下规则找到完成所有课程所需的最少月数：

* 如果满足先决条件，您可以随时开始学习课程
* 可以同时参加任意数量的课程

可以看出这道题就是考察有向无环图的最长路径，对于一门课程 a 所需要的时间，取决于其前面课程中耗时最多的课程 b，假如 T(x) 表示某个课程的所需时间，那么 T(a) = max(T(b)+time[a]) 。另外我们可以使用入度来表示一门课程是否其前面的课程都修完，每确定一个先修课程入度减一，当减为 0 的时候，说明该课程已经可以算出所需最大时间。

另外实现拓扑排序一般会用 BFS ，队列初始时刻先加入入度为 0 的课程，每次循环弹出最前面的课程，它的所有后续可能可以更新一次，发现某个课程的入度减为 0 ，说明其先修课程都已经结束，可以加入到队列中。




### 解答
				

	class Solution(object):
	    def minimumTime(self, n, relations, time):
	        """
	        :type n: int
	        :type relations: List[List[int]]
	        :type time: List[int]
	        :rtype: int
	        """
	        nextCourse = [[] for _ in range(n+1)]
	        inDegree = [0] * (n+1)
	        for a,b in relations:
	            nextCourse[a].append(b)
	            inDegree[b] += 1
	        
	        queue = []
	        T = [0]*(n+1)
	        for i in range(1, n+1):
	            if inDegree[i] == 0:
	                queue.append(i)
	                T[i] = time[i-1]
	        result = 0
	        while queue:
	            cur = queue.pop(0)
	            result = max(result, T[cur])
	            for n in nextCourse[cur]:
	                T[n] = max(T[n], T[cur]+time[n-1])
	                inDegree[n] -= 1
	                if inDegree[n] == 0:
	                    queue.append(n)
	        
	        return result
            	      
			
### 运行结果

	
	Runtime: 1771 ms, faster than 51.85% of Python online submissions for Parallel Courses III.
	Memory Usage: 43.3 MB, less than 98.15% of Python online submissions for Parallel Courses III.

原题链接：https://leetcode.com/problems/parallel-courses-iii/



您的支持是我最大的动力
