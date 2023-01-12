leetcode 2071. Maximum Number of Tasks You Can Assign （python）

### 描述


You have n tasks and m workers. Each task has a strength requirement stored in a 0-indexed integer array tasks, with the i<sup>th</sup> task requiring tasks[i] strength to complete. The strength of each worker is stored in a 0-indexed integer array workers, with the j<sup>th</sup> worker having workers[j] strength. Each worker can only be assigned to a single task and must have a strength greater than or equal to the task's strength requirement (i.e., workers[j] >= tasks[i]).

Additionally, you have pills magical pills that will increase a worker's strength by strength. You can decide which workers receive the magical pills, however, you may only give each worker at most one magical pill.

Given the 0-indexed integer arrays tasks and workers and the integers pills and strength, return the maximum number of tasks that can be completed.




Example 1:

	Input: tasks = [3,2,1], workers = [0,3,3], pills = 1, strength = 1
	Output: 3
	Explanation:
	We can assign the magical pill and tasks as follows:
	- Give the magical pill to worker 0.
	- Assign worker 0 to task 2 (0 + 1 >= 1)
	- Assign worker 1 to task 1 (3 >= 2)
	- Assign worker 2 to task 0 (3 >= 3)

	
Example 2:

	Input: tasks = [5,4], workers = [0,0,0], pills = 1, strength = 5
	Output: 1
	Explanation:
	We can assign the magical pill and tasks as follows:
	- Give the magical pill to worker 0.
	- Assign worker 0 to task 0 (0 + 5 >= 5)


Example 3:

	Input: tasks = [10,15,30], workers = [0,10,10,10,10], pills = 3, strength = 10
	Output: 2
	Explanation:
	We can assign the magical pills and tasks as follows:
	- Give the magical pill to worker 0 and worker 1.
	- Assign worker 0 to task 0 (0 + 10 >= 10)
	- Assign worker 1 to task 1 (10 + 10 >= 15)

	
Example 4:


	Input: tasks = [5,9,8,5,9], workers = [1,6,4,2,6], pills = 1, strength = 5
	Output: 3
	Explanation:
	We can assign the magical pill and tasks as follows:
	- Give the magical pill to worker 2.
	- Assign worker 1 to task 0 (6 >= 5)
	- Assign worker 2 to task 2 (4 + 5 >= 8)
	- Assign worker 4 to task 3 (6 >= 5)
	


Note:

	
	n == tasks.length
	m == workers.length
	1 <= n, m <= 5 * 10^4
	0 <= pills <= m
	0 <= tasks[i], workers[j], strength <= 10^9

### 解析

根据题意，题目中给出 n 个任务和 m 个工人。 每个任务都有一个存储在 0 索引整数数组 tasks 中的强度要求，第 i 个任务需要 tasks[i] 强度来完成。 每个 worker 的强度存储在一个索引为 0 的整数数组 workers 中，第 j 个 worker 具有 workers[j] 的强度。 每个工人只能分配到一个任务，并且强度必须大于或等于任务的强度要求（即，workers[j] >= tasks[i]）。此外，题中还给出了数量不确定的神奇的药丸 pills ，可以增加工人 strength 大小的力气。 让我们自行决定哪些工人获得药丸，但最多只能给每个工人一颗药丸。给定 0 索引的整数数组 tasks 和 workers 以及整数 pills 和 strength ，返回可以完成的最大任务数。

其实读完题意之后，并且结合限制条件，肯定无法进行暴力解题，因为会超时，而且像这种题暴力解题一下也无从下手，从题目中我们进行分析之后可以得到两个结论：

* 当这些工人刚好能完成 n 件任务的时候，那就肯定无法完成大于等于 n+1 件任务，这时我们就会自然想到使用二分搜索法来进行解题
* 在做任务的时候，因为简单的任务不知道分给能力最强的工人还是吃了药的工人，所以可以尝试从最难的任务开始解决，将最难的任务交给能力最强的工人，但是也可能最强的工人也做不了任务，那就只能将任务交给一个吃了药丸之后能力刚好大于任务的工人去做，这样是最经济的方案。在面对次最难的任务时，同样最强工人还没用的时候，让最强工人解决，否则在剩下的工人中找一个吃了药丸能力刚好超过任务的工人，这时我们就知道需要找一个能在删除元素时候自动排序的数据结构来存放工人

不知道为什么上面的解法总是超时，我将超时的用例放入 Testcase 中运行却能通过，百思不得其解。

### 解答

	from sortedcontainers import SortedList
	class Solution(object):
	    def maxTaskAssign(self, tasks, workers, pills, strength):
	        """
	        :type tasks: List[int]
	        :type workers: List[int]
	        :type pills: int
	        :type strength: int
	        :rtype: int
	        """
	        if tasks == workers: return len(tasks)
	        tasks.sort()
	        workers.sort()
	        
	        def check(tasks, workers, pills):
	            while len(tasks) and len(workers):
	                if workers[-1]>=tasks[-1]:
	                    tasks.pop()
	                    workers.pop()
	                elif pills:
	                    idx = bisect.bisect_left(workers, tasks[-1] - strength)
	                    if idx>=len(workers):
	                        return False
	                    workers.pop(idx)
	                    tasks.pop()
	                    pills -= 1
	                else:
	                    return False
	            return True
	        
	        left,right = 0,min(len(tasks), len(workers))
	        while left<right:
	            mid = right - (right-left)//2
	            if check(tasks[:mid], workers[-mid:], pills):
	                left = mid
	            else:
	                right = mid-1
	        return left
	            

### 运行结果


	Runtime: 1704 ms, faster than 92.86% of Python online submissions for Maximum Number of Tasks You Can Assign.
	Memory Usage: 22.1 MB, less than 25.00% of Python online submissions for Maximum Number of Tasks You Can Assign.


### 解析

另外，总体框架和上面一样，但是反正已经假定二分搜索法每次都能完成 k 个任务，我们找到最容易的 k 个任务，然后我们从能力最大的 k 个工人入手，这里面如果能力最小的的工人吃了药丸都无法解决选定任务中的最轻松的任务，直接判断无法完成 k 个任务，继续进行其他 k 值的二分搜索。这里很奇怪，如果把下面代码中的：

	idx = _tasks.bisect_right(worker + strength)
改成：
	
	idx = bisect.bisect_right(_tasks, worker + strength)

就会超时，莫名其妙，从函数的耗时来看，很明显后一种耗时更少，为啥为超时呢，更加莫名其妙。我总共提交了 15 次 TLE ，6 次 RE ，服了。

### 解答
				
	from sortedcontainers import SortedList
	class Solution(object):
	    def maxTaskAssign(self, tasks, workers, pills, strength):
	        """
	        :type tasks: List[int]
	        :type workers: List[int]
	        :type pills: int
	        :type strength: int
	        :rtype: int
	        """
	        tasks.sort()
	        workers.sort()
	        
	        def check(k):
	            _tasks = SortedList(tasks[:k])
	            _workers = workers[-k:]
	            _pills = pills
	            for worker in _workers:
	                task = _tasks[0]
	                if worker>=task:
	                    _tasks.pop(0)
	                elif worker+strength>=task and _pills:
	                    idx = _tasks.bisect_right(worker + strength)
	                    _tasks.pop(idx-1)
	                    _pills -= 1
	                else:
	                    return False
	            return True
	        
	        left,right = 0,min(len(tasks), len(workers))
	        while left<right:
	            mid = right - (right-left)//2
	            if check(mid):
	                left = mid
	            else:
	                right = mid-1
	        return left
	            		
### 运行结果

	Runtime: 5156 ms, faster than 87.50% of Python online submissions for Maximum Number of Tasks You Can Assign.
	Memory Usage: 21.4 MB, less than 80.36% of Python online submissions for Maximum Number of Tasks You Can Assign.
	
原题链接：https://leetcode.com/problems/maximum-number-of-tasks-you-can-assign/



您的支持是我最大的动力
