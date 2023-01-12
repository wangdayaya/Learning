leetcode  1834. Single-Threaded CPU（python）




### 描述


You are given n​​​​​​ tasks labeled from 0 to n - 1 represented by a 2D integer array tasks, where tasks[i] = [enqueueTime<sub>i</sub>, processingTime<sub>i</sub>] means that the i<sup>​​​​​​th​​​​</sup>​​​​ task will be available to process at enqueueTime<sub>​​​​i</sub>​​​​ and will take processingTime<sub>​​​​i</sub>​​​​ to finish processing. You have a single-threaded CPU that can process at most one task at a time and will act in the following way:

* If the CPU is idle and there are no available tasks to process, the CPU remains idle.
* If the CPU is idle and there are available tasks, the CPU will choose the one with the shortest processing time. If multiple tasks have the same shortest processing time, it will choose the task with the smallest index.
* Once a task is started, the CPU will process the entire task without stopping.
* The CPU can finish a task then start a new one instantly.

Return the order in which the CPU will process the tasks.


Example 1:

	Input: tasks = [[1,2],[2,4],[3,2],[4,1]]
	Output: [0,2,3,1]
	Explanation: The events go as follows: 
	- At time = 1, task 0 is available to process. Available tasks = {0}.
	- Also at time = 1, the idle CPU starts processing task 0. Available tasks = {}.
	- At time = 2, task 1 is available to process. Available tasks = {1}.
	- At time = 3, task 2 is available to process. Available tasks = {1, 2}.
	- Also at time = 3, the CPU finishes task 0 and starts processing task 2 as it is the shortest. Available tasks = {1}.
	- At time = 4, task 3 is available to process. Available tasks = {1, 3}.
	- At time = 5, the CPU finishes task 2 and starts processing task 3 as it is the shortest. Available tasks = {1}.
	- At time = 6, the CPU finishes task 3 and starts processing task 1. Available tasks = {}.
	- At time = 10, the CPU finishes task 1 and becomes idle.

	
Example 2:

	Input: tasks = [[7,10],[7,12],[7,5],[7,4],[7,2]]
	Output: [4,3,2,0,1]
	Explanation: The events go as follows:
	- At time = 7, all the tasks become available. Available tasks = {0,1,2,3,4}.
	- Also at time = 7, the idle CPU starts processing task 4. Available tasks = {0,1,2,3}.
	- At time = 9, the CPU finishes task 4 and starts processing task 3. Available tasks = {0,1,2}.
	- At time = 13, the CPU finishes task 3 and starts processing task 2. Available tasks = {0,1}.
	- At time = 18, the CPU finishes task 2 and starts processing task 0. Available tasks = {1}.
	- At time = 28, the CPU finishes task 0 and starts processing task 1. Available tasks = {}.
	- At time = 40, the CPU finishes task 1 and becomes idle.




Note:


* tasks.length == n
* 1 <= n <= 10^5
* 1 <= enqueueTime<sub>i</sub>, processingTime<sub>i</sub> <= 10^9

### 解析

根据题意，给定 n 个标记为 0 到 n - 1  的二维整数数组 tasks ，其中 tasks[i] = [enqueueTime<sub>i</sub>, processingTime<sub>i</sub>] 表示第 i 个任务将在  enqueueTime<sub>i</sub> 时开始处理，并且需要 processingTime<sub>i</sub>  个时间单位才能完成处理。我们有一个单线程 CPU ，一次最多可以处理一个任务，并将按以下方式操作，返回 CPU 处理所有任务的顺序。：

* 如果 CPU 处于空闲状态，并且没有待处理的任务，则 CPU 将保持空闲状态。
* 如果 CPU 处于空闲状态并且有待处理的任务，CPU 将选择耗时最短的任务去执行。如果多个任务具有相同的耗时，CPU 将选择具有最小索引的任务。
* 一旦任务启动，CPU 将不会中断，一直处理完整个任务。
* CPU 可以完成一项任务，然后立即启动一项新任务。


因为整个任务执行过程，是按照三个维度来进行的，先要判断是否是任务的开始时间是否靠前，再判断是否耗时较少，如果耗时相同，则优先进行索引较小的任务。所以我们可以转换成两个子任务，第一个就是如何按照时间先后顺序将任务分配给 CPU ，第二个就是找出耗时最少的索引靠前的任务交给 CPU 。

首先我们定义一个索引数组 idxs ，让其按照任务的开始时间来将任务的索引保存下来，并且使用一个指针 position 去指向需要执行的下一个任务。同时为了使用 timestamp 来记录当前时间。使用小根堆可以将所有接收到的任务按照耗时和索引进行排序，进行 N 次的遍历不断重复下面过程：

* 当 heap 为空的的时候，我们直接将指针移动到 position 对应的任务开始时间
* 如果存在有任务开始时间小于等于当前时间 timestamp ，则说明这些任务是待运行的任务，将他们都按照 [耗时，索引] 的元组形式加入到小根堆中，并且不断后移 position 。
* 我们从优先队列中弹出一个耗时最少，索引靠前的任务，更新时间戳 timestamp ，并且将其索引加入结果数组 result 中。

经过 N 次遍历最终会得到一个长度为 N 的索引数组。时间复杂度为 O(NlogN+NlogN) ，主要是消耗在了索引排序和小根堆的建立，空间复杂度为 O(N) 。



### 解答

	class Solution(object):
	    def getOrder(self, tasks):
	        """
	        :type tasks: List[List[int]]
	        :rtype: List[int]
	        """
	        heap = []
	        result = []
	        position = 0
	        timestamp = 0
	        N = len(tasks)
	        idxs = list(range(N))
	        idxs.sort(key=lambda x: tasks[x][0])
	        for i in range(N):
	            if not heap:
	                timestamp = max(timestamp, tasks[idxs[position]][0])
	            while position < N and tasks[idxs[position]][0] <= timestamp:
	                heapq.heappush(heap, [tasks[idxs[position]][1], idxs[position]])
	                position += 1
	            process_time, idx = heapq.heappop(heap)
	            timestamp += process_time
	            result.append(idx)
	        return result

### 运行结果

* Runtime 1884 ms ， Beats 94.44%
* Memory 64.2 MB ，Beats 50%

### 原题链接
https://leetcode.com/problems/single-threaded-cpu/

您的支持是我最大的动力
