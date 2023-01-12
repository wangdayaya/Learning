leetcode  2365. Task Scheduler II（python）




### 描述

You are given a 0-indexed array of positive integers tasks, representing tasks that need to be completed in order, where tasks[i] represents the type of the ith task. You are also given a positive integer space, which represents the minimum number of days that must pass after the completion of a task before another task of the same type can be performed. Each day, until all tasks have been completed, you must either:

* Complete the next task from tasks, or
* Take a break.

Return the minimum number of days needed to complete all tasks.



Example 1:


	Input: tasks = [1,2,1,2,3,1], space = 3
	Output: 9
	Explanation:
	One way to complete all tasks in 9 days is as follows:
	Day 1: Complete the 0th task.
	Day 2: Complete the 1st task.
	Day 3: Take a break.
	Day 4: Take a break.
	Day 5: Complete the 2nd task.
	Day 6: Complete the 3rd task.
	Day 7: Take a break.
	Day 8: Complete the 4th task.
	Day 9: Complete the 5th task.
	It can be shown that the tasks cannot be completed in less than 9 days.
	
Example 2:

	Input: tasks = [5,8,8,5], space = 2
	Output: 6
	Explanation:
	One way to complete all tasks in 6 days is as follows:
	Day 1: Complete the 0th task.
	Day 2: Complete the 1st task.
	Day 3: Take a break.
	Day 4: Take a break.
	Day 5: Complete the 2nd task.
	Day 6: Complete the 3rd task.
	It can be shown that the tasks cannot be completed in less than 6 days.



Note:


	1 <= tasks.length <= 10^5
	1 <= tasks[i] <= 10^9
	1 <= space <= tasks.length

### 解析

根据题意，给定一个 0 索引的正整数数组 tasks ，表示需要按顺序完成的任务，其中 tasks[i] 表示第 i 个任务的类型。 另外还给到一个正整数 space ，它表示在完成某一项任务后必须休息的最少天数，然后才能执行另一项相同类型的任务。 在完成所有任务之前，可以有以下操作：

* 执行任务，或者
* 由于 space 条件限制被迫休息

返回完成所有任务所需的最少天数。

其实这道题我们直接按照题意进行模拟过程即可。我们需要一个字典 d 来记录每类任务能够进行的最早日期，初始都为 0 ，我们定义最终需要的天数为 result ，然后遍历每个任务 t ，有两种情况：

* 如果当 d[t] 小于等于 result 的时候，说明当天我们可以进行任务，此时需要更新 d[t] 为下一个能够执行此类任务的最早时间，result 加一，然后去执行后一个任务
* 如果 d[t] 大于 result ，说明当天无法执行任务，需要休息，只需要等待 result 不断增加 即可，直到增大到某一天满足可以执行任务的条件为止，

时间复杂度为 O(N) ，空间复杂度为 O(N)。

### 解答

	class Solution(object):
	    def taskSchedulerII(self, tasks, space):
	        """
	        :type tasks: List[int]
	        :type space: int
	        :rtype: int
	        """
	        d = collections.defaultdict(int)
	        result = 0
	        i = 0
	        while i < len(tasks):
	            t = tasks[i]
	            if d[t] <= result:
	                d[t] = result + space + 1
	                result += 1
	                i += 1
	            elif d[t] > result:
	                result += d[t] - result
	        return result
### 运行结果

	61 / 61 test cases passed.
	Status: Accepted
	Runtime: 1747 ms
	Memory Usage: 25.8 MB

### 原题链接


https://leetcode.com/contest/biweekly-contest-84/problems/task-scheduler-ii/

您的支持是我最大的动力
