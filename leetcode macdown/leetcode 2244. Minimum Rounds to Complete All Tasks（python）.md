leetcode  2244. Minimum Rounds to Complete All Tasks（python）
这道题是第 289 场 leetcode 周赛的第二题，难度为 Medium ，主要考察的是就是对字符串和列表的基本操作。



### 描述

You are given a 0-indexed integer array tasks, where tasks[i] represents the difficulty level of a task. In each round, you can complete either 2 or 3 tasks of the same difficulty level.

Return the minimum rounds required to complete all the tasks, or -1 if it is not possible to complete all the tasks.

 




Example 1:


	Input: tasks = [2,2,3,3,2,4,4,4,4,4]
	Output: 4
	Explanation: To complete all the tasks, a possible plan is:
	- In the first round, you complete 3 tasks of difficulty level 2. 
	- In the second round, you complete 2 tasks of difficulty level 3. 
	- In the third round, you complete 3 tasks of difficulty level 4. 
	- In the fourth round, you complete 2 tasks of difficulty level 4.  
	It can be shown that all the tasks cannot be completed in fewer than 4 rounds, so the answer is 4.
	
Example 2:


	Input: tasks = [2,3,3]
	Output: -1
	Explanation: There is only 1 task of difficulty level 2, but in each round, you can only complete either 2 or 3 tasks of the same difficulty level. Hence, you cannot complete all the tasks, and the answer is -1.






Note:

	1 <= tasks.length <= 10^5
	1 <= tasks[i] <= 10^9


### 解析

根据题意，给定一个 0 索引整数数组 tasks，其中 tasks[i] 表示任务的难度级别。 在每一轮操作中，我们可以完成相同难度级别的 2 个或 3 个任务。返回完成所有任务所需的最少轮数，如果不可能完成所有任务，则返回 -1 。

这道题其实就是考察贪心并且考察解方程组的，思路很简单，我们对 tasks 中的各种任务及其出现的频率进行统计，都放入字典 c 中，然后我们遍历所有的 k-v 对：

* 如果 v 等于 1 ，则直接返回 -1 ，表示无法按照题意进行操作
* 否则我们就假设最多使用了 i 次“完 3 ”操作，那么我们只需要判断 (v - i * 3) % 2 是否等于 0 ，如果等于 0 说明“完 2 ”的操作使用了 (v - i * 3) // 2 次 ，那么将 v 个 k 使用最少次数完成则为 (v - i * 3) // 2 + i ，否则就将 i-1 进行下一轮的计算，
* 遍历所有的 k-v 对，计算结束直接返回 result 即可

时间复杂度为 O(N) ，空间复杂度为 O(N) 。 其实这里用到了双层循环，正常来说是 O(N^2) ，但是这里为什么是 O(N) ，没有超时呢，有兴趣的同学可以仔细思考一下。假如列表长度最大为限制条件 10^5 ，其中有两种不同的元素，那么每个元素有 5 * 10^4 个，第一层循环就是遍历 2 次，第二层循环最多遍历 5 * 10^4 //3 次，所有总共的两层 for 循环次数小于 10^5 ，所以尽管是两层 for ，但是其实时间复杂度就是 O(N) 。

### 解答
				
	class Solution(object):
	    def minimumRounds(self, tasks):
	        """
	        :type tasks: List[int]
	        :rtype: int
	        """
	        result = 0
	        c = collections.Counter(tasks)
	        for k,v in c.items():
	            if v == 1 :
	                return -1
	            for i in range(v//3, -1, -1):
	                if (v - i * 3) % 2 == 0:
	                    result += (v - i * 3) // 2 + i
	                    break
	        return result

            	      
			
### 运行结果



	78 / 78 test cases passed.
	Status: Accepted
	Runtime: 951 ms
	Memory Usage: 29.2 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-289/problems/minimum-rounds-to-complete-all-tasks/


您的支持是我最大的动力
