


### 描述


You are given the logs for users' actions on LeetCode, and an integer k. The logs are represented by a 2D integer array logs where each logs[i] = [ID<sub>i</sub>, time<sub>i</sub>] indicates that the user with ID<sub>i</sub> performed an action at the minute time<sub>i</sub>.

Multiple users can perform actions simultaneously, and a single user can perform multiple actions in the same minute.

The user active minutes (UAM) for a given user is defined as the number of unique minutes in which the user performed an action on LeetCode. A minute can only be counted once, even if multiple actions occur during it.

You are to calculate a 1-indexed array answer of size k such that, for each j (1 <= j <= k), answer[j] is the number of users whose UAM equals j.

Return the array answer as described above.
        


Example 1:

	Input: logs = [[0,5],[1,2],[0,2],[0,5],[1,3]], k = 5
	Output: [0,2,0,0,0]
	Explanation:
	The user with ID=0 performed actions at minutes 5, 2, and 5 again. Hence, they have a UAM of 2 (minute 5 is only counted once).
	The user with ID=1 performed actions at minutes 2 and 3. Hence, they have a UAM of 2.
	Since both users have a UAM of 2, answer[2] is 2, and the remaining answer[j] values are 0.

	
Example 2:

	Input: logs = [[1,1],[2,2],[2,3]], k = 4
	Output: [1,1,0,0]
	Explanation:
	The user with ID=1 performed a single action at minute 1. Hence, they have a UAM of 1.
	The user with ID=2 performed actions at minutes 2 and 3. Hence, they have a UAM of 2.
	There is one user with a UAM of 1 and one with a UAM of 2.
	Hence, answer[1] = 1, answer[2] = 1, and the remaining values are 0.






Note:

* 1 <= logs.length <= 10^4
* 0 <= ID<sub>i</sub> <= 10^9
* 1 <= time<sub>i</sub> <= 10^5
* k is in the range [The maximum UAM for a user, 10^5].


### 解析

根据题意，题目中给出了一个二维整数列表 logs 表示 leetcode 用户的操作日志，logs 格式为 logs[i] = [ID<sub>i</sub>, time<sub>i</sub>] 表示 ID<sub>i</sub> 的用户在第 time<sub>i</sub> 分钟时执行了一个操作。

多个用户可以同时执行操作，并且一个用户可以在同一分钟内执行多个操作。

用户活跃分钟数 (UAM) 的定义为用户在 LeetCode 上执行操作的不同分钟数字。 同一分钟就算发生了多次操作但也只算一个分钟数字。

要求最后返回一个大小为 k 从 1 开始索引的列表 answer，对于每个 j (1 <= j <= k)， answer[j] 是其 UAM 等于 j 的用户数。

这种题看起来很复杂，其实都是纸老虎，虽然题目中给出的条件限制的数量级很大，但是还是可以借用字典尝试用暴力破：
* 初始化一个 k 大小的列表
* 用字典 d 来存储每个用户对应的不同的分钟数字
* 然后遍历 d 中的每个键值对 k,v ，执行 result[len(v)-1] += 1 ，通过这一步就可以把所有 UAM 都为 j 的用户都统计到 result 中对应的位置中
* 最后得到的 result 即为答案




### 解答
				
	class Solution(object):
	    def findingUsersActiveMinutes(self, logs, k):
	        """
	        :type logs: List[List[int]]
	        :type k: int
	        :rtype: List[int]
	        """
	        result = [0] * k
	        d = {}
	        for i, j in logs:
	            if i not in d:
	                d[i] = [j]
	            elif i in d and j not in d[i]:
	                d[i].append(j)
	        for k,v in d.items():
	            result[len(v)-1] += 1
	        return result

            	      
			
### 运行结果

	Runtime: 1400 ms, faster than 19.23% of Python online submissions for Finding the Users Active Minutes.
	Memory Usage: 21.6 MB, less than 85.38% of Python online submissions for Finding the Users Active Minutes.


原题链接：https://leetcode.com/problems/finding-the-users-active-minutes/



您的支持是我最大的动力
