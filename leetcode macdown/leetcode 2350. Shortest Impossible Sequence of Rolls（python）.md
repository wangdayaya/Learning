leetcode  2350. Shortest Impossible Sequence of Rolls（python）




### 描述
You are given an integer array rolls of length n and an integer k. You roll a k sided dice numbered from 1 to k, n times, where the result of the ith roll is rolls[i].

Return the length of the shortest sequence of rolls that cannot be taken from rolls.

A sequence of rolls of length len is the result of rolling a k sided dice len times.

Note that the sequence taken does not have to be consecutive as long as it is in order.




Example 1:

	Input: rolls = [4,2,1,2,3,3,2,4,1], k = 4
	Output: 3
	Explanation: Every sequence of rolls of length 1, [1], [2], [3], [4], can be taken from rolls.
	Every sequence of rolls of length 2, [1, 1], [1, 2], ..., [4, 4], can be taken from rolls.
	The sequence [1, 4, 2] cannot be taken from rolls, so we return 3.
	Note that there are other sequences that cannot be taken from rolls.

	
Example 2:

	Input: rolls = [1,1,2,2], k = 2
	Output: 2
	Explanation: Every sequence of rolls of length 1, [1], [2], can be taken from rolls.
	The sequence [2, 1] cannot be taken from rolls, so we return 2.
	Note that there are other sequences that cannot be taken from rolls but [2, 1] is the shortest.


Example 3:


	Input: rolls = [1,1,3,2,2,2,3,3], k = 4
	Output: 1
	Explanation: The sequence [4] cannot be taken from rolls, so we return 1.
	Note that there are other sequences that cannot be taken from rolls but [4] is the shortest.


Note:

	n == rolls.length
	1 <= n <= 10^5
	1 <= rolls[i] <= k <= 10^5


### 解析

根据题意，给定一个长度为 n 的整数数组 rolls 和一个整数 k 。 掷一个从 1 到 k 的 k 面骰子 n 次，其中第 i 次掷骰的结果是 rolls[i]。返回不能从摇骰结果 rolls 中生成的最短序列的长度。一个长度为 len 的掷骰序列是 k 面骰子掷 len 次的结果。需要注意的是所取的序列不必是连续的，只要它是按相对顺序排列的即可。

对于这个序列的第一个数字，肯定是 1-k 中的某个数字，对于第二个数字，肯定是在有了第一个数字基础之上随机选择 1-k 中的某个数字，这样才能保证所有长度为 2 的摇骰结果序列都在 rolls 中存在，对于第三个数字，肯定是在有了前两个数字的基础之上随机选择 1-k 中的某个数字，这样才能保证所有的长度为 3 的摇骰结果序列都在 rolls 中存在，以此类推。所以我们发现只要从做往右遍历 rolls ， 1-k 所有数字都出现过一次，就能满足我们摇出长度为 1 的序列，然后我们再继续往后找出 1-k 又一次都能，就能满足我们摇出长度为 2 的序列，以此类推，只要我们找出 n 次 1-k 都出现过的集合，那么 n 就是我们能从 rolls 中找出的最短序列长度。

时间复杂度为 O(N) ，空间复杂度为 O(K) 。


### 解答

	class Solution(object):
	    def shortestSequence(self, rolls, k):
	        """
	        :type rolls: List[int]
	        :type k: int
	        :rtype: int
	        """
	        result = 1
	        s = set()
	        for n in rolls:
	            s.add(n)
	            if len(s) == k:
	                result += 1
	                s.clear()
	        return result

### 运行结果

	
	65 / 65 test cases passed.
	Status: Accepted
	Runtime: 800 ms
	Memory Usage: 27.6 MB

### 原题链接

	https://leetcode.com/contest/biweekly-contest-83/problems/shortest-impossible-sequence-of-rolls/


您的支持是我最大的动力
