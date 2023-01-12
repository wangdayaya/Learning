leetcode  2335. Minimum Amount of Time to Fill Cups（python）




### 描述



You have a water dispenser that can dispense cold, warm, and hot water. Every second, you can either fill up 2 cups with different types of water, or 1 cup of any type of water.

You are given a 0-indexed integer array amount of length 3 where amount[0], amount[1], and amount[2] denote the number of cold, warm, and hot water cups you need to fill respectively. Return the minimum number of seconds needed to fill up all the cups.

 

Example 1:

	Input: amount = [1,4,2]
	Output: 4
	Explanation: One way to fill up the cups is:
	Second 1: Fill up a cold cup and a warm cup.
	Second 2: Fill up a warm cup and a hot cup.
	Second 3: Fill up a warm cup and a hot cup.
	Second 4: Fill up a warm cup.
	It can be proven that 4 is the minimum number of seconds needed.

	
Example 2:


	Input: amount = [5,4,4]
	Output: 7
	Explanation: One way to fill up the cups is:
	Second 1: Fill up a cold cup, and a hot cup.
	Second 2: Fill up a cold cup, and a warm cup.
	Second 3: Fill up a cold cup, and a warm cup.
	Second 4: Fill up a warm cup, and a hot cup.
	Second 5: Fill up a cold cup, and a hot cup.
	Second 6: Fill up a cold cup, and a warm cup.
	Second 7: Fill up a hot cup.

Example 3:



	Input: amount = [5,0,0]
	Output: 5
	Explanation: Every second, we fill up a cold cup.

Note:

	amount.length == 3
	0 <= amount[i] <= 100
### 解析

根据题意，有一个饮水机，可以分配冷水、温水和热水。 每一秒可以装满 2 杯不同类型的水，或 1 杯任何类型的水。给定一个长度为 3 的 0 索引整数数组  amount ，其中 amount[0], amount[1], and amount[2] 分别表示需要填充的冷水杯、温水杯和热水杯的数量。 返回填满所有杯子所需的最少秒数。

这道题考查的就是贪心思想，我们一开始肯定是尽量多次数的进行每秒装 2 杯不同类型的水，然后再进行每秒装 1 杯任何类型的水的操作。所以我们一开始对 amount 进行排序，然后循环当 amount[-2] > 0 的时候，amount[-1] 肯定大于 0 ，所以可以将他们各自的数量减一，然后重新对 amount 进行排序，直到不满足条件之后，肯定最后只剩下 amount[-1] 还有残留的杯子，我们直接加到结果里就可以了。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。
### 解答

	class Solution(object):
	    def fillCups(self, amount):
	        """
	        :type amount: List[int]
	        :rtype: int
	        """
	        result = 0
	        amount.sort()
	        while amount[-2] > 0:
	            amount[-1] -= 1
	            amount[-2] -= 1
	            result += 1
	            amount.sort()
	        result += amount[-1]
	        return result

### 运行结果

	280 / 280 test cases passed.
	Status: Accepted
	Runtime: 42 ms
	Memory Usage: 13.2 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-301/problems/minimum-amount-of-time-to-fill-cups/


您的支持是我最大的动力
