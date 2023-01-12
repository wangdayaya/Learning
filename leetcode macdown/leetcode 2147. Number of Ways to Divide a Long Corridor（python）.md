leetcode 2147. Number of Ways to Divide a Long Corridor （python）

### 前言


我比赛的时候没有完成这道题，我看榜单前面的大佬们解决这道用了四、五分钟的样子，这就是我和大佬的差距。其实当时已经半夜了，对于我这种平时 10 点多就躺下睡觉的人，实在是有些难熬的，脑袋昏昏沉沉的，已经肝不动了，等第二天有时间了自己看了题目也做出来，耗时半个小时。这是 Biweekly Contest 70  的第四题，难度 Hard ，考察的是对题目的理解能力，其实就是找规律 。


### 描述


Along a long library corridor, there is a line of seats and decorative plants. You are given a 0-indexed string corridor of length n consisting of letters 'S' and 'P' where each 'S' represents a seat and each 'P' represents a plant.

One room divider has already been installed to the left of index 0, and another to the right of index n - 1. Additional room dividers can be installed. For each position between indices i - 1 and i (1 <= i <= n - 1), at most one divider can be installed.

Divide the corridor into non-overlapping sections, where each section has exactly two seats with any number of plants. There may be multiple ways to perform the division. Two ways are different if there is a position with a room divider installed in the first way but not in the second way.

Return the number of ways to divide the corridor. Since the answer may be very large, return it modulo 10^9 + 7. If there is no way, return 0.


Example 1:


![](https://assets.leetcode.com/uploads/2021/12/04/1.png)

	Input: corridor = "SSPPSPS"
	Output: 3
	Explanation: There are 3 different ways to divide the corridor.
	The black bars in the above image indicate the two room dividers already installed.
	Note that in each of the ways, each section has exactly two seats.
	

Note:

	n == corridor.length
	1 <= n <= 10^5
	corridor[i] is either 'S' or 'P'.


### 解析


根据题意，走廊有一排座椅和装饰植物。 给定一个长度为 n 的 0 索引字符串 corridor ，由字母 “S” 和 “P” 组成，其中每个 “S” 代表一个座位，每个 “P” 代表一个植物。如例子一，已在索引 0 的左侧安装了一个隔断板，在索引 n - 1 的右侧安装了隔断板。另外对于索引 i - 1 和 i (1 <= i <= n - 1) 之间的每个位置中间，最多可以安装一个分隔器。

题目要求将走廊分成若干个部分，每个部分正好有两个座位，可以种植任意数量的植物。 可以有多种方式来安排。返回划分走廊的方式数。 由于答案可能非常大，所以以 10^9 + 7 为模返回。如果没有办法，则返回 0 。

Hard 难度的题其实就是这样，猛的一看很难，细想一下觉得有了思路可以解题了，但是写代码的时候又觉得越来越难，说明还是对题目的理解不深。看了大佬的解答，真的是鞭辟入里，简洁易懂。我们先初始化一个列表 a ，里面存放着 corridor 中 S 的索引。其实对于对于 a[i] 位置的座位与 a[i+1] 位置的座位中间能有几种放隔板的方式，就是  a[i+1] -  a[i]  的值，因为我们已经对 a[i] 位置的座位与 a[i+1] 位置的座位进行了方案的设置，所以需要将 i 向后移动两位，去计算接下来的两个座位之间隔板的放置方案数量，每次将这些方案数量值乘起来，就是最后可能存在的放置隔板的方式。

需要注意的是在 corridor 中当 S 的个数为奇数，或者小于 2 的时候，是无法按照题意放置隔板的，所以只能直接返回 0 。对于我们乘积的结果可能会很大，所以要对 10^9 + 7 进行取模运算。


### 解答
				
	class Solution(object):
	    def numberOfWays(self, corridor):
	        """
	        :type corridor: str
	        :rtype: int
	        """
	        a = [i for i,c in enumerate(corridor) if c == 'S']
	        res = 1
	        for i in xrange(1, len(a) - 1, 2):
	            res *= a[i+1] - a[i]
	        return res % (10**9+7) * (len(a) % 2 == 0 and len(a) >= 2)  
	            
	        

            	      
			
### 运行结果


	248 / 248 test cases passed.
	Status: Accepted
	Runtime: 600 ms
	Memory Usage: 18.2 MB

### 原题链接



https://leetcode.com/contest/biweekly-contest-70/problems/number-of-ways-to-divide-a-long-corridor/


您的支持是我最大的动力
