leetcode  2141. Maximum Running Time of N Computers（python）

由亚马逊公司赞助的 Leetcode Weekly Contest 276 ，看了一下榜单第一名是位北大的选手，为国争光，只要第一名国籍是中国的就是好样的，吾辈楷模，我的水平令人惭愧，制作出两道，第三道超时了。本文介绍的是 276 周赛第四道题目，难度 Hard ，我比赛时还没看到这个题目，卡到第三个上面了，这个题纯考查对题目理解，如果理解到位代码很简单，理解不到位代码又臭又长还不一定过。



### 描述


You have n computers. You are given the integer n and a 0-indexed integer array batteries where the i<sub>th</sub> battery can run a computer for batteries[i] minutes. You are interested in running all n computers simultaneously using the given batteries.

Initially, you can insert at most one battery into each computer. After that and at any integer time moment, you can remove a battery from a computer and insert another battery any number of times. The inserted battery can be a totally new battery or a battery from another computer. You may assume that the removing and inserting processes take no time.

Note that the batteries cannot be recharged.

Return the maximum number of minutes you can run all the n computers simultaneously.


Example 1:

![](https://assets.leetcode.com/uploads/2022/01/06/example1-fit.png)


	Input: n = 2, batteries = [3,3,3]
	Output: 4
	Explanation: 
	Initially, insert battery 0 into the first computer and battery 1 into the second computer.
	After two minutes, remove battery 1 from the second computer and insert battery 2 instead. Note that battery 1 can still run for one minute.
	At the end of the third minute, battery 0 is drained, and you need to remove it from the first computer and insert battery 1 instead.
	By the end of the fourth minute, battery 1 is also drained, and the first computer is no longer running.
	We can run the two computers simultaneously for at most 4 minutes, so we return 4.



	
Example 2:

![](https://assets.leetcode.com/uploads/2022/01/06/example2.png)

	Input: n = 2, batteries = [1,1,1,1]
	Output: 2
	Explanation: 
	Initially, insert battery 0 into the first computer and battery 2 into the second computer. 
	After one minute, battery 0 and battery 2 are drained so you need to remove them and insert battery 1 into the first computer and battery 3 into the second computer. 
	After another minute, battery 1 and battery 3 are also drained so the first and second computers are no longer running.
	We can run the two computers simultaneously for at most 2 minutes, so we return 2.




Note:

	
	1 <= n <= batteries.length <= 10^5
	1 <= batteries[i] <= 10^9

### 解析

根据题意，现在有 n 台电脑， 给定整数 n 和一个索引为 0 的整数数组 batteries ，其中第 i 个电池可以让计算机运行 batteries[i] 分钟。 我们要使用给定的电池同时让所有 n 台电脑保持运行（因为没电就关机不运行了，所有只要有电就行）。开始的时候最多可以为每台计算机插一块电池。 之后在任何整数时刻，我们都可以从计算机中拔出电池并插入另一块电池。 插入的电池可以是全新的电池，也可以是另一台计算机用过的还有电的电池。 假设拔出和插入过程不需要时间，电脑也不会因拔出电池关机。而且整个过程中电池无法充电，只能一直消耗。返回可以同时运行所有 n 台计算机的最大分钟数。

其实事后看这道题，感觉不是很难，我这里找到一张图

![](https://assets.leetcode.com/users/images/fbb78517-2c87-405a-96d0-a28204b25de7_1642308671.3179777.png)

这就是一个最朴素的思路，将 batteries 降序排序，然后将 batteries[n:] 子数组中的电池的电量都分摊到前 n 个电脑，就像一杯水倒入前 n 个电池的池塘中，这样就能保证前 n 台电脑的最低工作时间会最高，但是我这里的代码有一个问题，那就是用了两次 for 循环，导致超时，因为条件限制 batteries.length 最大为 10^5 ，所以 O(n^2) 的解法肯定会超时。

### 解答
				

	class Solution(object):
		    def maxRunTime(self, n, batteries):
		        """
		        :type n: int
		        :type batteries: List[int]
		        :rtype: int
		        """
		        if n>len(batteries): return 0
		        batteries.sort(reverse=True)
		        b = batteries[:n]
		        t = batteries[n:]
		        for i in range(len(t)-1, -1, -1):
		            for _ in range(t[i]):
		                idx = b.index(min(b))
		                b[idx] += 1
		        return min(b)
			
### 运行结果


	Time Limit Exceeded


### 解析

这种 Hard 题就是这样，看起来容易做起来难，我看了点赞最高的大佬解法，才有所启发。我们用最朴素的想法就是将所有的电量都分摊给 n 个电脑，找到最小的工作时长 sum(batteries) // n 分钟即为答案。但是像这种情况 n = 2, batteries = [3,3,9] ，结果得到 7 是错误的答案。因为第三块电池多出来的电量只能供给一个电脑使用，不能同时供给两台电脑。所以我们比较平均值和最大电量的电池，如果最大电量的电池比平均值大，我们就用这个电池一直给某个电脑充电，接下来我们只需要考虑剩下的 n-1 台电脑和剩下的若干块电池进行比较。一直循环这个过程，直到剩余电池中的最大电量不大于剩余电池的平均电量，那么说明剩下的电池都可以均匀分摊使用。


### 解答


	class Solution(object):
	    def maxRunTime(self, n, batteries):
	        """
	        :type n: int
	        :type batteries: List[int]
	        :rtype: int
	        """
	        if n == 1 : return sum(batteries)
	        batteries.sort()
	        total = sum(batteries)
	        while batteries[-1] > total / n:
	            n -= 1
	            total -= batteries.pop()
	        return total / n

### 运行结果
	
	Runtime: 548 ms, faster than 100.00% of Python online submissions for Maximum Running Time of N Computers.
	Memory Usage: 25.3 MB, less than 100.00% of Python online submissions for Maximum Running Time of N Computers.


### 原题链接

https://leetcode.com/contest/weekly-contest-276/problems/maximum-running-time-of-n-computers/


### 每日经典

《道德经》 ——老子（春秋）

天下皆知美之为美，恶已；皆知善，斯不善矣。有无之相生也，难易之相成也，长短之相刑也，高下之相盈也，音声之相和也，先后之相随，恒也。是以圣人居无为之事，行不言之教，万物作而弗始也，为而弗志也，成功而弗居也。夫唯弗居，是以弗去。

您的支持是我最大的动力
