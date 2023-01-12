leetcode  1560. Most Visited Sector in a Circular Track（python）

### 描述

Given an integer n and an integer array rounds. We have a circular track which consists of n sectors labeled from 1 to n. A marathon will be held on this track, the marathon consists of m rounds. The i<sup>th</sup> round starts at sector rounds[i - 1] and ends at sector rounds[i]. For example, round 1 starts at sector rounds[0] and ends at sector rounds[1]

Return an array of the most visited sectors sorted in ascending order.

Notice that you circulate the track in ascending order of sector numbers in the counter-clockwise direction (See the first example).





Example 1:

![](https://assets.leetcode.com/uploads/2020/08/14/tmp.jpg)
	
	Input: n = 4, rounds = [1,3,1,2]
	Output: [1,2]
	Explanation: The marathon starts at sector 1. The order of the visited sectors is as follows:
	1 --> 2 --> 3 (end of round 1) --> 4 --> 1 (end of round 2) --> 2 (end of round 3 and the marathon)
	We can see that both sectors 1 and 2 are visited twice and they are the most visited sectors. Sectors 3 and 4 are visited only once.

	
Example 2:
	
	
	Input: n = 2, rounds = [2,1,2,1,2,1,2,1,2]
	Output: [2]

Example 3:

	Input: n = 7, rounds = [1,3,5,7]
	Output: [1,2,3,4,5,6,7]

	

Note:

	2 <= n <= 100
	1 <= m <= 100
	rounds.length == m + 1
	1 <= rounds[i] <= n
	rounds[i] != rounds[i + 1] for 0 <= i < m


### 解析


根据题意，给定一个整数 n 和一个整数数组 rounds 。 我们有一个圆形赛道，它由 n 个标记为 1 到 n 的扇区组成。 马拉松将在这条赛道上举行，马拉松要逆时针跑 m 轮。 第 i 轮从扇区 rounds[i - 1] 开始，到扇区 rounds[i] 结束。 例如，第 1 轮从扇区 rounds[0] 开始并在扇区 rounds[1] 结束，题目要求我们返回按升序排序访问量最大的扇区数组。

其实思路很简单，就是逆时针在围绕赛道跑，找出经过最多次数的扇区，我们可以将这条圆形赛道拉直看成是条直线赛道，最多可能跑的轮数是 len(rounds) ，假如我们 n=4 , rounds = [2,3,1,2] ，我们可以将赛道变成如下形式：

	[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]

这样我们先找到起始位置 2，并将 2 前面的数字都去掉，此时赛道变成：

	[2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]

此时我们只需要找出经过 2 、3 、1 、2 四个数字的次数即可，此时为：
	
	1 经过了 1 次
	2 经过了 2 次
	3 经过了 1 次
	4 经过了 1 次
最后找出经过次数最多的数字组成列表，升序返回即可。


### 解答
				

	class Solution(object):
	    def mostVisited(self, n, rounds):
	        """
	        :type n: int
	        :type rounds: List[int]
	        :rtype: List[int]
	        """
	        L = [i for i in range(1, n + 1)] * (len(rounds))
	        d = {i: 0 for i in range(1, n + 1)}
	        for i in range(L.index(rounds[0])):
	            L.pop(0)
	        for pos in rounds[1:]:
	            i = L.index(pos)
	            for j in range(i + 1):
	                d[L.pop(0)] += 1
	        result = [k for k, v in d.items() if v == max(d.values())]
	        result.sort()
	        return result
            	      
			
### 运行结果

	
	Runtime: 168 ms, faster than 8.82% of Python online submissions for Most Visited Sector in a Circular Track.
	Memory Usage: 13.6 MB, less than 8.82% of Python online submissions for Most Visited Sector in a Circular Track.

### 解析

其实像上面这种模拟的方法是比较笨拙的，我看了高手的解法，只需要找规律发现访问最多频次的扇区只与起/终点位置有关，经过简单的计算就可以找到答案。

### 解答

	class Solution(object):
	    def mostVisited(self, n, rounds):
	        """
	        :type n: int
	        :type rounds: List[int]
	        :rtype: List[int]
	        """
	        start=rounds[0]
	        end=rounds[-1]
	        result=[]
	        if(start>end):
	            for i in range(end):
	                result.append(i+1)
	            for i in range(start,n+1):
	                result.append(i)
	        else:
	            for i in range(start,end+1):
	                result.append(i)
	        return result
	



### 运行结果

	Runtime: 32 ms, faster than 67.65% of Python online submissions for Most Visited Sector in a Circular Track.
	Memory Usage: 13.3 MB, less than 94.12% of Python online submissions for Most Visited Sector in a Circular Track.
	
原题链接：https://leetcode.com/problems/most-visited-sector-in-a-circular-track/



您的支持是我最大的动力
