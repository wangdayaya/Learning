leetcode  2139. Minimum Moves to Reach Target Score（python）


由亚马逊公司赞助的 Leetcode Weekly Contest 276 ，优秀者还能获得亚马逊公司的面试机会（慕了），看了一下榜单第一名是个北大的选手，只用了 8 分钟就解完了 4 道题，佩服得六体投地【狗头】，吾辈榜样及楷模，估计我看完四道题目描述估计都得 8 分钟，惭愧。本文介绍的是周赛第二道题目，难度 Medium ，也不是很难，考察的是贪心，可以用递归也可以用常规的代码思路。


### 每日经典

《道德经》 ——老子（春秋）

道，可道，非常道。名，可名，非常名。无，名天地之始；有，名万物之母。故常无，欲以观其妙；常有，欲以观其徼。此两者同出而异名，同谓之玄。玄之又玄，众妙之门。

### 描述

You are playing a game with integers. You start with the integer 1 and you want to reach the integer target.

In one move, you can either:

* Increment the current integer by one (i.e., x = x + 1).
* Double the current integer (i.e., x = 2 * x).
* You can use the increment operation any number of times, 
however, you can only use the double operation at most maxDoubles times.

Given the two integers target and maxDoubles, return the minimum number of moves needed to reach target starting with 1.



Example 1:


	Input: target = 5, maxDoubles = 0
	Output: 4
	Explanation: Keep incrementing by 1 until you reach target.
	
Example 2:

	Input: target = 19, maxDoubles = 2
	Output: 7
	Explanation: Initially, x = 1
	Increment 3 times so x = 4
	Double once so x = 8
	Increment once so x = 9
	Double again so x = 18
	Increment once so x = 19


Example 3:

	Input: target = 10, maxDoubles = 4
	Output: 4
	Explanation: Initially, x = 1
	Increment once so x = 2
	Double once so x = 4
	Increment once so x = 5
	Double again so x = 10

	


Note:

	1 <= target <= 1^9
	0 <= maxDoubles <= 100


### 解析

根据题意，要玩一个整数游戏。 从整数 1 开始，要求最终到达整数 target 。每一步的操作我们可以按照如下规则：

* 将当前整数加一（即 x = x + 1）。
* 将当前整数加倍（即 x = 2 * x）。
* 可以使用任意次数的加一操作，但是最多只能使用 maxDoubles 次加倍操作。

给定两个整数 target 和 maxDoubles，返回从 1 开始到达 target 所需的最小操作次数。其实这道题就是考察贪心，我们用最朴素的想法肯定知道在 maxDoubles 满足的情况下，尽量多用加倍操作，最后的操作次数肯定是最小的，所以我这里直接用递归解题即可。

* 我们调用递归函数 dfs ，递归函数的设计是从 target 减为 1 的过程，这个和题目从 1 变为 target 的答案是一样的，只不过我觉得方便理解。递归终止的条件就是当 target 减为 1 的时候返回操作计数器 count 。
* 当 maxDoubles 大于 0 的时候，如果 target 为偶数可以使用减倍操作，同时 maxDoubles 加一， count 减一，直接调用递归 dfs(target // 2, maxDoubles - 1, count + 1) 
* 当 target 为奇数的时候只能用减一操作，同时 maxDoubles 保持不变但是 count + 1 ，直接调用递归 dfs(target - 1, maxDoubles, count + 1)
* 如果 maxDoubles 为 0 ，那么直接返回  count+target-1 即可


### 解答
				

	class Solution(object):
	    def minMoves(self, target, maxDoubles):
	        """
	        :type target: int
	        :type maxDoubles: int
	        :rtype: int
	        """
	        return self.dfs(target, maxDoubles, 0)
	
	    def dfs(self, target, maxDoubles, count):
	        if target == 1:
	            return count
	        if maxDoubles > 0:
	            if target % 2 == 0:
	                return self.dfs(target // 2, maxDoubles - 1, count + 1)
	            else:
	                return self.dfs(target - 1, maxDoubles, count + 1)
	        else:
	            return count+target-1
	        
            	      
			
### 运行结果

	Runtime: 19 ms
	Memory Usage: 13.8 MB


原题链接：https://leetcode.com/contest/weekly-contest-276/problems/minimum-moves-to-reach-target-score/



您的支持是我最大的动力
