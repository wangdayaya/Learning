leetcode  2400. Number of Ways to Reach a Position After Exactly k Steps（python）




### 描述

You are given two positive integers startPos and endPos. Initially, you are standing at position startPos on an infinite number line. With one step, you can move either one position to the left, or one position to the right.

Given a positive integer k, return the number of different ways to reach the position endPos starting from startPos, such that you perform exactly k steps. Since the answer may be very large, return it modulo 109 + 7.

Two ways are considered different if the order of the steps made is not exactly the same.

Note that the number line includes negative integers.



Example 1:


	Input: startPos = 1, endPos = 2, k = 3
	Output: 3
	Explanation: We can reach position 2 from 1 in exactly 3 steps in three ways:
	- 1 -> 2 -> 3 -> 2.
	- 1 -> 2 -> 1 -> 2.
	- 1 -> 0 -> 1 -> 2.
	It can be proven that no other way is possible, so we return 3.
	
Example 2:

	Input: startPos = 2, endPos = 5, k = 10
	Output: 0
	Explanation: It is impossible to reach position 5 from position 2 in exactly 10 steps.




Note:


	1 <= startPos, endPos, k <= 1000

### 解析

根据题意，给定两个正整数 startPos 和 endPos。 最初站在无限长数轴上的 startPos 位置。 执行一次操作可以向左移动一个位置，或向右移动一个位置。给定一个正整数 k ，在执行 k 次操作情况下，返回从 startPos 到达 endPos 位置的不同方式的数量。 由于答案可能非常大，因此以 10^9 + 7 为模返回。如果所进行的步骤的顺序不完全相同，则两种方式被认为是不同的。请注意，数字行包括负整数。

其实这道题就是考察我们 DFS ，因为从起始位置开始，我们就可以向左一位或者向右一位，不断向下分裂，我们自己可以画一个图一目了然，其实就是一个 top-down 的二叉树结构，一旦出现二叉树结构的题我们最方便的解法就是 DFS ，我们定义函数 dfs ，表示当前到了 s 点并且还剩下 left 次操作的方案总数有多少，这自然是 dfs(s-1, left-1) 与 dfs(s+1, left-1) 的和，推出条件就是当 left 为 0 的时候表示次数用完，此时如果 s 等于 endPos 说明正确到达终点，是一种成功的方案返回 1 ，否则表示此方案无效返回 0 。

需要注意的时避免数值太多，我们每次在计算的时候要取模，并且为了速度能快点，我们用到了内置的函数修饰 @cache ，其实也就是实现了记忆化的 DFS 。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution:
	    def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:
	        @cache
	        def dfs(s, left):
	            if left == 0:
	                if s == endPos:
	                    return 1
	                return 0
	            return (dfs(s-1, left-1) + dfs(s+1, left-1)) % (10**9+7)
	        return dfs(startPos, k) % (10**9+7)

### 运行结果

	
	35 / 35 test cases passed.
	Status: Accepted
	Runtime: 2559 ms
	Memory Usage: 648.1 MB

### 原题链接

	https://leetcode.com/contest/weekly-contest-309/problems/number-of-ways-to-reach-a-position-after-exactly-k-steps/


您的支持是我最大的动力
