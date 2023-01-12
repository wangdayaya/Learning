leetcode 2318. Number of Distinct Roll Sequences （python）



### 描述


You are given an integer n. You roll a fair 6-sided dice n times. Determine the total number of distinct sequences of rolls possible such that the following conditions are satisfied:

* The greatest common divisor of any adjacent values in the sequence is equal to 1.
* There is at least a gap of 2 rolls between equal valued rolls. More formally, if the value of the ith roll is equal to the value of the jth roll, then abs(i - j) > 2.
Return the total number of distinct sequences possible. Since the answer may be very large, return it modulo 10^9 + 7.

Two sequences are considered distinct if at least one element is different.


Example 1:


	Input: n = 4
	Output: 184
	Explanation: Some of the possible sequences are (1, 2, 3, 4), (6, 1, 2, 3), (1, 2, 3, 1), etc.
	Some invalid sequences are (1, 2, 1, 3), (1, 2, 3, 6).
	(1, 2, 1, 3) is invalid since the first and third roll have an equal value and abs(1 - 3) = 2 (i and j are 1-indexed).
	(1, 2, 3, 6) is invalid since the greatest common divisor of 3 and 6 = 3.
	There are a total of 184 distinct sequences possible, so we return 184.
	
Example 2:


	Input: n = 2
	Output: 22
	Explanation: Some of the possible sequences are (1, 2), (2, 1), (3, 2).
	Some invalid sequences are (3, 6), (2, 4) since the greatest common divisor is not equal to 1.
	There are a total of 22 distinct sequences possible, so we return 22.




Note:


1 <= n <= 10^4

### 解析

根据题意，给你一个整数 n 。 掷一个 6 面骰子 n 次。 确定可能的不同滚动序列的总数，以满足以下条件：

* 序列中任何相邻值的最大公约数等于 1 
* 如果按顺序第 i 次摇出的值等于第 j 次摇出的值，则 abs(i - j) > 2

返回可能的不同序列的总数。 由于答案可能非常大，因此以 10^9 + 7 为模返回。

通过两个条件我们就知道，我们假设当前的数字位 cur ，那么其前一个数字为 pre ，前一个的前一个数字为 prepre ，那么 cur 能加入当前序列的前提为 ：

* 	gcd(cur, pre) == 1
* 	j!=pre and j!=prepre
	
我们可以通过 DFS 来解题，同时加上注解 @ lru_cache 来加速。

时间复杂度为 O(6^3\*N) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution:
	    def distinctSequences(self, n: int) -> int:
	        MOD = 10 ** 9 + 7
	        @lru_cache()
	        def dfs(i, pre, prepre): 
	            if i == n:   
	                return 1
	            ans = 0
	            for j in range(1, 7):
	                if j != pre and j != prepre and gcd(j, pre) == 1:   
	                    ans = (ans + dfs(i + 1, j, pre)) % MOD
	            return ans
	        return dfs(0, 7, 7)
            	      
			
### 运行结果


	Runtime: 6356 ms, faster than 50.00% of Python3 online submissions for Number of Distinct Roll Sequences.
	Memory Usage: 29.8 MB, less than 75.00% of Python3 online submissions for Number of Distinct Roll Sequences.

### 原题链接

https://leetcode.com/contest/biweekly-contest-81/problems/number-of-distinct-roll-sequences/


您的支持是我最大的动力
