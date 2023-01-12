leetcode  2376. Count Special Integers（python）




### 描述

We call a positive integer special if all of its digits are distinct. Given a positive integer n, return the number of special integers that belong to the interval [1, n].

 



Example 1:

	Input: n = 20
	Output: 19
	Explanation: All the integers from 1 to 20, except 11, are special. Thus, there are 19 special integers

	
Example 2:

	Input: n = 5
	Output: 5
	Explanation: All the integers from 1 to 5 are special.


Example 3:


	Input: n = 135
	Output: 110
	Explanation: There are 110 integers from 1 to 135 that are special.
	Some of the integers that are not special are: 22, 114, and 131.


Note:


1 <= n <= 2 * 10^9

### 解析

根据题意，如果一个整数的所有数字都是不同的，我们称它为特殊的正整数。 给定一个正整数 n ，返回属于区间 [1, n] 的特殊整数的个数。

这道题考察的是数位 DP  ，可惜我不会做，大家直接看灵神的代码吧，我解释无非就是重复一次他的话 https://leetcode.cn/problems/count-special-integers/solution/shu-wei-dp-mo-ban-by-endlesscheng-xtgx/ 。

### 解答

	class Solution:
	    def countSpecialNumbers(self, n: int) -> int:
	        s = str(n)
	        @cache
	        def f(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
	            if i == len(s):
	                return int(is_num)
	            res = 0
	            if not is_num:  
	                res = f(i + 1, mask, False, False)
	            up = int(s[i]) if is_limit else 9
	            for d in range(0 if is_num else 1, up + 1):   
	                if mask >> d & 1 == 0:  
	                    res += f(i + 1, mask | (1 << d), is_limit and d == up, True)
	            return res
	        return f(0, 0, True, False)




### 运行结果


	120 / 120 test cases passed.
	Status: Accepted
	Runtime: 580 ms
	Memory Usage: 20.5 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-306/problems/count-special-integers/


您的支持是我最大的动力
