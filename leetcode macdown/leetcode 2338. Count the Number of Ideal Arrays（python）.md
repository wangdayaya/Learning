leetcode 2338. Count the Number of Ideal Arrays （python）




### 描述

You are given two integers n and maxValue, which are used to describe an ideal array. A 0-indexed integer array arr of length n is considered ideal if the following conditions hold:

* Every arr[i] is a value from 1 to maxValue, for 0 <= i < n.
* Every arr[i] is divisible by arr[i - 1], for 0 < i < n.

Return the number of distinct ideal arrays of length n. Since the answer may be very large, return it modulo 10^9 + 7.



Example 1:

	Input: n = 2, maxValue = 5
	Output: 10
	Explanation: The following are the possible ideal arrays:
	- Arrays starting with the value 1 (5 arrays): [1,1], [1,2], [1,3], [1,4], [1,5]
	- Arrays starting with the value 2 (2 arrays): [2,2], [2,4]
	- Arrays starting with the value 3 (1 array): [3,3]
	- Arrays starting with the value 4 (1 array): [4,4]
	- Arrays starting with the value 5 (1 array): [5,5]
	There are a total of 5 + 2 + 1 + 1 + 1 = 10 distinct ideal arrays.

	
Example 2:

	Input: n = 5, maxValue = 3
	Output: 11
	Explanation: The following are the possible ideal arrays:
	- Arrays starting with the value 1 (9 arrays): 
	   - With no other distinct values (1 array): [1,1,1,1,1] 
	   - With 2nd distinct value 2 (4 arrays): [1,1,1,1,2], [1,1,1,2,2], [1,1,2,2,2], [1,2,2,2,2]
	   - With 2nd distinct value 3 (4 arrays): [1,1,1,1,3], [1,1,1,3,3], [1,1,3,3,3], [1,3,3,3,3]
	- Arrays starting with the value 2 (1 array): [2,2,2,2,2]
	- Arrays starting with the value 3 (1 array): [3,3,3,3,3]
	There are a total of 9 + 1 + 1 = 11 distinct ideal arrays.




Note:

	2 <= n <= 10^4
	1 <= maxValue <= 10^4


### 解析

根据题意，给定两个整数 n 和 maxValue ，用于描述理想数组。 如果满足以下条件，则长度为 n 的 0 索引整数数组 arr 被认为是理想的：

* 每个 arr[i] 是一个从 1 到 maxValue 的值，对于 0 <= i < n。
* 每个 arr[i] 都可以被 arr[i - 1] 整除，因为 0 < i < n。

返回长度为 n 的不同理想数组的数量。 由于答案可能非常大，因此以 10^9 + 7 为模返回。

怎么说呢，这道题有点超纲，我没有做出来，因为这道题不仅要分析清楚思路，而且考察的还有组合计算以及分解质因子等数学内容，我是看了[灵神的讲解](https://www.bilibili.com/video/BV1aU4y1q7BA?vd_source=66ea1dd09047312f5bc02b99f5652ac6)才明白，大家直接看视频吧。



### 解答

	class Solution:
	    def idealArrays(self, n , maxValue )  :
	        mod = 10 ** 9 + 7
	        max_k = 13
	        max_n = n + max_k
	        dp = [[0] * (max_k + 1) for _ in range(max_n)]
	        dp[0][0] = 1
	        for i in range(1, max_n):
	            dp[i][0] = 1
	            for j in range(1, min(i, max_k) + 1):
	                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % mod
	
	        ks = [[] for _ in range(maxValue + 1)]
	        for i in range(2, maxValue + 1):
	            p, x = 2, i
	            while p * p <= x:
	                if x % p == 0:
	                    k = 0
	                    while x % p == 0:
	                        k += 1
	                        x //= p
	                    ks[i].append(k)
	                p += 1
	            if x > 1:
	                ks[i].append(1)
	
	        res = 0
	        for x in range(1, maxValue + 1):
	            mul = 1
	            for k in ks[x]:
	                mul = mul * dp[n - 1 + k][k] % mod
	            res = (res + mul) % mod
	        return res


### 运行结果

	47 / 47 test cases passed.
	Status: Accepted
	Runtime: 708 ms
	Memory Usage: 20.1 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-301/problems/count-the-number-of-ideal-arrays/


您的支持是我最大的动力
