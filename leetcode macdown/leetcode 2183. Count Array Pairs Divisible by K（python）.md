leetcode 2183. Count Array Pairs Divisible by K （python）

### 前言

这是 Weekly Contest 281 的第四题，难度 Hard ，考查的就是对数学除数的理解，难度也不是很难，但是比赛的时候超时了，烦。


### 描述

Given a 0-indexed integer array nums of length n and an integer k, return the number of pairs (i, j) such that:

* 0 <= i < j <= n - 1 and
* nums[i] * nums[j] is divisible by k.



Example 1:

	Input: nums = [1,2,3,4,5], k = 2
	Output: 7
	Explanation: 
	The 7 pairs of indices whose corresponding products are divisible by 2 are
	(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (2, 3), and (3, 4).
	Their products are 2, 4, 6, 8, 10, 12, and 20 respectively.
	Other pairs such as (0, 2) and (2, 4) have products 3 and 15 respectively, which are not divisible by 2.    

	
Example 2:

	Input: nums = [1,2,3,4], k = 5
	Output: 0
	Explanation: There does not exist any pair of indices whose corresponding product is divisible by 5.



Note:

	1 <= nums.length <= 10^5
	1 <= nums[i], k <= 10^5


### 解析

根据题意，给定一个长度为 n 的 0 索引整数数组 nums 和一个整数 k，返回对 (i, j) 的数量，使得：

* 0 <= i < j <= n - 1 
* nums[i] * nums[j] 可以被 k 整除


我当时比赛的时候也没什么思路，打算直接暴力求解，但是因为 nums 的长度最长可能为 10^5 ，并且 nums[i] 和 k 的最大值也可能是 10^5 ，导致形成的候选 (i,j) 对暴增，并且在求整除的时候计算量也很大，肯定是超时的，只能放弃了。

n 如果能被 k 整除，那么能整除 k 的其他整数 divisors 肯定也能整除 n 。所以我们找到了能整除 k 的整数列表 divisors 。我们遍历 nums 中的每个数字 nums[i] ：

* 	计算 k // math.gcd(k, nums[i]) ，很明显这个结果肯定是能整除 k 的并且存在于 divisors 中的数字
* 	 counter 记录了之前已经出现过的 remainder 的次数，这也就是满足条件的 (i,j) 对个数，将其计入到结果 result 中即可
* 	 遍历 divisors 中的每一个除数 divisor ，如果 nums[i] 能被 divisor 整除，则 count[divisors] 加一

遍历结束，最后将 result 返回即可。时间复杂度为 O(N\*sqrt(K)) ，空间复杂度为 O(sqrt(K)) 。



### 解答
				
	class Solution:
	    def countPairs(self, nums: List[int], k: int) -> int:
	        N = len(nums)
	        result = 0
	        divisors = []
	        counter = Counter()
	        for i in range(1, k + 1):
	            if k % i == 0:
	                divisors.append(i)
	        for i in range(0, N):
	            remainder = k // math.gcd(k, nums[i])
	            result += counter[remainder]
	            for divisor in divisors:
	                if nums[i] % divisor == 0:
	                    counter[divisor] += 1
	        return result

            	      
			
### 运行结果


	115 / 115 test cases passed.
	Status: Accepted
	Runtime: 3403 ms
	Memory Usage: 28 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-281/problems/count-array-pairs-divisible-by-k/

您的支持是我最大的动力
