leetcode  2354. Number of Excellent Pairs（python）




### 描述

You are given a 0-indexed positive integer array nums and a positive integer k. A pair of numbers (num1, num2) is called excellent if the following conditions are satisfied:

* Both the numbers num1 and num2 exist in the array nums.
* The sum of the number of set bits in num1 OR num2 and num1 AND num2 is greater than or equal to k, where OR is the bitwise OR operation and AND is the bitwise AND operation.

Return the number of distinct excellent pairs. Two pairs (a, b) and (c, d) are considered distinct if either a != c or b != d. For example, (1, 2) and (2, 1) are distinct. Note that a pair (num1, num2) such that num1 == num2 can also be excellent if you have at least one occurrence of num1 in the array.

 



Example 1:

	Input: nums = [1,2,3,1], k = 3
	Output: 5
	Explanation: The excellent pairs are the following:
	- (3, 3). (3 AND 3) and (3 OR 3) are both equal to (11) in binary. The total number of set bits is 2 + 2 = 4, which is greater than or equal to k = 3.
	- (2, 3) and (3, 2). (2 AND 3) is equal to (10) in binary, and (2 OR 3) is equal to (11) in binary. The total number of set bits is 1 + 2 = 3.
	- (1, 3) and (3, 1). (1 AND 3) is equal to (01) in binary, and (1 OR 3) is equal to (11) in binary. The total number of set bits is 1 + 2 = 3.
	So the number of excellent pairs is 5.

	
Example 2:


	Input: nums = [5,1,1], k = 10
	Output: 0
	Explanation: There are no excellent pairs for this array.




Note:



	1 <= nums.length <= 10^5
	1 <= nums[i] <= 10^9
	1 <= k <= 60

### 解析

根据题意，给定一个 0 索引的正整数数组 nums 和一个正整数 k 。返回不同的优秀数字对的数量。  如果满足以下条件，则一对数字 (num1, num2) 称为优秀的：

* 数字 num1 和 num2 都存在于数组 nums 中。
* num1 OR num2 和 num1 AND num2 结果中 1 的个数大于等于 k ，其中 OR 为按位或运算，AND 为按位与运算。

这道题就是找规律题，我们计算 3 和 5 的按位或运算和按位与运算：

	9	5	or	and
	1	0	1	0
	0	1	1	0
	0	0	0	0
	1	1	1	1
	
从上面可以发现其实题目中要求的运算可以转化为两个数字的二进制表示中出现的 1 的个数之和。我们先将 set(nums) 中的每种数字算出其自身的二进制表示的 1 的个数，然后我们对其进行计数为 counter ，counter 中的 key 是可能出现的 1 的个数，value 表示的是能出现二进制中 1 的个数为 key 的不同十进制数字的个数。因为我们要找不同的数字对，所以我们只要对 counter 中不同的键 i 和键 j 进行两两比较，如果 i+j 大于等于 k ，那么说明满足题意，我们可以构成的不同的数字对有 counter[i] \* counter[j]  个，加入到结果 result 中即可。

函数 c 的时间复杂度基本是常数级别，主要的时间耗在了双重循环上面，时间复杂度为 O(N^2) ，空间复杂度为 O(N) 。



### 解答

	class Solution(object):
	    def countExcellentPairs(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        def c(n):
	            result = 0
	            while n != 0:
	                n &= n-1
	                result += 1
	            return result
	        s = set(nums)
	        counter = collections.Counter(c(n) for n in s)
	        result = 0
	        for i in counter:
	            for j in counter:
	                if i + j >= k:
	                    result += counter[i] * counter[j]
	        return result

### 运行结果

	56 / 56 test cases passed.
	Status: Accepted
	Runtime: 1488 ms
	Memory Usage: 31.7 MB
	Submitted: 0 minutes ago


### 原题链接

https://leetcode.com/contest/weekly-contest-303/problems/number-of-excellent-pairs/


您的支持是我最大的动力
