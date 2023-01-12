leetcode 2401. Longest Nice Subarray （python）




### 描述

You are given an array nums consisting of positive integers. We call a subarray of nums nice if the bitwise AND of every pair of elements that are in different positions in the subarray is equal to 0.

Return the length of the longest nice subarray. A subarray is a contiguous part of an array. Note that subarrays of length 1 are always considered nice.



Example 1:

	Input: nums = [1,3,8,48,10]
	Output: 3
	Explanation: The longest nice subarray is [3,8,48]. This subarray satisfies the conditions:
	- 3 AND 8 = 0.
	- 3 AND 48 = 0.
	- 8 AND 48 = 0.
	It can be proven that no longer nice subarray can be obtained, so we return 3.

	
Example 2:

	Input: nums = [3,1,5,11,13]
	Output: 1
	Explanation: The length of the longest nice subarray is 1. Any subarray of length 1 can be chosen.





Note:

	1 <= nums.length <= 10^5
	1 <= nums[i] <= 10^9


### 解析

根据题意，给定一个由正整数组成的数组 nums 。 如果子数组中不同位置的每对元素的按位与等于 0，我们称 nums 子数组为 nice 的。返回最长的 nice 子数组的长度。 子数组是数组的连续部分。 请注意长度为 1 的子数组总是被认为是好的。

其实这道题的关键就是考察位运算，我这里参照的时[灵神的思路](https://leetcode.cn/problems/longest-nice-subarray/solution/bao-li-mei-ju-pythonjavacgo-by-endlessch-z6t6/)，首先我们通过题意找出一个关键信息，因为 nums[i] 最大限制范围为 10^9 ，按照二进制换算也就是最多有 30 个二进制位，然后我们根据按位与运算为 0 的这个条件就知道，最长的优雅子数组长度为 30 ，因为要使每个二进制位上最多有一个 1 出现，而每个 nums[i] 都是正整数，所以最多子数组的长度最多有 30 个。这个结论得出就可以知道我们的算法可以暴力解决，对于 nums 中的每个位置，我们向前找子数组即可，不用担心会超时成为时间复杂度 O(N^2) 的运算。

另外我们想要解决子数组中每对元素的按位与都为 0 ，其实这里有个小技巧，只要我们将子数组中的所有元素都进行按位或运算得到 cur ，那么只要来了一个新的元素 nums[j] ，只要 cur & nums[j] 等于 0 就相当于 nums[j] 和子数组中的所有元素的按位与运算都为 0 。

时间复杂度为 O(N * 30)，空间复杂度为 (1) 。

### 解答

	class Solution(object):
	    def longestNiceSubarray(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        for i, c in enumerate(nums):
	            j = i - 1
	            while j >= 0 and c & nums[j] == 0:
	                c |= nums[j]
	                j -= 1
	            result = max(result, i - j)
	        return result

### 运行结果


	61 / 61 test cases passed.
	Status: Accepted
	Runtime: 1738 ms
	Memory Usage: 25.5 MB

### 原题链接

	https://leetcode.com/contest/weekly-contest-309/problems/longest-nice-subarray/


您的支持是我最大的动力
