leetcode 523. Continuous Subarray Sum （python）




### 描述

Given an integer array nums and an integer k, return true if nums has a continuous subarray of size at least two whose elements sum up to a multiple of k, or false otherwise. An integer x is a multiple of k if there exists an integer n such that x = n * k. 0 is always a multiple of k.





Example 1:

	Input: nums = [23,2,4,6,7], k = 6
	Output: true
	Explanation: [2, 4] is a continuous subarray of size 2 whose elements sum up to 6.

	
Example 2:

	Input: nums = [23,2,6,4,7], k = 6
	Output: true
	Explanation: [23, 2, 6, 4, 7] is an continuous subarray of size 5 whose elements sum up to 42.
	42 is a multiple of 6 because 42 = 7 * 6 and 7 is an integer.


Example 3:

	Input: nums = [23,2,6,4,7], k = 13
	Output: false



Note:

	1 <= nums.length <= 10^5
	0 <= nums[i] <= 10^9
	0 <= sum(nums[i]) <= 2^31 - 1
	1 <= k <= 2^31 - 1


### 解析

根据题意，给定一个整数数组 num 和一个整数 k ，如果 nums 中有长度至少 2 的连续子数组，且其元素总和为 k 的倍数，则返回 true ，否则为 false。如果存在一个整数 n ，使得 x = n * k ， 则整数 x 是 k 的倍数，而且 0 始终是 k 的倍数。

像这种子数组求和的题目，一般情况可以优先试用前缀和去解题，而且这道题还需要用到一点数学知识—— 同余定理：

* 即当两个数除以某个数的余数相等的情况下，这两个数字相减后的值肯定可以被该数整除。
* 举例，8 和 13 对 5 取模，都是 3 ，那么 13 - 8 == 5 就就可以被 5 整除

我们定一个字典 d 用来保存出现过的余数及其对应的数组索引，然后在计算前缀和的同时不断判断出现的余数是否已经在 d 中出现，且两者的索引差大于等于 2 ，如果满足条件则直接返回 True 即可，由于可能存在 nums 前 N 个数字和恰好被 k 整除的情况，我们预先设置字典 {0:-1} 来规避该问题。当遍历结束之后，还没有找到直接返回 False 。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。




### 解答

	class Solution(object):
	    def checkSubarraySum(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: bool
	        """
	        d = {0: -1}
	        pre = 0
	        for i, num in enumerate(nums):
	            pre += num
	            result = pre % k
	            target = d.get(result, i)
	            if target == i:
	                d[result] = i
	            elif i - 2 >= target:
	                return True
	        return False

### 运行结果

	Runtime: 2157 ms, faster than 19.71% of Python online submissions for Continuous Subarray Sum.
	Memory Usage: 33.5 MB, less than 19.46% of Python online submissions for Continuous Subarray Sum.


### 原题链接
https://leetcode.com/problems/continuous-subarray-sum/



您的支持是我最大的动力
