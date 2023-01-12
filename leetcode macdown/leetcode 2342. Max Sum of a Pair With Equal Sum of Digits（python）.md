leetcode 2342. Max Sum of a Pair With Equal Sum of Digits （python）




### 描述


You are given a 0-indexed array nums consisting of positive integers. You can choose two indices i and j, such that i != j, and the sum of digits of the number nums[i] is equal to that of nums[j].

Return the maximum value of nums[i] + nums[j] that you can obtain over all possible indices i and j that satisfy the conditions.


Example 1:


	Input: nums = [18,43,36,13,7]
	Output: 54
	Explanation: The pairs (i, j) that satisfy the conditions are:
	- (0, 2), both numbers have a sum of digits equal to 9, and their sum is 18 + 36 = 54.
	- (1, 4), both numbers have a sum of digits equal to 7, and their sum is 43 + 7 = 50.
	So the maximum sum that we can obtain is 54.
	
Example 2:


	Input: nums = [10,12,19,14]
	Output: -1
	Explanation: There are no two numbers that satisfy the conditions, so we return -1.




Note:

	1 <= nums.length <= 10^5
	1 <= nums[i] <= 10^9


### 解析

根据题意，给定一个由正整数组成的索引为 0 的数组 nums 。 您可以选择两个索引 i 和 j ，使得 i != j，并且数字 nums[i] 的位数之和等于 nums[j] 的位数之和。返回 nums[i] + nums[j] 的最大值，可以在满足条件的所有可能索引 i 和 j 上获得该最大值。

这道题其实很简单，我们定义字典 d ，d 的键位数字各位数之和，d 的值为这种键对应出现的数字 n ，然后我们只需要遍历 d 中的键值对 k ，v ，如果 v 的长度大于 1， 那么我们就将 v 进行排序，找出最大的两个值之和 v[-1] + v[-2] 去和 result 进行比较取最大值更新 result 即可。遍历结束之后返回 result 即可。

时间复杂度为 O(NlogN) ，空间复杂度为 O(N) 。


### 解答

	class Solution(object):
	    def maximumSum(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = -1
	        d = collections.defaultdict(list)
	        for n in nums:
	            d[self.digitSum(n)].append(n)
	        for k,v in d.items():
	            if len(v) > 1:
	                v.sort()
	                result = max(result, v[-1] + v[-2])
	        return result
	
	
	    def digitSum(self, n):
	        if n == 0:
	            return 0
	        return self.digitSum(n//10) + n % 10

### 运行结果


	82 / 82 test cases passed.
	Status: Accepted
	Runtime: 1464 ms
	Memory Usage: 26.5 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-302/problems/max-sum-of-a-pair-with-equal-sum-of-digits/

您的支持是我最大的动力
