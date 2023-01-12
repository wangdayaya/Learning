leetcode  2341. Maximum Number of Pairs in Array（python）




### 描述


You are given a 0-indexed integer array nums. In one operation, you may do the following:

* Choose two integers in nums that are equal.
* Remove both integers from nums, forming a pair.

The operation is done on nums as many times as possible. Return a 0-indexed integer array answer of size 2 where answer[0] is the number of pairs that are formed and answer[1] is the number of leftover integers in nums after doing the operation as many times as possible.


Example 1:

	Input: nums = [1,3,2,1,3,2,2]
	Output: [3,1]
	Explanation:
	Form a pair with nums[0] and nums[3] and remove them from nums. Now, nums = [3,2,3,2,2].
	Form a pair with nums[0] and nums[2] and remove them from nums. Now, nums = [2,2,2].
	Form a pair with nums[0] and nums[1] and remove them from nums. Now, nums = [2].
	No more pairs can be formed. A total of 3 pairs have been formed, and there is 1 number leftover in nums.


	
Example 2:

	Input: nums = [1,1]
	Output: [1,0]
	Explanation: Form a pair with nums[0] and nums[1] and remove them from nums. Now, nums = [].
	No more pairs can be formed. A total of 1 pair has been formed, and there are 0 numbers leftover in nums.


Example 3:

	Input: nums = [0]
	Output: [0,1]
	Explanation: No pairs can be formed, and there is 1 number leftover in nums.



Note:


	1 <= nums.length <= 100
	0 <= nums[i] <= 100

### 解析

给定一个 0 索引的整数数组 nums。 在一项操作中，可以执行以下操作：

* 在 nums 中选择两个相等的整数
* 从 nums 中删除这两个整数

该操作尽可能多地在 nums 上完成。 返回一个大小为 2 的 0 索引整数数组 answer ，其中 answer[0] 是删掉的数字对数，而 answer[1] 是在尽可能多地执行操作后以 nums 表示的剩余整数的数量。

这道题很简单，我们先统计一个所有数字出现的个数存放在计数器 c 中，然后遍历 c 中的每个键值对 k 和 v ，k 表示出现的数字， v 表示其出现的个数，如果 v 是奇数，说明最后肯定会剩余一个，所以将 result[1] 加一，同时数字 k 可以进行的操作次数为 v//2 ，将其加到 result[0] ，遍历结束返回 result 。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def numberOfPairs(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        c = collections.Counter(nums)
	        result = [0,0]
	        for k,v in c.items():
	            if v%2!=0:
	                result[1] += 1
	            result[0] += v//2
	        return result

### 运行结果

	128 / 128 test cases passed.
	Status: Accepted
	Runtime: 20 ms
	Memory Usage: 13.5 MB

### 原题链接

	https://leetcode.com/contest/weekly-contest-302/problems/maximum-number-of-pairs-in-array/


您的支持是我最大的动力
