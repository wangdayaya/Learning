leetcode  2364. Count Number of Bad Pairs（python）




### 描述

You are given a 0-indexed integer array nums. A pair of indices (i, j) is a bad pair if i < j and j - i != nums[j] - nums[i].

Return the total number of bad pairs in nums.



Example 1:

	Input: nums = [4,1,3,3]
	Output: 5
	Explanation: The pair (0, 1) is a bad pair since 1 - 0 != 1 - 4.
	The pair (0, 2) is a bad pair since 2 - 0 != 3 - 4, 2 != -1.
	The pair (0, 3) is a bad pair since 3 - 0 != 3 - 4, 3 != -1.
	The pair (1, 2) is a bad pair since 2 - 1 != 3 - 1, 1 != 2.
	The pair (2, 3) is a bad pair since 3 - 2 != 3 - 3, 1 != 0.
	There are a total of 5 bad pairs, so we return 5.

	
Example 2:

	Input: nums = [1,2,3,4,5]
	Output: 0
	Explanation: There are no bad pairs.




Note:

	1 <= nums.length <= 10^5
	1 <= nums[i] <= 10^9


### 解析

根据题意，给定一个 0 索引的整数数组 nums 。 如果 i < j 且 j - i != nums[j] - nums[i]，则一对索引 (i, j) 是错误对。返回以 nums 为单位的错误对的总数。

其实这道题我们可以往相反的方向进行思考，加入我们知道了有多少个正确对，用总对数减去正确对数是错误对数。

	坏数对数量 = 总数对数量 - 好数对的数量
	
所以总的对数就是 N \* (N-1) // 2 ，然后我们计算正确的对数，根据公式我们知道  nums[i] - i == nums[j] - j 即可，其实 i 就是 j ，我们去计算每个元素减去其索引的结果 nums[i] - i 即为 k ，k 出现的次数为 v ，然后使用计数器统计找出出现次数大于 1 说明有能组成  v\*(v-1)//2  个正确对，我们将出现不同 k 的所有的正确对数计算出来，再使用总的对数减去所有正确的对数就能算出来错误的对数有多少。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。
### 解答

	class Solution(object):
	    def countBadPairs(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        N = len(nums)
	        result = N * (N - 1) // 2
	        cnt = collections.Counter([c - i for i, c in enumerate(nums)])
	        good_n = [v for k, v in cnt.items() if v > 1]
	        for n in good_n:
	            result -= n * (n - 1) // 2
	        return result


### 运行结果
	
	65 / 65 test cases passed.
	Status: Accepted
	Runtime: 749 ms
	Memory Usage: 36.5 MB
	Submitted: 0 minutes ago



### 原题链接

https://leetcode.com/contest/biweekly-contest-84/problems/count-number-of-bad-pairs/


您的支持是我最大的动力
