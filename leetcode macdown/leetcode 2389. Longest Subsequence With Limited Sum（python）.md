leetcode  2389. Longest Subsequence With Limited Sum（python）




### 描述

You are given an integer array nums of length n, and an integer array queries of length m. Return an array answer of length m where answer[i] is the maximum size of a subsequence that you can take from nums such that the sum of its elements is less than or equal to queries[i].

A subsequence is an array that can be derived from another array by deleting some or no elements without changing the order of the remaining elements.



Example 1:


	Input: nums = [4,5,2,1], queries = [3,10,21]
	Output: [2,3,4]
	Explanation: We answer the queries as follows:
	- The subsequence [2,1] has a sum less than or equal to 3. It can be proven that 2 is the maximum size of such a subsequence, so answer[0] = 2.
	- The subsequence [4,5,1] has a sum less than or equal to 10. It can be proven that 3 is the maximum size of such a subsequence, so answer[1] = 3.
	- The subsequence [4,5,2,1] has a sum less than or equal to 21. It can be proven that 4 is the maximum size of such a subsequence, so answer[2] = 4.
	
Example 2:

	Input: nums = [2,3,4,5], queries = [1]
	Output: [0]
	Explanation: The empty subsequence is the only subsequence that has a sum less than or equal to 1, so answer[0] = 0.





Note:

	n == nums.length
	m == queries.length
	1 <= n, m <= 1000
	1 <= nums[i], queries[i] <= 10^6


### 解析

根据题意，给定一个长度为 n 的整数数组 nums ，以及一个长度为 m 的整数数组 queries 。 返回长度为 m 的数组答案，其中 answer[i] 是可以从 nums 中获取的子序列的最大长度，使得其元素的总和小于或等于 query[i] 。子序列是一个数组，可以通过从另一个数组中删除一些元素或不删除元素而不改变剩余元素的顺序派生出来。

其实这道题就是排序，因为是要找最长的子序列，所以我们可以先将 nums 按照从小到大的顺序进行排序，然后我们求出 nums 的前缀和，这样如果 queries[i] 刚好大于等于某个前缀和，这个时候找出来的子序列肯定是最长的，我们找出来的长度加入到结果 result 中即可。遍历 queries 之后，我们返回 result 即可。

时间复杂度为 O(NlogN) ，空间复杂度为 O(N) 。



### 解答

	class Solution(object):
	    def answerQueries(self, nums, queries):
	        """
	        :type nums: List[int]
	        :type queries: List[int]
	        :rtype: List[int]
	        """
	        nums.sort()
	        presum = [nums[0]]
	        for n in nums[1:]:
	            presum.append(presum[-1] + n)
	        result = []
	        for q in queries:
	            i = bisect.bisect(presum, q)
	            result.append(i)
	        return result

### 运行结果

	57 / 57 test cases passed.
	Status: Accepted
	Runtime: 98 ms
	Memory Usage: 13.8 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-308/problems/longest-subsequence-with-limited-sum/


您的支持是我最大的动力
