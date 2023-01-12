leetcode  2343. Query Kth Smallest Trimmed Number（python）




### 描述


You are given a 0-indexed array of strings nums, where each string is of equal length and consists of only digits.

You are also given a 0-indexed 2D integer array queries where queries[i] = [ki, trimi]. For each queries[i], you need to:

* Trim each number in nums to its rightmost trimi digits.
* Determine the index of the kith smallest trimmed number in nums. If two trimmed numbers are equal, the number with the lower index is considered to be smaller.
* Reset each number in nums to its original length.

Return an array answer of the same length as queries, where answer[i] is the answer to the ith query.

Note:

* To trim to the rightmost x digits means to keep removing the leftmost digit, until only x digits remain.
* Strings in nums may contain leading zeros.


Example 1:

	Input: nums = ["102","473","251","814"], queries = [[1,1],[2,3],[4,2],[1,2]]
	Output: [2,2,1,0]
	Explanation:
	1. After trimming to the last digit, nums = ["2","3","1","4"]. The smallest number is 1 at index 2.
	2. Trimmed to the last 3 digits, nums is unchanged. The 2nd smallest number is 251 at index 2.
	3. Trimmed to the last 2 digits, nums = ["02","73","51","14"]. The 4th smallest number is 73.
	4. Trimmed to the last 2 digits, the smallest number is 2 at index 0.
	   Note that the trimmed number "02" is evaluated as 2.

	
Example 2:


	Input: nums = ["24","37","96","04"], queries = [[2,1],[2,2]]
	Output: [3,0]
	Explanation:
	1. Trimmed to the last digit, nums = ["4","7","6","4"]. The 2nd smallest number is 4 at index 3.
	   There are two occurrences of 4, but the one at index 0 is considered smaller than the one at index 3.
	2. Trimmed to the last 2 digits, nums is unchanged. The 2nd smallest number is 24.




Note:

	1 <= nums.length <= 100
	1 <= nums[i].length <= 100
	nums[i] consists of only digits.
	All nums[i].length are equal.
	1 <= queries.length <= 100
	queries[i].length == 2
	1 <= ki <= nums.length
	1 <= trimi <= nums[i].length


### 解析

根据题意，给定一个 0 索引的字符串 nums 数组，其中每个字符串的长度相等且仅由数字组成。给定一个 0 索引的 2D 整数数组 queries ，其中 queries[i] = [ki, trimi]。 对于每个 queries[i]，您需要做以下的操作：

* 将 nums 中的每个数字修剪到剩下最右边的 trimi 个数字。
* 确定此时 nums 中第 k 个最小数的索引。 如果两个修剪后的数字相等，则认为具有较低索引的数字较小。
* 将 nums 中的每个数字重置为其原始模样。

返回与查询长度相同的数组答案，其中 answer[i] 是第 i 个查询的答案。需要注意的是：

* 修剪到最右边的 x 个数字意味着不断删除最左边的数字，直到只剩下 x 个数字。
* nums 中的字符串可能包含前导零。

这道题很简单，只需要按照题意写代码即可。

时间复杂度为 O(Q\*M\*NlogN)) ，空间复杂度为 O(N) ，其中 Q 是 queries 的长度，N 时 nums 的长度，M 是 nums[i] 的长度 。

### 解答
	class Solution(object):
	    def smallestTrimmedNumbers(self, nums, queries):
	        """
	        :type nums: List[str]
	        :type queries: List[List[int]]
	        :rtype: List[int]
	        """
	        result = []
	        for k,t in queries:
	            tmp = [[n[-t:], i] for i,n in enumerate(nums)]
	            tmp.sort()	
	            result.append(tmp[k-1][1])
	        return result


### 运行结果

	251 / 251 test cases passed.
	Status: Accepted
	Runtime: 1299 ms
	Memory Usage: 13.6 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-302/problems/query-kth-smallest-trimmed-number/


您的支持是我最大的动力
