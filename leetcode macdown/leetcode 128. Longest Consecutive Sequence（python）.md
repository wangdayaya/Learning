leetcode  128. Longest Consecutive Sequence（python）




### 描述

Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.





Example 1:

	Input: nums = [100,4,200,1,3,2]
	Output: 4
	Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

	
Example 2:


	Input: nums = [0,3,7,2,5,8,4,6,0,1]
	Output: 9





Note:

* 0 <= nums.length <= 10^5
* -10^9 <= nums[i] <= 10^9


### 解析


根据题意，给定一个未排序的整数数组 nums ，返回最长连续元素序列的长度。题目要求必须编写一个在 O(n) 时间内运行的算法。

先用最朴素的算法，题目要求我们找连续的最长序列，那么我们先将 nums 进行去重，然后将 nums 进行排序，然后从左往右找连续序列的最长长度即可。

时间复杂度为 O(NlogN) ，空间复杂度为 O(1) 。
### 解答
				
	class Solution(object):
	    def longestConsecutive(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        if not nums: return 0
	        nums = list(set(nums))
	        nums.sort()
	        result = 0
	        mx_L = 1
	        for i in range(1, len(nums)):
	            if nums[i]-nums[i-1] == 1:
	                mx_L += 1
	            else:
	                result = max(result, mx_L)
	                mx_L = 1
	        result = max(result, mx_L)
	        return result



            	      
			
### 运行结果

	Runtime: 374 ms, faster than 73.02% of Python online submissions for Longest Consecutive Sequence.
	Memory Usage: 29.6 MB, less than 18.22% of Python online submissions for Longest Consecutive Sequence.

### 解析

尽管上面的解法能 AC ，但是我们尽量按照题目的要求来解题，因为是要在 O(N) 时间内完成解题，所以我们只能经过一次遍历找到答案，按照最朴素的逻辑，我们知道某个数字 x ，如果要找连续序列的元素，那么肯定要在 nums 中找 x-1 或者 x+1 这两个中的一个，如果有那么连续序列的长度增加一，然后再继续按照这个思路不断扩大区间继续找.

所以我们可以使用哈希表来存储每个值的对应的连续区间长度，遍历每个数字 x ，如果 x 不在表中，我们取其左边的数字已有的左边连续区间长度 L和右边的数字已有的右边连续区间长度 R ，然后计算 L+1+R 更新结果 result ，然后更新哈希表中左边数字、x 、右边数字的最长连续序列长度。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答

	class Solution(object):
	    def longestConsecutive(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        d = {}
	        result = 0
	        for n in nums:
	            if n not in d:
	                L = d.get(n-1, 0)
	                R = d.get(n+1, 0)
	                length = L + 1 + R
	                result = max(result, length)
	                d[n] = length
	                d[n-L] = length
	                d[n+R] = length
	        return result

### 运行结果

	Runtime: 580 ms, faster than 48.27% of Python online submissions for Longest Consecutive Sequence.
	Memory Usage: 30.7 MB, less than 9.98% of Python online submissions for Longest Consecutive Sequence.
### 原题链接

https://leetcode.com/problems/longest-consecutive-sequence/


您的支持是我最大的动力
