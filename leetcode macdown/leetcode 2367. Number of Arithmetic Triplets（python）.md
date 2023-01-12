leetcode  2367. Number of Arithmetic Triplets（python）




### 描述

You are given a 0-indexed, strictly increasing integer array nums and a positive integer diff. A triplet (i, j, k) is an arithmetic triplet if the following conditions are met:

* i < j < k,
* nums[j] - nums[i] == diff, and
* nums[k] - nums[j] == diff.

Return the number of unique arithmetic triplets.



Example 1:

	Input: nums = [0,1,4,6,7,10], diff = 3
	Output: 2
	Explanation:
	(1, 2, 4) is an arithmetic triplet because both 7 - 4 == 3 and 4 - 1 == 3.
	(2, 4, 5) is an arithmetic triplet because both 10 - 7 == 3 and 7 - 4 == 3. 

	
Example 2:
	
	Input: nums = [4,5,6,7,8,9], diff = 2
	Output: 2
	Explanation:
	(0, 2, 4) is an arithmetic triplet because both 8 - 6 == 2 and 6 - 4 == 2.
	(1, 3, 5) is an arithmetic triplet because both 9 - 7 == 2 and 7 - 5 == 2.




Note:


	3 <= nums.length <= 200
	0 <= nums[i] <= 200
	1 <= diff <= 50
	nums is strictly increasing.

### 解析

根据题意，给定一个 0 索引、严格递增的整数数组 nums 和一个正整数 diff 。 如果满足以下条件，则三元组 (i, j, k) 是算术三元组，返回所有不同的算术三元组的数量。：

* i < j < k 
* nums[j] - nums[i] == diff
* nums[k] - nums[j] == diff 

其实这道题目说过 nums 元素严格递增，这里面已经暗示了所有的 nums 元素都是不同的，然后我们只需要遍历 nums 中的每个元素 x ，如果 x-diff 存在于 nums 中并且 x+diff 存在于 nums 中，那就说明这个三元组是唯一的，结果 result 加一即可，遍历结束返回 result 。

时间复杂度为 O(N^2) ，空间复杂度为 O(1) 。

当然了，因为我们知道了 nums 中的元素都不同，为了提高速度，我们可以把 nums 变成集合，这样我们在查找元素的时候时间复杂度为 O(1) ，整体的时间复杂度变成了 O(N) 。
### 解答

	class Solution(object):
	    def arithmeticTriplets(self, nums, diff):
	        """
	        :type nums: List[int]
	        :type diff: int
	        :rtype: int
	        """
	        result = 0
	        for x in nums:
	            if x-diff in nums and x+diff in nums:
	                result += 1
	        return result

### 运行结果


	
	104 / 104 test cases passed.
	Status: Accepted
	Runtime: 24 ms
	Memory Usage: 13.4 MB
### 原题链接

https://leetcode.com/contest/weekly-contest-305/problems/number-of-arithmetic-triplets/


您的支持是我最大的动力
