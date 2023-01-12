leetcode  228. Summary Ranges（python）




### 描述

You are given a sorted unique integer array nums.

Return the smallest sorted list of ranges that cover all the numbers in the array exactly. That is, each element of nums is covered by exactly one of the ranges, and there is no integer x such that x is in one of the ranges but not in nums.

Each range [a,b] in the list should be output as:

* "a->b" if a != b
* "a" if a == b




Example 1:


	Input: nums = [0,1,2,4,5,7]
	Output: ["0->2","4->5","7"]
	Explanation: The ranges are:
	[0,2] --> "0->2"
	[4,5] --> "4->5"
	[7,7] --> "7"
	
Example 2:

	Input: nums = [0,2,3,4,6,8,9]
	Output: ["0","2->4","6","8->9"]
	Explanation: The ranges are:
	[0,0] --> "0"
	[2,4] --> "2->4"
	[6,6] --> "6"
	[8,9] --> "8->9"






Note:


	0 <= nums.length <= 20
	-2^31 <= nums[i] <= 2^31 - 1
	All the values of nums are unique.
	nums is sorted in ascending order.

### 解析

根据题意，给定一个排序的唯一整数数组 nums。返回精确覆盖数组中所有数字的最小排序范围列表。 也就是说，nums 的每个元素都被恰好一个范围所覆盖。列表中的每个范围 [a,b] 应输出为：

* "a->b" 如果 a != b
* "a" 如果 a == b

一开始我是先遍历了一次，把范围存到列表中，然后在遍历列表进行结果相应格式化的转化，但是代码又臭又长，不太好看，所以干脆优化了代码，打算一次性把结果生成放入结果列表中。

我们先初始化一个索引 start ，用来表示某个范围的起始索引，然后从第二个元素开始遍历，如果当前索引等于 nums 长度 ，说明最后一个元素自成一个范围，直接将其变为字符串加入结果列表 result 中。或者当后一个元素与前一个元素的差不为 1 ，说明范围断开，将 [nums[start], nums[i-1]] 变换为相应的字符串个是加入结果列表 result 中，同时更新 start 的值为当前索引。遍历 nums 结束直接返回 result 。

### 解答
				
	class Solution(object):
	    def summaryRanges(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[str]
	        """
	        start = 0
	        result = []
	        for i in range(1, len(nums)+1):
	            if i==len(nums) or nums[i] - nums[i-1] != 1:
	                result.append( '->'.join(map(str,[nums[start], nums[i-1]])) if nums[start]!=nums[i-1] else str(nums[i-1]))
	                start = i
	        return result
        

### 运行结果

	Runtime: 31 ms, faster than 19.57% of Python online submissions for Summary Ranges.
	Memory Usage: 13.4 MB, less than 85.60% of Python online submissions for Summary Ranges.


### 原题链接

https://leetcode.com/problems/summary-ranges/


您的支持是我最大的动力
