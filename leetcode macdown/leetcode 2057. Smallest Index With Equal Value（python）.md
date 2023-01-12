leetcode  2057. Smallest Index With Equal Value（python）

### 描述


Given a 0-indexed integer array nums, return the smallest index i of nums such that i mod 10 == nums[i], or -1 if such index does not exist.

x mod y denotes the remainder when x is divided by y.


Example 1:


	Input: nums = [0,1,2]
	Output: 0
	Explanation: 
	i=0: 0 mod 10 = 0 == nums[0].
	i=1: 1 mod 10 = 1 == nums[1].
	i=2: 2 mod 10 = 2 == nums[2].
	All indices have i mod 10 == nums[i], so we return the smallest index 0.
	
Example 2:

		
	Input: nums = [4,3,2,1]
	Output: 2
	Explanation: 
	i=0: 0 mod 10 = 0 != nums[0].
	i=1: 1 mod 10 = 1 != nums[1].
	i=2: 2 mod 10 = 2 == nums[2].
	i=3: 3 mod 10 = 3 != nums[3].
	2 is the only index which has i mod 10 == nums[i].

Example 3:

	Input: nums = [1,2,3,4,5,6,7,8,9,0]
	Output: -1
	Explanation: No index satisfies i mod 10 == nums[i].

	
Example 4:

	Input: nums = [2,1,3,5,2]
	Output: 1
	Explanation: 1 is the only index with i mod 10 == nums[i].




Note:

	
	1 <= nums.length <= 100
	0 <= nums[i] <= 9

### 解析


 根据题意，就是给出来了一个从 0 开始索引的整数列表 nums ，题目要求我们返回最小的索引 i ，使 i mod 10 == nums[i] 成立，如果没有符合题意的索引，那么直接返回 -1 。
 
 x mod y 表示 x 除以 y 的余数。
 
看起来很难，但是这道题其实很简单，就是遍历 range(len(nums)) 每个索引 i ，如果 i%10 == nums[i] 直接返回 i ，否则遍历结束直接返回 -1 。

### 解答
				

	class Solution(object):
	    def smallestEqual(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        for i in range(len(nums)):
	            if i%10 == nums[i]:
	                return i
	        return -1
            	      
			
### 运行结果

	Runtime: 60 ms, faster than 100.00% of Python online submissions for Smallest Index With Equal Value.
	Memory Usage: 13.5 MB, less than 100.00% of Python online submissions for Smallest Index With Equal Value.

### 解析

还可以使用内置函数 next ，直接一行代码就可以搞定。

### 解答

	class Solution(object):
	    def smallestEqual(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        return next((i for i, x in enumerate(nums) if i%10 == x), -1)


### 运行结果

	Runtime: 52 ms, faster than 100.00% of Python online submissions for Smallest Index With Equal Value.
	Memory Usage: 13.4 MB, less than 100.00% of Python online submissions for Smallest Index With Equal Value.
原题链接：https://leetcode.com/problems/smallest-index-with-equal-value/

您的支持是我最大的动力
