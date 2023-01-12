leetcode  1. Two Sum（python）

### 描述


Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:


	Input: nums = [2,7,11,15], target = 9
	Output: [0,1]
	Output: Because nums[0] + nums[1] == 9, we return [0, 1].
	
Example 2:

	Input: nums = [3,2,4], target = 6
	Output: [1,2]


Example 3:

	Input: nums = [3,3], target = 6
	Output: [0,1]

	



Note:

	2 <= nums.length <= 10^4
	-10^9 <= nums[i] <= 10^9
	-10^9 <= target <= 10^9
	Only one valid answer exists.


### 解析
根据题意，题中给出了一个整数列表 nums 和一个整数的目标 target ，返回两个索引，要求这两个索引对应的 nums 中的元素的和为 target ，题目中已经限定了每一种输入正好对应有一种结果，并且不会用同一个元素超过一次，结果返回的顺序可以是任意的。思路比较简单：

* 初始化字典 d 用来存放每个元素及其索引
* 遍历 nums 中的每个元素 num ，如果 target-num 在 d 中直接返回 [d[target-num], i] ，否则将 d[num] = i 
* 遍历中肯定会找到答案，因为题目中已经说明了肯定会有答案

### 解答
				
	class Solution(object):
	    def twoSum(self, nums, target):
	        """
	        :type nums: List[int]
	        :type target: int
	        :rtype: List[int]
	        """
	        d = {}
	        for i,num in enumerate(nums):
	            if target-num in d:
	                return [d[target-num], i]
	            if num not in d:
	                d[num] = i
	            

            	      
			
### 运行结果

	
	Runtime: 48 ms, faster than 76.99% of Python online submissions for Two Sum.
	Memory Usage: 14.1 MB, less than 90.53% of Python online submissions for Two Sum.

原题链接：https://leetcode.com/problems/two-sum/



您的支持是我最大的动力
