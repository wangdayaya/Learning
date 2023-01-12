leetcode  1909. Remove One Element to Make the Array Strictly Increasing（python）

### 描述

Given a 0-indexed integer array nums, return true if it can be made strictly increasing after removing exactly one element, or false otherwise. If the array is already strictly increasing, return true.

The array nums is strictly increasing if nums[i - 1] < nums[i] for each index (1 <= i < nums.length).



Example 1:

	Input: nums = [1,2,10,5,7]
	Output: true
	Explanation: By removing 10 at index 2 from nums, it becomes [1,2,5,7].
	[1,2,5,7] is strictly increasing, so return true.	
	
Example 2:

	Input: nums = [2,3,1,2]
	Output: false
	Explanation:
	[3,1,2] is the result of removing the element at index 0.
	[2,1,2] is the result of removing the element at index 1.
	[2,3,2] is the result of removing the element at index 2.
	[2,3,1] is the result of removing the element at index 3.
	No resulting array is strictly increasing, so return false.

Example 3:

	Input: nums = [1,1,1]
	Output: false
	Explanation: The result of removing any element is [1,1].
	[1,1] is not strictly increasing, so return false.
	
Example 4:

	Input: nums = [1,2,3]
	Output: true
	Explanation: [1,2,3] is already strictly increasing, so return true.
	

Note:

	2 <= nums.length <= 1000
	1 <= nums[i] <= 1000

### 解析

根据题意，就是随意去除 nums 中的某一个元素之后，如果 nums 可以变成一个升序列，直接返回 True ，如果 nums 删除任意一个元素之后，都不能变成一个升序列，则返回 False 。

思路比较简单，就是去掉 nums 中的 i 个元素得到列表 tmp ，然后遍历 tmp ，当后一个元素大于前一个元素的时候，则使用计数器 conut 加一，如果遍历结束计数器的个数等于 tmp 的长度减一，则说明去掉该元素剩下的 tmp 是升序列，直接返回 True 。如果后一个元素小于等于前一个元素的时候，则直接终止该次遍历。进行 nums 中的对下一个元素去除之后所得 tmp 的判断过程，和上面一样，如果所有元素都试过仍然不能得到升序列，则返回 False 。（可能是新出来的题， 我这第一次提交就是超过了 100% 的提交答案）

### 解答
				
	class Solution(object):
	    def canBeIncreasing(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: bool
	        """
	        N = len(nums)
	        for i in range(N):
	            tmp = nums[:i]+nums[i+1:]
	            count = 0
	            for i in range(1,len(tmp)):
	                if tmp[i]<=tmp[i-1]:
	                    break
	                else:
	                    count+=1
	            if count == len(tmp)-1:
	                return True
	        return False
	        
	            
	            
            	      
			
### 运行结果

	Runtime: 740 ms, faster than 100.00% of Python online submissions for Remove One Element to Make the Array Strictly Increasing.
	Memory Usage: 13.4 MB, less than 100.00% of Python online submissions for Remove One Element to Make the Array Strictly Increasing.


原题链接：https://leetcode.com/problems/remove-one-element-to-make-the-array-strictly-increasing/


您的支持是我最大的动力
