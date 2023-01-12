leetcode 1827. Minimum Operations to Make the Array Increasing （python）

### 描述


You are given an integer array nums (0-indexed). In one operation, you can choose an element of the array and increment it by 1.

* For example, if nums = [1,2,3], you can choose to increment nums[1] to make nums = [1,3,3].
* Return the minimum number of operations needed to make nums strictly increasing.

An array nums is strictly increasing if nums[i] < nums[i+1] for all 0 <= i < nums.length - 1. An array of length 1 is trivially strictly increasing.

 


Example 1:


	Input: nums = [1,1,1]
	Output: 3
	Explanation: You can do the following operations:
	1) Increment nums[2], so nums becomes [1,1,2].
	2) Increment nums[1], so nums becomes [1,2,2].
	3) Increment nums[2], so nums becomes [1,2,3].
	
Example 2:

	Input: nums = [1,5,2,4,1]
	Output: 14

Example 3:


	Input: nums = [8]
	Output: 0
	


Note:

	1 <= nums.length <= 5000
	1 <= nums[i] <= 10^4



### 解析


根据题意，就是将 nums 经过特定操作转换为增序列，特定操作就是能够每次对某一个元素加一，求最后将 nums 转换为增序列至少需要多少次操作。很简单就是遍历所有元素，如果后一个元素小于等于前一个元素，那么将后一个元素变为前一个元素加一的大小即可，将操作次数加入 result ，遍历结束得到的 result 即为结果。

### 解答
				
	class Solution(object):
	    def minOperations(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        i = 1
	        while i < len(nums):
	            if nums[i] <= nums[i-1]:
	                increment = nums[i - 1] - nums[i] + 1
	                result += increment
	                nums[i] += increment
	            i += 1
	        return result

            	      
			
### 运行结果
	
	Runtime: 108 ms, faster than 39.07% of Python online submissions for Minimum Operations to Make the Array Increasing.
	Memory Usage: 14.1 MB, less than 20.09% of Python online submissions for Minimum Operations to Make the Array Increasing.

### 解析

思路还和上面一样的，其实还可以更加地简化一下代码过程，遍历 nums 中的每个元素，如果当前元素大于 p 直接使用 p 记录，否则将 p 增加 1 表示当前位置应该存在的最小的数字，然后用 result 来记录当前元素 增加到应该存在的最小的数字需要操作的次数，最后遍历结束得到的 result 就是结果。

### 解答

	class Solution(object):
	    def minOperations(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        p,result=0,0
	        for num in nums:
	            if p < num:
	                p = num
	            else:
	                p += 1
	                result += p - num
	        return result


### 运行结果

	Runtime: 92 ms, faster than 94.77% of Python online submissions for Minimum Operations to Make the Array Increasing.
	Memory Usage: 13.8 MB, less than 99.35% of Python online submissions for Minimum Operations to Make the Array Increasing.

原题链接：https://leetcode.com/problems/minimum-operations-to-make-the-array-increasing/



您的支持是我最大的动力
