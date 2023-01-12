leetcode  665. Non-decreasing Array（python）



### 描述


Given an array nums with n integers, your task is to check if it could become non-decreasing by modifying at most one element.

We define an array is non-decreasing if nums[i] <= nums[i + 1] holds for every i (0-based) such that (0 <= i <= n - 2).

 


Example 1:


	Input: nums = [4,2,3]
	Output: true
	Explanation: You could modify the first 4 to 1 to get a non-decreasing array.
	
Example 2:

	Input: nums = [4,2,1]
	Output: false
	Explanation: You can't get a non-decreasing array by modify at most one element.




Note:


	n == nums.length
	1 <= n <= 10^4
	-10^5 <= nums[i] <= 10^5

### 解析

根据题意，给定一个具有 n 个整数的数组 nums ，要求我们检查它是否可以通过最多修改一个元素来变为非递减。题目定义一个数组是非递减的，如果 nums[i] <= nums[i + 1] 对于每个 i（从 0 开始）都成立。

假如有三个例子，如下：

	[4,2,5]  只能将 4 变为 2 或者将 2 改为 4、5
	[1,4,2,5] 只能将 4 改为 1 、2 ，将 2 改为 4、5
	[3,4,2,5] 只能将 2 改为 4、5

为了保证单调性，我们尽量去修改前面的元素，少动后面的元素，使得整个数组能单调非递增，通过上面的例子我们发现，遍历 nums 时每当 nums[i] 破坏了单调性，也就是 nums[i-1] > nums[i] ，我们使用 result 计数加一，此时如果 i 为 1 或者 nums[i] >= nums[i-2]，我们尽量修改 nums[i-1] 为 nums[i] 。当 i 大于 1 且 nums[i] < nums[i-2] ，为了维持单调性只能修改 nums[i] 为 nums[i-1] 。最后只需要判断 result 是否小于等于 1 即可。

时间复杂度为 O(N) ，空间复杂度为 O(1)。

### 解答
				
	class Solution(object):
	    def checkPossibility(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: bool
	        """
	        N = len(nums)
	        result = 0
	        for i in range(1,N):
	            if nums[i-1] > nums[i]:
	                result += 1
	                if i == 1 or nums[i] >= nums[i-2]:
	                    nums[i-1] = nums[i]
	                else:
	                    nums[i] = nums[i-1]
	        return result <= 1

            	      
			
### 运行结果


	Runtime: 162 ms, faster than 79.75% of Python online submissions for Non-decreasing Array.
	Memory Usage: 14.8 MB, less than 37.67% of Python online submissions for Non-decreasing Array.

### 原题链接

https://leetcode.com/problems/non-decreasing-array/

您的支持是我最大的动力
