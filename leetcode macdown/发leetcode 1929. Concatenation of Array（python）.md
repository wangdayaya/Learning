leetcode  1929. Concatenation of Array（python）

### 描述

Given an integer array nums of length n, you want to create an array ans of length 2n where ans[i] == nums[i] and ans[i + n] == nums[i] for 0 <= i < n (0-indexed).

Specifically, ans is the concatenation of two nums arrays.

Return the array ans.



Example 1:


	Input: nums = [1,2,1]
	Output: [1,2,1,1,2,1]
	Explanation: The array ans is formed as follows:
	- ans = [nums[0],nums[1],nums[2],nums[0],nums[1],nums[2]]
	- ans = [1,2,1,1,2,1]
	
Example 2:

	Input: nums = [1,3,2,1]
	Output: [1,3,2,1,1,3,2,1]
	Explanation: The array ans is formed as follows:
	- ans = [nums[0],nums[1],nums[2],nums[3],nums[0],nums[1],nums[2],nums[3]]
	- ans = [1,3,2,1,1,3,2,1]



Note:

	n == nums.length
	1 <= n <= 1000
	1 <= nums[i] <= 1000


### 解析

根据题意，就是要得到两个 nums 拼接而成的字符串并返回，直接将 nums 加两次就行，这也太简单了。。。。


### 解答
				

	class Solution(object):
	    def getConcatenation(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        return nums+nums
### 解答
				

	class Solution(object):
	    def getConcatenation(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        return nums*2            	      
			
### 运行结果
	
	Runtime: 68 ms, faster than 100.00% of Python online submissions for Concatenation of Array.
	Memory Usage: 13.8 MB, less than 100.00% of Python online submissions for Concatenation of Array.

### 解析

基本思路一样，就是将 nums 拼接到 nums 之后，还能利用 python 中的 append 函数，在 nums 列表之后，继续追加一次 nums 中的各个元素即可得到答案。


### 解答
				

	class Solution(object):
	    def getConcatenation(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        for i in range(0,len(nums)):
	            nums.append(nums[i])
	        return nums
			
### 运行结果

	Runtime: 76 ms, faster than 27.80% of Python online submissions for Concatenation of Array.
	Memory Usage: 13.6 MB, less than 93.57% of Python online submissions for Concatenation of Array.
	
### 解析

再优化的方法我也想不出来了，只能想一个最笨的方法了，就是将 nums 中的元素按照顺序加入结果列表 result ，然后再按顺序加入 result 中。不推荐这个方法，有凑字数的嫌疑我深以为不齿，但是为了凑字数不齿也没法了，幸好脸皮厚。不过看结果好像这种自认为最慢的方法却不是最慢的，只不过空间所占比较大，嘻嘻。

### 解答
				

	class Solution(object):
	    def getConcatenation(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        result = []
	        N = len(nums)
	        for i in range(0, 2*N):
	            result.append(nums[i%N])
	        return result
			
### 运行结果

	Runtime: 72 ms, faster than 53.89% of Python online submissions for Concatenation of Array.
	Memory Usage: 13.9 MB, less than 14.72% of Python online submissions for Concatenation of Array.




原题链接：https://leetcode.com/problems/concatenation-of-array



您的支持是我最大的动力
