
leetcode  2104. Sum of Subarray Ranges（python）
### 描述

You are given an integer array nums. The range of a subarray of nums is the difference between the largest and smallest element in the subarray.

Return the sum of all subarray ranges of nums.

A subarray is a contiguous non-empty sequence of elements within an array.


Follow-up: Could you find a solution with O(n) time complexity?

Example 1:

	Input: nums = [1,2,3]
	Output: 4
	Explanation: The 6 subarrays of nums are the following:
	[1], range = largest - smallest = 1 - 1 = 0 
	[2], range = 2 - 2 = 0
	[3], range = 3 - 3 = 0
	[1,2], range = 2 - 1 = 1
	[2,3], range = 3 - 2 = 1
	[1,2,3], range = 3 - 1 = 2
	So the sum of all ranges is 0 + 0 + 0 + 1 + 1 + 2 = 4.

	
Example 2:


	Input: nums = [1,3,3]
	Output: 4
	Explanation: The 6 subarrays of nums are the following:
	[1], range = largest - smallest = 1 - 1 = 0
	[3], range = 3 - 3 = 0
	[3], range = 3 - 3 = 0
	[1,3], range = 3 - 1 = 2
	[3,3], range = 3 - 3 = 0
	[1,3,3], range = 3 - 1 = 2
	So the sum of all ranges is 0 + 0 + 0 + 2 + 0 + 2 = 4.

Example 3:

	Input: nums = [4,-2,-3,4,1]
	Output: 59
	Explanation: The sum of all subarray ranges of nums is 59.

	


Note:

	1 <= nums.length <= 1000
	-10^9 <= nums[i] <= 10^9


### 解析

根据题意，给定一个整数数组 nums。 找出 nums 所有子数组中最大和最小元素之间的差值，最后返回 nums 的所有子数组差值的总和。题目还给有能力的同学提出了更高的要求，能否找到时间复杂度为 O(n) 的解决方案。

结合题目的限制条件，我们会发现条件很宽松，使用暴力也是可以通过的，也就是双层循环，找出所有的子数组，然后按照题意找出每个子数组中的最大值和最小值的差值，然后将所有的差值都加起来即可，思路比较简单清晰，就是耗时会长一点但也能通过，毕竟最多也只有 1000 个数字，时间复杂度也只是 O(n^2)。

### 解答
				

	class Solution(object):
	    def subArrayRanges(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        N = len(nums)
	        result = 0
	        for i in range(N):
	            mn = nums[i] 
	            mx = nums[i]
	            for j in range(i, N):
	                mn = min(mn, nums[j])
	                mx = max(mx, nums[j])
	                result += mx - mn
	        return result
	                
            	      
			
### 运行结果


	Runtime: 3172 ms, faster than 44.49% of Python online submissions for Sum of Subarray Ranges.
	Memory Usage: 13.7 MB, less than 27.54% of Python online submissions for Sum of Subarray Ranges.


### 解析

其实换一个思路就是，这个题要求的就是所有子数组的最大值和减去所有子数组的最小值和的的结果，所以我们可以使用单调栈思想。

对于每个元素 nums[i] ，找出以自己作为最大值的子数组个数 a 以及以自己作为最小值子数组个数 b ，得到的 nums[i]\*a -nums[i]\*b ，每个元素都按照这个方法计算将结果相加即可。我们提前使用单调栈找出每个元素的 preSmaller 所在位置 l 和 nextSmaller 所在位置 r ，这样的子数组个数有 a=(i-l)\*(r-i) 。同理找出每个元素的 preGreater 所在位置 l 和 nextGreater 所在位置 r ，这样的子数组有 b=(i-l)\*(r-i) 。如果区间内有多个相同的值，我们约定最右边的才是最大值。

可以看出这种 O(n) 的解法速度上快了很多，看来大多数人都是直接使用暴力解题。

### 解答

	class Solution(object):
	    def subArrayRanges(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        N = len(nums)
	        
	        stack = []
	        nextSmaller = [N for _ in range(N)]
	        for i in range(N):
	            while stack and nums[stack[-1]] > nums[i]:
	                nextSmaller[stack.pop()] = i
	            stack.append(i)
	            
	        stack = []
	        preSmaller = [-1 for _ in range(N)]
	        for i in range(N-1, -1, -1):
	            while stack and nums[stack[-1]] >= nums[i]:
	                preSmaller[stack.pop()] = i
	            stack.append(i)
	            
	        stack = []
	        nextGreater = [N for _ in range(N)]
	        for i in range(N):
	            while stack and nums[stack[-1]] < nums[i]:
	                nextGreater[stack.pop()] = i
	            stack.append(i)
	            
	        stack = []
	        preGreater = [-1 for _ in range(N)]
	        for i in range(N-1,-1,-1):
	            while stack and nums[stack[-1]] <= nums[i]:
	                preGreater[stack.pop()] = i
	            stack.append(i)
	            
	        result = 0
	        for i in range(N):
	            l = preGreater[i]
	            r = nextGreater[i]
	            result += nums[i]*(i-l)*(r-i)
	            
	        for i in range(N):
	            l = preSmaller[i]
	            r = nextSmaller[i]
	            result -= nums[i]*(i-l)*(r-i)
	            
	        return result



### 运行结果

	Runtime: 96 ms, faster than 91.53% of Python online submissions for Sum of Subarray Ranges.
	Memory Usage: 13.7 MB, less than 54.24% of Python online submissions for Sum of Subarray Ranges.



原题链接：https://leetcode.com/problems/sum-of-subarray-ranges/



您的支持是我最大的动力
