leetcode  1480. Running Sum of 1d Array（python）

### 描述



Given an array nums. We define a running sum of an array as runningSum[i] = sum(nums[0]…nums[i]).

Return the running sum of nums.

Example 1:

	Input: nums = [1,2,3,4]
	Output: [1,3,6,10]
	Explanation: Running sum is obtained as follows: [1, 1+2, 1+2+3, 1+2+3+4].

	
Example 2:


	Input: nums = [1,1,1,1,1]
	Output: [1,2,3,4,5]
	Explanation: Running sum is obtained as follows: [1, 1+1, 1+1+1, 1+1+1+1, 1+1+1+1+1].

Example 3:

	Input: nums = [3,1,2,10,1]
	Output: [3,4,6,16,17]



Note:

	1 <= nums.length <= 1000
	-10^6 <= nums[i] <= 10^6


### 解析

根据题意，就是给出了一个整数列表 nums ，然后定义了一个 runningSum 概念， runningSum[i] = sum(nums[0]…nums[i]) ，即第 i 个位置的 runningSum 就是从 0 到 i 位置所有元素的和，思路比较简单，直接遍历每个位置的元素，直接将前面的所有元素累加起来重新赋值给当前列表位置就可以了。


### 解答
				
	
class Solution(object):
    def runningSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for i in range(1,len(nums)):
            nums[i] += nums[i-1]
        return nums
            	      
			
### 运行结果

	
	Runtime: 28 ms, faster than 62.70% of Python online submissions for Running Sum of 1d Array.
	Memory Usage: 13.5 MB, less than 87.68% of Python online submissions for Running Sum of 1d Array.

### 解析

另外同样的思路，直接用 python 的内置函数 sum 求和即可。

### 解答

	class Solution(object):
	    def runningSum(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        result = []
	        for i in range(len(nums)):
	            result.append(sum(nums[:i+1]))
	        return result


### 运行结果

	Runtime: 44 ms, faster than 18.22% of Python online submissions for Running Sum of 1d Array.
	Memory Usage: 13.5 MB, less than 87.68% of Python online submissions for Running Sum of 1d Array.

原题链接：https://leetcode.com/problems/running-sum-of-1d-array/



您的支持是我最大的动力
