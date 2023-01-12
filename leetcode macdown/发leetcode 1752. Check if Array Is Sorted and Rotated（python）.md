leetcode  1752. Check if Array Is Sorted and Rotated（python）

### 描述


Given an array nums, return true if the array was originally sorted in non-decreasing order, then rotated some number of positions (including zero). Otherwise, return false.

There may be duplicates in the original array.

Note: An array A rotated by x positions results in an array B of the same length such that A[i] == B[(i+x) % A.length], where % is the modulo operation.


Example 1:


	Input: nums = [3,4,5,1,2]
	Output: true
	Explanation: [1,2,3,4,5] is the original sorted array.
	You can rotate the array by x = 3 positions to begin on the the element of value 3: [3,4,5,1,2].
	
Example 2:

	
	Input: nums = [2,1,3,4]
	Output: false
	Explanation: There is no sorted array once rotated that can make nums.

Example 3:

	Input: nums = [1,2,3]
	Output: true
	Explanation: [1,2,3] is the original sorted array.
	You can rotate the array by x = 0 positions (i.e. no rotation) to make nums.

	
Example 4:


	Input: nums = [1,1,1]
	Output: true
	Explanation: [1,1,1] is the original sorted array.
	You can rotate any number of positions to make nums.
	
Example 5:

	Input: nums = [2,1]
	Output: true
	Explanation: [1,2] is the original sorted array.
	You can rotate the array by x = 5 positions to begin on the element of value 2: [2,1].


Note:


	1 <= nums.length <= 100
	1 <= nums[i] <= 100

### 解析

根据题意，就是将 nums 从某个位置截断后，然后将两部分前后互换再拼接起来，是否得到一个非降序的列表，也就是升序或者全相等列表。思路比较简单，就是将 nums 先进行升序排序为 tmp ，然后判断 tmp 和 nums 是否相等，如果相等则返回 True ，如果不相等，则遍历 nums ，判断 nums[i:] + nums[:i] 和 tmp 是否相等，如果相等则返回 True ，如果不相等则最后返回 False 。


### 解答
				

	class Solution(object):
	    def check(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: bool
	        """
	        tmp = sorted(nums)
	        if nums == tmp:
	            return True
	        for i in range(1,len(nums)):
	            if nums[i:] + nums[:i] == tmp:
	                return True
	        return False
	            	      
			
### 运行结果

	Runtime: 20 ms, faster than 76.39% of Python online submissions for Check if Array Is Sorted and Rotated.
	Memory Usage: 13.5 MB, less than 33.33% of Python online submissions for Check if Array Is Sorted and Rotated.

### 解析

另外还有一种思路，那就是找规律，非降序有升序和全相等两种情况，那么：

* 假如  nums=[3,4,5,1,2] 真的可以经过变换变成非降序的列表  [1,2,3,4,5] ，那么肯定只有一个索引 i 导致 nums[i]>nums[i+1] ，这里是索引 2
* 假如 nums=[1,1,1] 或者 nums=[1,2,3] 那么不可能有索引导致上面的 nums[i]>nums[i+1] 现象出现
* 假如 nums=[2,1,3,4]，尽管只有一个索引 i 导致 nums[i]>nums[i+1] ，但是由于 nums[0]<nums[-1]，所以经过变化之后得到的列表 [1,3,4,2] 也不是非降序的


所以思路很清楚了：

* 用计数器 count 记录 nums[i]<nums[i+1] 出现的次数
* 如果 count 大于 1 则直接返回 False
* 如果 count==0 或者 nums[0]>=nums[-1] 则返回 True ，否则返回 False

### 解答

	class Solution(object):
	    def check(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: bool
	        """
	        count = 0
	        for i in range(1,len(nums)):
	            if nums[i-1]>nums[i]:
	                count+=1
	                if count>1:
	                    return False
	        return count==0 or nums[0]>=nums[-1]

### 运行结果

	Runtime: 24 ms, faster than 55.14% of Python online submissions for Check if Array Is Sorted and Rotated.
	Memory Usage: 13.5 MB, less than 12.15% of Python online submissions for Check if Array Is Sorted and Rotated.
	
原题链接：https://leetcode.com/problems/check-if-array-is-sorted-and-rotated/



您的支持是我最大的动力
