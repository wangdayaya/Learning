leetcode  1991. Find the Middle Index in Array（python）

### 描述


Given a 0-indexed integer array nums, find the leftmost middleIndex (i.e., the smallest amongst all the possible ones).
A middleIndex is an index where nums[0] + nums[1] + ... + nums[middleIndex-1] == nums[middleIndex+1] + nums[middleIndex+2] + ... + nums[nums.length-1].
If middleIndex == 0, the left side sum is considered to be 0. Similarly, if middleIndex == nums.length - 1, the right side sum is considered to be 0.
Return the leftmost middleIndex that satisfies the condition, or -1 if there is no such index.




Example 1:


	Input: nums = [2,3,-1,8,4]
	Output: 3
	Explanation:
	The sum of the numbers before index 3 is: 2 + 3 + -1 = 4
	The sum of the numbers after index 3 is: 4 = 4	

	
Example 2:

	Input: nums = [1,-1,4]
	Output: 2
	Explanation:
	The sum of the numbers before index 2 is: 1 + -1 = 0
	The sum of the numbers after index 2 is: 0



Example 3:

	Input: nums = [2,5]
	Output: -1
	Explanation:
	There is no valid middleIndex.


	
Example 4:

	Input: nums = [1]
	Output: 0
	Explantion:
	The sum of the numbers before index 0 is: 0
	The sum of the numbers after index 0 is: 0





Note:

	1 <= nums.length <= 100
	-1000 <= nums[i] <= 1000



### 解析

根据题意，给出了一个索引从 0 开始的整数列表，返回最左的 middleIndex ，要使左右两边的子列表的和相等，特殊情况当索引为 0 的时候左边和为 0 ，或者索引为 nums.length-1 的时候右边和为 0 。如果没有合法的 middleIndex 直接返回 -1 。其实思路比较简单：

* 当 nums 的长度为 1 的时候，直接返回 0
* 从左到右遍历 nums ，当索引为 0 或者索引为 nums.length-1 的时候如果另一半的和为 0 直接返回当前索引。否则如果当两边的子列表的和相等时候直接返回当前索引
* 如果遍历结束仍然没有找到，直接返回 -1



### 解答
				

	class Solution(object):
	    def findMiddleIndex(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        if len(nums)==1:
	            return 0
	        N = len(nums)
	        for i in range(N):
	            if i == 0 and sum(nums[i+1:])==0:
	                return i
	            if i == N-1 and sum(nums[:i])==0:
	                return i
	            if sum(nums[:i]) == sum(nums[i+1:]):
	                return i
	        return -1
	            
	            
	        	      
			

            	      
			
### 运行结果
	
	Runtime: 45 ms, faster than 22.26% of Python online submissions for Find the Middle Index in Array.
	Memory Usage: 13.4 MB, less than 65.66% of Python online submissions for Find the Middle Index in Array.



原题链接：https://leetcode.com/problems/find-the-middle-index-in-array/



您的支持是我最大的动力
