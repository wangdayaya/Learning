leetcode  1437. Check If All 1's Are at Least Length K Places Away（python）

### 描述


Given an array nums of 0s and 1s and an integer k, return True if all 1's are at least k places away from each other, otherwise return False.




Example 1:

![avatar](https://assets.leetcode.com/uploads/2020/04/15/sample_1_1791.png)

	Input: nums = [1,0,0,0,1,0,0,1], k = 2
	Output: true
	Explanation: Each of the 1s are at least 2 places away from each other.

	
Example 2:

![avatar](https://assets.leetcode.com/uploads/2020/04/15/sample_2_1791.png)

	Input: nums = [1,0,0,1,0,1], k = 2
	Output: false
	Explanation: The second 1 and third 1 are only one apart from each other.

Example 3:


	Input: nums = [1,1,1,1,1], k = 0
	Output: true
	
Example 4:


	Input: nums = [0,1,0,1], k = 1
	Output: true
	


Note:

	1 <= nums.length <= 10^5
	0 <= k <= nums.length
	nums[i] is 0 or 1


### 解析


根据题意，就是判断 nums 中的 1 是否中间至少有 k 个 0 。这个题有点坑，题意有点没说明白，导致我两次提交错误，其实当 nums 为全 0 或者只有 1 个 1 的时候，这也算是 True ，卧槽无情。其他的就是正常的判断，遍历 nums ，找出两个为 1 的元素的索引差和 k 比较一下即可，不满足题意直接返回 False ，否则遍历结束之后直接返回 True 。

### 解答
				

	class Solution(object):
	    def kLengthApart(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: bool
	        """
	        if sum(nums)<=1:
	            return True
	        start = nums.index(1)
	        end = nums.index(1,start+1)
	        if end-start<k+1:
	            return False
	        index = end+1
	        while index<len(nums):
	            if nums[index]==1:
	                if index-end<k+1:
	                    return False
	                end = index
	            index += 1
	        return True
            	      
			
### 运行结果


	Runtime: 468 ms, faster than 95.71% of Python online submissions for Check If All 1's Are at Least Length K Places Away.
	Memory Usage: 15.9 MB, less than 82.14% of Python online submissions for Check If All 1's Are at Least Length K Places Away.
	
### 解析


上面的代码其实比较乱，我又整理了一下，原理和上面一样，比较简洁易懂一点。

### 解答
				
	class Solution(object):
	    def kLengthApart(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: bool
	        """
	        count = k
	        for num in nums:
	            if num == 1:
	                if count < k:
	                    return False
	                count = 0
	            else:
	                count += 1
	        return True
	

	
            	      
			
### 运行结果

	Runtime: 468 ms, faster than 91.23% of Python online submissions for Check If All 1's Are at Least Length K Places Away.
	Memory Usage: 15.8 MB, less than 78.95% of Python online submissions for Check If All 1's Are at Least Length K Places Away.
	
原题链接：https://leetcode.com/problems/check-if-all-1s-are-at-least-length-k-places-away/



您的支持是我最大的动力
