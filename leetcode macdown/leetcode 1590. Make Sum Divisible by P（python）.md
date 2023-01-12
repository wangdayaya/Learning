leetcode  1590. Make Sum Divisible by P（python）

### 描述


Given an array of positive integers nums, remove the smallest subarray (possibly empty) such that the sum of the remaining elements is divisible by p. It is not allowed to remove the whole array.

Return the length of the smallest subarray that you need to remove, or -1 if it's impossible.

A subarray is defined as a contiguous block of elements in the array.


Example 1:

	Input: nums = [3,1,4,2], p = 6
	Output: 1
	Explanation: The sum of the elements in nums is 10, which is not divisible by 6. We can remove the subarray [4], and the sum of the remaining elements is 6, which is divisible by 6.

	
Example 2:

	Input: nums = [6,3,5,2], p = 9
	Output: 2
	Explanation: We cannot remove a single element to get a sum divisible by 9. The best way is to remove the subarray [5,2], leaving us with [6,3] with sum 9.


Example 3:

	Input: nums = [1,2,3], p = 3
	Output: 0
	Explanation: Here the sum is 6. which is already divisible by 3. Thus we do not need to remove anything.

	
Example 4:

	Input: nums = [1,2,3], p = 7
	Output: -1
	Explanation: There is no way to remove a subarray in order to get a sum divisible by 7.

	
Example 5:

	Input: nums = [1000000000,1000000000,1000000000], p = 3
	Output: 0


Note:

	1 <= nums.length <= 10^5
	1 <= nums[i] <= 10^9
	1 <= p <= 10^9


### 解析


根据题意，给定一个正整数数组 nums，删除最小的子数组（可能为空），使得剩余元素的总和可以被 p 整除，但是不允许删除整个数组。返回需要删除的最小子数组的长度，如果不可能，则返回 -1。

其实整个数组的和对 p 取余的到 mod ，其实就把题转换成找出最小的子数组的和等于 mode 。让子数组的右边界为 j ，那么假如左边界为 i ，那么 sum(nums[:j])%p = r ，那么 sum(nums[:i])%p = r-mod ，那么又将题转化为了满足此公式的与 j 最近的 i 。另外用字典 d 保存好余数和索引的映射关系，可以节省计算量。需要注意的是数字可能很大，取余的时候需要注意溢出。

### 解答
				

	class Solution(object):
	    def minSubarray(self, nums, p):
	        """
	        :type nums: List[int]
	        :type p: int
	        :rtype: int
	        """
	        total = 0
	        for x in nums:
	            total = (total + x) % p
	        mod = total % p
	        if mod == 0: return 0
	        result = 10000
	        prefixMod = 0
	        d = {0:-1}
	        for i in range(len(nums)):
	            prefixMod += nums[i]
	            prefixMod %= p
	            diff = (prefixMod+p-mod)%p
	            if diff in d:
	                result = min(result, i-d[diff])
	            d[prefixMod] = i
	        if result>=len(nums): return -1
	        return result
	            
	            
	            
	            
            	      
			
### 运行结果

	Runtime: 785 ms, faster than 6.90% of Python online submissions for Make Sum Divisible by P.
	Memory Usage: 30.9 MB, less than 24.14% of Python online submissions for Make Sum Divisible by P.


原题链接：https://leetcode.com/problems/make-sum-divisible-by-p/



您的支持是我最大的动力
