leetcode 560. Subarray Sum Equals K （python）




### 描述

Given an array of integers nums and an integer k, return the total number of continuous subarrays whose sum equals to k.





Example 1:

	Input: nums = [1,1,1], k = 2
	Output: 2

	
Example 2:


	Input: nums = [1,2,3], k = 3
	Output: 2



Note:


	1 <= nums.length <= 2 * 10^4
	-1000 <= nums[i] <= 1000
	-10^7 <= k <= 10^7

### 解析


根据题意，给定一个整数数组 nums 和一个整数 k，返回总和等于 k 的连续子数组的数量。

题目很简单，我们可以尝试暴力将所有的子数组都找一遍，如果子数组的和等于 k 直接计数器加一，最后将计数器返回即可。这个方法肯定是报错的，因为题目中的 nums.length 最大为 2 \* 10^4 ，而算法的时间复杂度为 O(N^2) ，所以肯定是超时的。这只是最简单的方法，如果限制条件能再降个数量级，那么可以优先考虑使用这个方法，毕竟简单易写。

### 解答
				

	class Solution(object):
	    def subarraySum(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        result = 0
	        N = len(nums)
	        for start in range(N):
	            tmp = 0
	            for end in range(start, N):
	                tmp += nums[end]
	                if tmp == k:
	                    result += 1
	        return result
            	      
			
### 运行结果

	Time Limit Exceeded

### 解析

其实这道题一看就是前缀和类型的题目，我们计算从索引 0 到 j 的元素的累加和 a ，然后计算从索引 0 到 i 的元素的累加和 b ，i<j ，那么 a-b 如果为 k 说明在窗口 [i+1,j] 范围的子字符串和为 k ，所以按照前缀和的思路去解题肯定错不了。



我们定义一个字典 preSum 用来保存从索引 0 开始到位置 i 的所有元素之和及其出现的次数。定义 total 来对遍历经过的所有元素的累加和，定义 result 为计数器。我们从左到右遍历 nums ，计算累加和 total ，当 total - k 在 preSum 的时候，说明出现了和为 k 的子数组，所以我们将 preSum[total - k] 加入到计数器 result 中即可。然后对字典中的 total 对应的个数加一，表示出现和为 total 的子数组次数增加了一次。遍历结束得到的 result 即为结果，这种解法的时间复杂度和空间复杂度都为  O(N) 。

需要注意的是我们在初始化 preSum 的时候要先  preSum[0] = 1 ，因为我们假定当窗口长度为 0 的时候前缀和为 0 出现的次数为 1 。

### 解答

	class Solution(object):
	    def subarraySum(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        preSum={}
	        preSum[0]=1
	        total=0
	        result=0
	        for n in nums:
	            total+=n
	            if total-k in preSum:
	                result+=preSum[total-k]
	            if total in preSum:
	                preSum[total]+=1
	            else:
	                preSum[total]=1
	                
	        return result
### 运行结果

	Runtime: 293 ms, faster than 46.69% of Python online submissions for Subarray Sum Equals K.
	Memory Usage: 16.1 MB, less than 29.65% of Python online submissions for Subarray Sum Equals K.

### 原题链接


https://leetcode.com/problems/subarray-sum-equals-k/

您的支持是我最大的动力
