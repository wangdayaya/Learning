leetcode  1524. Number of Sub-arrays With Odd Sum（python）

### 描述

Given an array of integers arr, return the number of subarrays with an odd sum.

Since the answer can be very large, return it modulo 10^9 + 7.

 



Example 1:

	Input: arr = [1,3,5]
	Output: 4
	Explanation: All subarrays are [[1],[1,3],[1,3,5],[3],[3,5],[5]]
	All sub-arrays sum are [1,4,9,3,8,5].
	Odd sums are [1,9,3,5] so the answer is 4.

	
Example 2:

	Input: arr = [2,4,6]
	Output: 0
	Explanation: All subarrays are [[2],[2,4],[2,4,6],[4],[4,6],[6]]
	All sub-arrays sum are [2,6,12,4,10,6].
	All sub-arrays have even sum and the answer is 0.


Example 3:

	
	Input: arr = [1,2,3,4,5,6,7]
	Output: 16
	




Note:

	1 <= arr.length <= 10^5
	1 <= arr[i] <= 100


### 解析


根据题意，给定一个整数数组 arr，返回具有奇数和的子数组的数量。注意由于答案可能非常大，将其取模 10^9 + 7 返回。

想解决这个题我们要知道两个规律：

* 偶数-奇数=奇数
* 奇数-偶数=奇数

我们使用类似前缀和的思想，找子数组 arr[i:j] 的和为奇数，我们可以转化两种情况：

* 第一种情况就是 sum(arr[:j]) 为奇数，本身算一个子数组组合即 result 加一，再加上前面出现的和为偶数的子数组的个数 even ，表示有 even 个 i 的位置能使 arr[i:j] 和为奇数
* 第二种情况就是 sum(arr[:j]) 为偶数，直接将 result 加上其前面出现的和为奇数的子数组的个数 odd ，表示有 odd 个 i 的位置能使 arr[i:j] 和为奇数

将上面这写结果累加起来就是最后的答案。


### 解答
				

	class Solution(object):
	    def numOfSubarrays(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: int
	        """
	        presum = arr[0]
	        odd = presum%2
	        even = 1 if presum%2==0 else 0
	        result = odd
	        for x in arr[1:]:
	            presum += x
	            if presum % 2 == 1:
	                result += 1
	                result += even
	                odd += 1
	            else:
	                result += odd
	                even += 1
	        return result%(10**9+7)
	            
            	      
			
### 运行结果

	Runtime: 1081 ms, faster than 69.09% of Python online submissions for Number of Sub-arrays With Odd Sum.
	Memory Usage: 17.2 MB, less than 92.73% of Python online submissions for Number of Sub-arrays With Odd Sum.


原题链接：https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum/



您的支持是我最大的动力
